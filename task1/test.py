import os
import math
import librosa
import torch
from pydub import AudioSegment
from datasets import Dataset, Audio 
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, default_data_collator 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def make_dataset(file_path, class_name): 
    m4a_filename = os.path.basename(file_path)
    m4a_path = os.path.join(file_path)
    split_sec = 10  # 10초 단위 분할

    # ffmpeg 설정
    AudioSegment.converter = "ffmpeg"
    AudioSegment.ffmpeg = "ffmpeg"
    AudioSegment.ffprobe = "ffprobe"

    # 오디오 로드
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    duration_sec = len(audio) / 1000
    if duration_sec < split_sec:
        print(f"10초 미만이므로 건너뜀 (길이: {duration_sec:.2f}초)")
        return None

    num_chunks = math.ceil(duration_sec / split_sec)

    # 저장 디렉토리
    wav_dir = f"./dataset/test/"
    os.makedirs(wav_dir, exist_ok=True)

    files = []
    labels = []

    for chunk_num in range(num_chunks):
        start_ms = chunk_num * split_sec * 1000
        end_ms = min((chunk_num + 1) * split_sec * 1000, len(audio))
        chunk_audio = audio[start_ms:end_ms]

        # 파일 이름 및 경로
        wav_filename = f"{os.path.splitext(m4a_filename)[0]}_{chunk_num}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        # 저장
        chunk_audio.export(
            wav_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-loglevel", "error"]
        )

        # 리스트에 저장 (파일 경로는 메모리에 저장되지 않음)
        files.append(wav_path)
        if class_name == 'real':
            labels.append(0)
        elif class_name == 'scam':
            labels.append(1)  

    return files, labels

def preprocess_function(examples, feature_extractor):
    audio_arrays = [x["array"] for x in examples["file"]]
    sampling_rates = [x["sampling_rate"] for x in examples["file"]]
    
    # wav2vec2 모델은 일반적으로 16kHz 샘플링 레이트 사용
    target_sampling_rate = feature_extractor.sampling_rate
    
    # 필요한 경우 리샘플링
    processed_audios = []
    for audio, rate in zip(audio_arrays, sampling_rates):
        if rate != target_sampling_rate:
            audio = librosa.resample(y=audio, orig_sr=rate, target_sr=target_sampling_rate)
        processed_audios.append(audio)
    
    # 특성 추출기로 전처리: 최대 10초까지 패딩 또는 자르기
    inputs = feature_extractor(
        processed_audios,
        sampling_rate=target_sampling_rate,
        padding="longest",
        max_length=target_sampling_rate * 10,
        truncation=True,
        return_tensors="pt"
    )
    
    inputs["labels"] = examples["label"]
    return inputs

def predict_with_model(model, dataset, device):
    """Fine-tuned 모델을 사용하여 예측 수행"""
    model.eval()
    predictions = []
    true_labels = []
    
    # DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="예측 중"):
            # 입력 데이터를 device로 이동
            input_values = batch["input_values"].to(device)
            labels = batch["labels"]
            
            # 모델 예측
            outputs = model(input_values=input_values)
            logits = outputs.logits
            
            # 예측 클래스 계산
            predicted_class_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(predicted_class_ids)
            true_labels.extend(labels)
    
    return predictions, true_labels

def evaluate_predictions(predictions, true_labels, label_names=None):
    """예측 결과 평가"""
    if label_names is None:
        label_names = ['real', 'scam']  # 기본 라벨 이름
    
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=label_names)
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"정확도: {accuracy:.4f}")
    print("\n분류 보고서:")
    print(report)
    print("\n혼동 행렬:")
    print(cm)
    
    return accuracy, report, cm

def main():
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    # Fine-tuned 모델과 Feature Extractor 로드
    model_path = "./fine_tuned_model"
    print(f"모델 로드 중: {model_path}")
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = AutoModelForAudioClassification.from_pretrained(model_path)
        model.to(device)
        print("모델 로드 완료")
        print(f"모델 예상 샘플링 레이트: {feature_extractor.sampling_rate}Hz")
        print(f"레이블 매핑: {model.config.label2id}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 1. 데이터셋 로드 및 전처리
    print("테스트 데이터셋 생성 중...")
    file1, label1 = make_dataset("./sample(real).m4a", 'real')
    file2, label2 = make_dataset("./sample(scam).m4a", 'scam')
    
    files = file1 + file2
    labels = label1 + label2
    
    test_dataset = Dataset.from_dict({
        "file": files,
        "label": labels
    }).cast_column("file", Audio())
    
    if test_dataset is None:
        print("데이터셋 생성 실패")
        return
    
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 2. 전처리 적용
    print("전처리 적용 중...")
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        batch_size=8,
        remove_columns=["file"]
    )
    
    # 3. 라벨을 숫자로 변환 (만약 문자열인 경우)
    label2id = model.config.label2id
    if isinstance(test_dataset[0]["labels"], str):
        def convert_labels(example):
            example["labels"] = label2id[example["labels"]]
            return example
        test_dataset = test_dataset.map(convert_labels)
    
    print("전처리 완료")
    
    # 4. 예측 수행
    print("예측 수행 중...")
    predictions, true_labels = predict_with_model(model, test_dataset, device)
    
    # 5. 결과 평가
    print("\n=== 평가 결과 ===")
    id2label = {v: k for k, v in label2id.items()}
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    
    accuracy, report, cm = evaluate_predictions(predictions, true_labels, label_names)
    
    # 6. 개별 예측 결과 출력 (처음 10개)
    print("\n=== 개별 예측 결과 (처음 10개) ===")
    for i in range(min(10, len(predictions))):
        true_label = label_names[true_labels[i]]
        pred_label = label_names[predictions[i]]
        print(f"샘플 {i+1}: 실제={true_label}, 예측={pred_label}")
    
    # 7. 예측 통계
    print(f"\n=== 예측 통계 ===")
    pred_counts = {label: predictions.count(i) for i, label in enumerate(label_names)}
    true_counts = {label: true_labels.count(i) for i, label in enumerate(label_names)}
    
    print("실제 라벨 분포:")
    for label, count in true_counts.items():
        print(f"  {label}: {count}개")
    
    print("예측 라벨 분포:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count}개")

if __name__ == "__main__":
    main()
