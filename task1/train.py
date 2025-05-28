import os
import librosa
import numpy as np
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from sklearn.metrics import precision_recall_fscore_support
import torch
import evaluate
import transformers

# 버전 확인
print(f"Transformers 버전: {transformers.__version__}")
print(f"PyTorch 버전: {torch.__version__}")

# CUDA 상태 확인
cuda_available = torch.cuda.is_available()
print(f"CUDA 사용 가능: {cuda_available}")
if cuda_available:
    print(f"CUDA 장치 수: {torch.cuda.device_count()}")
    print(f"CUDA 장치 이름: {torch.cuda.get_device_name(0)}")

# ---------------------------
# 디버깅을 위한 Callback 클래스
# ---------------------------
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 각 로그 스텝에서 loss 및 learning rate 등 출력
        if logs is not None:
            if "loss" in logs:
                print(f"[DEBUG] Step {state.global_step}, Loss: {logs['loss']}")
            if "learning_rate" in logs:
                print(f"[DEBUG] Step {state.global_step}, Learning Rate: {logs['learning_rate']}")
    
    def on_train_batch_end(self, args, state, control, **kwargs):
        # 매 배치 종료 후 gradient norm 계산 (가능한 경우)
        model = kwargs.get("model")
        if model is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"[DEBUG] Step {state.global_step}, Gradient Norm: {total_norm:.4f}")

# 1. 데이터셋 준비 함수
def create_dataset(): 
    data = []
    for class_name in [0, 1] :
        for folder_name in ['original_split', 'reverse_split', 'original_rd_split', 'reverse_rd_split']: 
            for file_name in os.listdir(f"./dataset/wav/{class_name}/{folder_name}"):
                data.append({"file": os.path.join(f"./dataset/wav/{class_name}/{folder_name}", file_name), "label": class_name})
                 
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df).cast_column("file", Audio())

# 2. 오디오 전처리 함수
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

# 3. 메트릭 계산 함수
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    
    return {
        "accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

from collections import Counter

def print_label_distribution(dataset, name):
    label_counts = Counter(dataset["label"])
    print(f"\n{name} 라벨 분포:")
    for label, count in sorted(label_counts.items()):
        print(f"  라벨 {label}: {count}개")
        
def main():
    # 데이터셋 생성
    dataset = create_dataset()
    print(f"전체 데이터셋 크기: {len(dataset)}")
    
    # train/test 분할 후, train에서 validation 분할
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_valid = dataset['train'].train_test_split(test_size=0.1, seed=42)
        
    train_dataset = train_valid['train']
    valid_dataset = train_valid['test']
    test_dataset = dataset['test']
    
    print(f"학습 데이터 수: {len(train_dataset)}")
    print(f"검증 데이터 수: {len(valid_dataset)}")
    print(f"테스트 데이터 수: {len(test_dataset)}")
    
    # print_label_distribution(train_dataset, "학습 데이터")
    # print_label_distribution(valid_dataset, "검증 데이터")
    # print_label_distribution(test_dataset, "테스트 데이터") 
    
    # Feature Extractor 로드
    model_name = "mo-thecreator/Deepfake-audio-detection"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    # 모델 예상 샘플링 레이트 확인
    print(f"모델 예상 샘플링 레이트: {feature_extractor.sampling_rate}Hz")
    
    # 전처리 적용
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        batch_size=8,
        remove_columns=["file"]
    )
    
    valid_dataset = valid_dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        batch_size=8,
        remove_columns=["file"]
    )
    
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        batch_size=8,
        remove_columns=["file"]
    )
    
    # 모델 로드
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
   
    model.config.label2id = {"real": 0, "scam": 1}
    model.config.id2label = {0: "real", 1: "scam"} 
    
    # 모델 정보 출력
    print(f"모델 타입: {model.config.model_type}")
    print(f"레이블 매핑: {model.config.label2id}")
    
    # TrainingArguments 설정
    training_args_kwargs = {
        # "output_dir": "./results",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 5,
        "warmup_ratio": 0.1,
        "logging_dir": "./logs",
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "push_to_hub": False,
        "fp16": False,  # CUDA 사용 불가 시
        "report_to": "none"
    }
    
    training_args = TrainingArguments(**training_args_kwargs)
    
    # Trainer 설정
    # 이제 deprecated된 tokenizer 대신 processing_class를 사용하며, 디버깅을 위한 Callback을 추가합니다.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # processing_class=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[DebugCallback()]
    )
    
    # 학습 실행
    try:
        trainer.train()
        
        # 평가
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        print(f"테스트 세트 평가 결과: {eval_result}")
        
        # 모델과 Feature Extractor 저장
        model.save_pretrained("./fine_tuned_model")
        feature_extractor.save_pretrained("./fine_tuned_model")
        print("모델이 './fine_tuned_model' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"학습 중 오류 발생: {e}") 
    
if __name__ == "__main__":
    main()
