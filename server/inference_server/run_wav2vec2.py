import os
import torchaudio  
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 디렉토리 및 로딩
MODEL_DIR = "./fine_tuned_model"
extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# 레이블 매핑
label_map = {
    0: "fake",
    1: "real"
}

# 추론 함수 (입력은 .wav PCM 파일)
def wav2vec2_infer(call_id: str, chunk_num: int, wav_path: str) -> str:  
    try:
        waveform, sr = torchaudio.load(wav_path)  
        waveform_np = waveform.squeeze().numpy()

        print(f"\n--- [DEBUG] {call_id}_{chunk_num} ---")
        print(f"[샘플링레이트] 원본: {sr}Hz, 모델 기대: {extractor.sampling_rate}Hz")
        print(f"[오디오 길이] {len(waveform_np)/sr:.2f}초, 배열 shape: {waveform_np.shape}")

        if sr != extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=extractor.sampling_rate)
            waveform = resampler(waveform)  
            waveform_np = waveform.squeeze().numpy()
            print(f"[리샘플링 후] 길이: {len(waveform_np)/extractor.sampling_rate:.2f}초")

        # 너무 짧은 경우 무시
        if len(waveform_np) < extractor.sampling_rate * 2:  # 2초 미만
            print(f"[무시됨] 입력이 너무 짧습니다: {len(waveform_np)/extractor.sampling_rate:.2f}초")
            return "too_short"

        max_len = extractor.sampling_rate * 10

        inputs = extractor(
            waveform_np,
            sampling_rate=extractor.sampling_rate,
            return_tensors="pt",
            padding=False,           
            truncation=True,
            max_length=max_len
        )
        print(f"[추론 입력 텐서] input_values shape: {inputs['input_values'].shape}")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_label = np.argmax(probs)
            confidence = probs[predicted_label]
            result = label_map[predicted_label]

        print(f"[logits] {logits.cpu().numpy()}")
        print(f"[softmax 확률] {probs}")
        print(f"[예측 결과] label={result}, confidence={confidence:.4f}")

        return result

    except Exception as e:
        print(f"[Wav2Vec2 추론 오류] {call_id}_{chunk_num}: {e}")
        return "error"