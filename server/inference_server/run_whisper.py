# run_whisper.py
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_DIR = "./whisper"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

def whisper_infer(call_id: str, chunk_num: int, wav_path: str) -> str:  
    try:
        waveform, sr = torchaudio.load(wav_path)  
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        input_features = processor(
            waveform.squeeze(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print(f"[Whisper 추론 결과] {call_id}_{chunk_num}: {transcription}")
        return transcription

    except Exception as e:
        print(f"[Whisper 추론 오류] {call_id}_{chunk_num}: {e}")
        return "error"
