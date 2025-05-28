import os
import math
import random
from pydub import AudioSegment

def m4a_to_wav(class_name, reverse_type):
    for m4a_filename in os.listdir(f"./dataset/{class_name}/{reverse_type}"):  
        m4a_path = os.path.join(f"./dataset/{class_name}/{reverse_type}", m4a_filename) 
        split_sec = 10  # 10초 단위 분할

        # ffmpeg 경로 설정
        AudioSegment.converter = "ffmpeg"
        AudioSegment.ffmpeg = "ffmpeg"
        AudioSegment.ffprobe = "ffprobe"

        # 오디오 로드
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        duration_sec = len(audio) / 1000  # 전체 길이(초)

        # 10초 미만이면 pass
        if duration_sec < split_sec:
            print(f"{class_name}.m4a 는 10초 미만이므로 건너뜀 (길이: {duration_sec:.2f}초)")
            return

        num_chunks = math.ceil(duration_sec / split_sec)

        # 저장 디렉토리 설정
        wav_dir = f"./dataset/wav/{class_name}/{reverse_type}_split"
        os.makedirs(wav_dir, exist_ok=True)

        for chunk_num in range(num_chunks):
            start_ms = chunk_num * split_sec * 1000
            end_ms = min((chunk_num + 1) * split_sec * 1000, len(audio))
            chunk_audio = audio[start_ms:end_ms]

            # wav로 저장 (pcm_s16le, 16kHz, mono)
            wav_filename = f"{m4a_filename.split('.')[0]}_{chunk_num}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)
            chunk_audio.export(
                wav_path,
                format="wav",
                parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-loglevel", "error"]
            )
            print(f'WAV 파일 저장: {wav_path}')

def make_reverse(class_name) :
    for m4a_filename in os.listdir(f"./dataset/{class_name}/original"):  
        m4a_path = os.path.join(f"./dataset/{class_name}/original", m4a_filename)
        # print(m4a_path)
        audio = AudioSegment.from_file(m4a_path, format="m4a")

        # 역재생 (reverse)
        reversed_audio = audio.reverse()

        # 저장
        if not os.path.exists(f"./dataset/{class_name}/reverse"):
            os.makedirs(f"./dataset/{class_name}/reverse")
        reversed_audio.export(f"./dataset/{class_name}/reverse/{m4a_filename.split('.')[0]}.m4a", format="mp4")

def random_split(class_name, reverse_type):
    for m4a_filename in os.listdir(f"./dataset/{class_name}/{reverse_type}"):  
        m4a_path = os.path.join(f"./dataset/{class_name}/{reverse_type}", m4a_filename)
        # print(m4a_path)
        split_sec = 10  # 10초 단위 분할

        # ffmpeg 경로 설정
        AudioSegment.converter = "ffmpeg"
        AudioSegment.ffmpeg = "ffmpeg"
        AudioSegment.ffprobe = "ffprobe"

        # 오디오 로드
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        duration_sec = len(audio) / 1000  # 전체 길이(초)

        # 10초 미만이면 pass
        if duration_sec < split_sec:
            print(f"{class_name}.m4a 는 10초 미만이므로 건너뜀 (길이: {duration_sec:.2f}초)")
            return

        num_chunks = math.ceil(duration_sec / split_sec)

        # 저장 디렉토리 설정
        wav_dir = f"./dataset/wav/{class_name}/{reverse_type}_rd_split/"
        os.makedirs(wav_dir, exist_ok=True)

        for chunk_num in range(num_chunks):
            start_ms = chunk_num * split_sec * 1000
            end_ms = min((chunk_num + 1) * split_sec * 1000, len(audio)) 
            rndm = random.randint(1, 9)
            chunk_audio = audio[start_ms+rndm*1000:end_ms+rndm*1000]
            print(f"rndm: {rndm}, start_ms: {start_ms+rndm*1000}, end_ms: {end_ms+rndm*1000}")

            # wav로 저장 (pcm_s16le, 16kHz, mono)
            wav_filename = f"{m4a_filename.split('.')[0]}_{chunk_num}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)
            chunk_audio.export(
                wav_path,
                format="wav",
                parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-loglevel", "error"]
            )
            print(f'WAV 파일 저장: {wav_path}')

# m4a를 reverse 해서 m4a로 저장
make_reverse(0)
make_reverse(1)

# m4a를 wav로 변환
m4a_to_wav(0, "original")
m4a_to_wav(0, "reverse")
m4a_to_wav(1, "original")
m4a_to_wav(1, "reverse")

# random으로 split
random_split(0, "original")
random_split(0, "reverse")
random_split(1, "original")
random_split(1, "reverse")
