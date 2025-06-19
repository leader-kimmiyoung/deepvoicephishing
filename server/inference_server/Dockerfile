# CUDA 및 PyTorch 포함된 이미지 사용
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY ./inference_server.py /app/inference_server.py
COPY ./run_wav2vec2.py /app/run_wav2vec2.py
COPY ./run_whisper.py /app/run_whisper.py
COPY ./retrieval.py /app/retrieval.py


# 로그 디렉토리 생성
RUN mkdir -p logs

RUN apt-get update 
RUN apt-get install -y libsndfile1

RUN apt-get install -y ffmpeg


# 환경 변수 설정 
ENV PYTHONUNBUFFERED=1
ENV PINECONE_API_KEY=pcsk_vhb28_A9wigSKvvEsEW2HrigpAUuXHPTr31p7HyLYxDWrHX8CxpTG3Ch6uDC9R5FF9kwU
ENV PINECONE_ENV=us-east1-gcp

# 포트 설정
EXPOSE 5001

# FastAPI 실행 (로깅 레벨 포함)
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "5001", "--log-level", "debug"]
