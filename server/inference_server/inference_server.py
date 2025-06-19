import os
import json
import asyncio
import redis
import logging
import aiohttp
import subprocess  
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from run_wav2vec2 import wav2vec2_infer
from run_whisper import whisper_infer

from pydub import AudioSegment  

from retrieval import Retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Retrieval 인스턴스 생성 및 app.state에 저장
    app.state.retriever = Retrieval()
    task = asyncio.create_task(inference_worker())
    background_tasks.add(task)
    yield
    for task in background_tasks:
        task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

UPLOAD_SERVER_HTTP_URL = "http://upload_server:5000/inference-result/"


@app.get("/alive")
async def alive():
    return {"status": "inference server alive"}


# 재인코딩 방식으로 정확하게 자르기 위한 유틸 함수
def cut_audio_ffmpeg(input_path, output_path, start_sec, duration_sec):
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", input_path,
            "-ar", "16000",  
            "-ac", "1",      
            "-vn",           
            output_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"[ffmpeg 자르기 실패] {input_path} → {output_path}: {e}")
        raise

# 추론 및 전송
async def forward_result(call_id, chunk_num, wav_path):  
    try:
        # 오디오 길이 확인용
        audio = AudioSegment.from_file(wav_path, format="wav")
        duration_sec = len(audio) / 1000  # ms → sec

        # 자른 파일 저장 경로
        wav2vec_path = f"/tmp/{call_id}_{chunk_num}_last10.wav"
        whisper_path = f"/tmp/{call_id}_{chunk_num}_last50.wav"

        # ffmpeg로 자르기 (재인코딩 방식)
        if duration_sec > 10:
            cut_audio_ffmpeg(wav_path, wav2vec_path, duration_sec - 10, 10)
        else:
            cut_audio_ffmpeg(wav_path, wav2vec_path, 0, duration_sec)

        if duration_sec > 50:
            cut_audio_ffmpeg(wav_path, whisper_path, duration_sec - 50, 50)
        else:
            cut_audio_ffmpeg(wav_path, whisper_path, 0, duration_sec)

        # 각각의 추론을 예외로 감싸서 개별 오류 허용
        try:
            result = wav2vec2_infer(call_id, chunk_num, wav2vec_path)
        except Exception as e:
            logger.error(f"[Wav2Vec2 추론 오류] {call_id}_{chunk_num}: {e}")
            result = "error"

        try:
            result_whisper = whisper_infer(call_id, chunk_num, whisper_path)
        except Exception as e:
            logger.error(f"[Whisper 추론 오류] {call_id}_{chunk_num}: {e}")
            result_whisper = "error"

        # retrieve
        retrieved_chunks = app.state.retriever.retrieve(result_whisper)

        message = {
            "inference_result": result,
            "inference_result2": result_whisper,
            "call_id": call_id,
            "chunk_number": chunk_num,
            "status": "analyzed",
            "retrieved_chunks": retrieved_chunks
        }

        # 결과 전송 부분도 예외로 감싸기
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(UPLOAD_SERVER_HTTP_URL, json=message) as resp:
                    logger.info(f"[추론 결과 전송 완료] {message} → status={resp.status}")
        except Exception as e:
            logger.error(f"[결과 전송 오류] {call_id}_{chunk_num}: {e}")

        # 임시 파일 삭제
        os.remove(wav2vec_path)
        os.remove(whisper_path)

    except Exception as e:
        logger.error(f"[전처리 또는 추론 실패] {e}")

async def inference_worker():
    logger.info("Redis 큐 대기 시작")
    while True:
        try:
            task = redis_client.blpop("inference_tasks", timeout=5)
            if task:
                data = json.loads(task[1])
                call_id = data["call_id"]
                chunk_num = data["chunk_num"]
                wav_path = data["wav_path"]  # mp3_path → wav_path

                logger.info(f"[작업 수신] call_id={call_id}, chunk={chunk_num}")
                await forward_result(call_id, chunk_num, wav_path)
            else:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[작업 처리 오류] {e}")
            await asyncio.sleep(1)
