from flask import Flask, request, jsonify, make_response
from pydub import AudioSegment
import subprocess
import os
import time
import logging
import sys
import redis
import json

# Redis 연결
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

# pydub 로깅 레벨 설정
pydub_logger = logging.getLogger("pydub.converter")
pydub_logger.setLevel(logging.WARNING)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

FFMPEG_OPTIONS = {
    'loglevel': 'error',
    'hide_banner': True
}

app = Flask(__name__)

@app.route('/alive', methods=['GET'])
def aliveTest():
    return {"msg": "ALIVE"}

@app.route('/recover', methods=['GET'])
def recoverM4A():
    params = request.args.to_dict()
    session_id = params.get("sessionId")
    state = params.get("state", "1")
    
    if session_id:
        try:
            DATA_PATH = os.environ.get("FLASK_DATA_PATH", "/data/WebSocket")
            CALLS_BASE = os.environ.get("FLASK_CALLS_BASE", "/app/calls")
            
            call_id, chunk_num = session_id.rsplit('_', 1)
            call_dir = os.path.join(CALLS_BASE, call_id)
            recovered_dir = os.path.join(call_dir, 'recovered')
            mp3_dir = os.path.join(call_dir, 'mp3')
            wav_dir = os.path.join(call_dir, 'wav')  # [추가] WAV 저장 디렉토리

            os.makedirs(recovered_dir, exist_ok=True)
            os.makedirs(mp3_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)  # [추가]

            logger.info(f'복구 시작: {session_id}')
            logger.info(f'작업 디렉토리: {DATA_PATH}/{session_id}')
            
            subprocess.run(["untrunc", f"/app/ok.m4a", f"{DATA_PATH}/{session_id}/record.m4a"], check=True)

            recovered_filename = f"{call_id}_{chunk_num}_recovered.m4a"
            recovered_path = os.path.join(recovered_dir, recovered_filename)
            subprocess.run(["mv", f"{DATA_PATH}/{session_id}/record.m4a_fixed.m4a", recovered_path], check=True)
            logger.info(f'복구된 파일 저장: {recovered_path}')
            
            mp3_filename = f"{call_id}_{chunk_num}.mp3"
            mp3_path = os.path.join(mp3_dir, mp3_filename)

            AudioSegment.converter = "ffmpeg"
            AudioSegment.ffmpeg = "ffmpeg"
            AudioSegment.ffprobe = "ffprobe"

            audio_file = AudioSegment.from_file(recovered_path, format="m4a", parameters=["-loglevel", "error"])
            audio_file.export(mp3_path, format="mp3", parameters=["-loglevel", "error"])
            logger.info(f'MP3 파일 저장: {mp3_path}')

            # mp3 → wav (PCM 16bit, 16kHz, mono) 변환
            wav_filename = f"{call_id}_{chunk_num}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)
            AudioSegment.from_mp3(mp3_path).export(
                wav_path,
                format="wav",
                parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-loglevel", "error"]
            )
            logger.info(f'WAV 파일 저장: {wav_path}')

            # 작업 큐에는 wav_path만 전달
            task = {
                "call_id": call_id,
                "chunk_num": chunk_num,
                "wav_path": wav_path
            }
            redis_client.rpush("inference_tasks", json.dumps(task))

            subprocess.run(["rm", "-rf", f"{DATA_PATH}/{session_id}"])
            
            return {
                "msg": "success",
                "recovered_path": recovered_path,
                "mp3_path": mp3_path,
                "wav_path": wav_path  
            }
        except subprocess.CalledProcessError as e:
            logger.error(f'복구 실패: {str(e)}')
            return make_response({"msg": "fail", "error": str(e)}, 500)
        except Exception as e:
            logger.error(f'오류 발생: {str(e)}')
            return make_response({"msg": "error", "error": str(e)}, 500)
    else:
        return make_response({"msg": "No session"}, 400)

if __name__ == '__main__':
    logger.info("start_flask")
    app.run(host="0.0.0.0", port=8301)
