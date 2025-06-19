import os
import asyncio
import traceback
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CALLS_BASE = 'calls'
DATA_PATH = '/data/WebSocket'
result_sockets = {}  # 추론 결과 WebSocket 연결 저장소

def ensure_call_directories(call_id):
    call_dir = os.path.join(CALLS_BASE, call_id)
    dirs = [
        os.path.join(call_dir, 'chunks'),
        os.path.join(call_dir, 'recovered'),
        os.path.join(call_dir, 'mp3'),
        os.path.join(call_dir, 'wav')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f'폴더 생성 확인: {dir_path}')
    return call_dir

def get_next_chunk_number(chunks_dir):
    try:
        existing_chunks = [f for f in os.listdir(chunks_dir) if f.endswith('.m4a')]
        return len(existing_chunks)
    except:
        return 0

@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    call_dir = ensure_call_directories(call_id)
    chunks_dir = os.path.join(call_dir, 'chunks')
    result_sockets[call_id] = websocket  # WebSocket 저장
    print(f"[연결됨] 통화 ID: {call_id}")

    try:
        while True:
            data = await websocket.receive_bytes()

            chunk_num = get_next_chunk_number(chunks_dir)
            chunk_filename = f"{call_id}_{chunk_num}.m4a"
            chunk_path = os.path.join(chunks_dir, chunk_filename)
            print(f'[청크 저장] {chunk_path}')

            try:
                with open(chunk_path, 'wb') as f:
                    f.write(data)
                print(f"[수신 완료] {chunk_filename}")
            except Exception as e:
                print(f'파일 저장 중 오류 발생: {str(e)}')
                await websocket.send_json({'error': f'파일 저장 실패: {str(e)}'})
                continue

            session_id = f"{call_id}_{chunk_num}"
            os.makedirs(f"{DATA_PATH}/{session_id}", exist_ok=True)
            os.system(f"cp {chunk_path} {DATA_PATH}/{session_id}/record.m4a")

            print(f'복구 요청 시작: {session_id}')
            try:
                response = requests.get(f"http://voice_restore_server:8301/recover?sessionId={session_id}&state=1")
                print(f'복구 응답: {response.status_code}')
            except Exception as e:
                print(f'복구 요청 실패: {str(e)}')
                print(traceback.format_exc())
                await websocket.send_json({'error': f'복구 요청 실패: {str(e)}'})
                continue

            if response.status_code == 200:
                print('복구 성공')
                await websocket.send_json({
                    'message': '처리 완료',
                    'chunk_number': chunk_num,
                    'chunk_path': chunk_path
                })
            else:
                print(f'복구 실패: {response.status_code}')
                await websocket.send_json({'error': '복구 서버 오류'})
    except WebSocketDisconnect:
        print(f"[연결 종료] 통화 ID: {call_id}")
        result_sockets.pop(call_id, None)  # 연결 해제 시 제거

@app.post("/inference-result/")  # 추론 서버가 결과를 POST하는 엔드포인트
async def receive_inference_result(data: dict):
    call_id = data.get("call_id")
    websocket = result_sockets.get(call_id)
    if websocket:
        await websocket.send_json(data)
        print(f"[결과 전송 완료] {data}")
        return {"status": "sent"}
    else:
        print(f"[경고] WebSocket 연결 없음: {call_id}")
        return {"status": "no connection"}
