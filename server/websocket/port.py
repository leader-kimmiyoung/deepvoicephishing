# port2.py
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
clients = []

display_html = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>📊 통화 내용 분석 결과 📊</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      font-family:'Segoe UI',sans-serif;
      background: linear-gradient(135deg, #4e54c8, #8f94fb);
      height:100vh; display:flex;
      align-items:center; justify-content:center;
      color:#fff;
    }
    .container {
      width:90vw; max-width:800px; padding:40px;
      background:rgba(255,255,255,0.1);
      backdrop-filter: blur(10px);
      border-radius:16px;
      box-shadow:0 8px 32px rgba(0,0,0,0.3);
      display:flex; flex-direction:column; gap:24px;
    }
    h1 {
      text-align:center; font-size:2rem; margin-bottom:16px;
    }
    .section-title {
      font-size:1.2rem; font-weight:500; opacity:0.8;
      margin-bottom:8px;
    }
    .text-block {
      background:rgba(255,255,255,0.2);
      padding:24px; border-radius:12px;
      font-size:1.4rem; line-height:1.6;
    }
    .status-grid {
      display:flex; justify-content:space-around; gap:20px;
    }
    .status-item {
      flex:1; display:flex; flex-direction:column; gap:8px;
      text-align:center;
    }
    .status-item .status-title {
      font-size:1rem; opacity:0.8;
    }
    .status-box {
      background:rgba(255,255,255,0.2);
      padding:20px 0; border-radius:12px;
      font-size:2.5rem;       /* 이전 1.5rem → 2.5rem */
      font-weight:700;        /* 더 강조된 볼드체 */
    }
    .reason-block {
      background:rgba(255,255,255,0.2);
      padding:20px; border-radius:12px;
      font-size:1.2rem; line-height:1.5;
    }
    .footer {
      display:flex; justify-content:space-between;
      font-size:0.9rem; opacity:0.7;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 통화 내용 분석 결과 📊</h1>

    <div class="section-title">✅ 통화 내용</div>
    <div class="text-block" id="inputText">
      💬 통화 내용 대기 중...
    </div>

    <div class="section-title">🔍 분석 결과</div>
    <div class="status-grid">
      <div class="status-item">
        <div class="status-title">🎙️ Deepfake 여부</div>
        <div class="status-box" id="deepScore">🟢 Normal</div>
      </div>
      <div class="status-item">
        <div class="status-title">🛡️ Voice Phishing 여부</div>
        <div class="status-box" id="phishScore">🟢 Safe</div>
      </div>
    </div>

    <div class="section-title">💡 분석 사유</div>
    <div class="reason-block" id="reason">
      분석 사유 대기 중...
    </div>

    <div class="section-title">ℹ️ 기타 정보</div>
    <div class="footer">
      <div id="callDateTime">📅 --</div>
      <div id="phoneNum">📞 --</div>
    </div>
  </div>

  <script>
    const ws = new WebSocket("ws://" + location.host + "/ws/ingest");
    ws.onmessage = e => {
      const d = JSON.parse(e.data);
      document.getElementById('inputText').textContent =
        `💬 "${d.input_text}"`;
      document.getElementById('deepScore').textContent =
        d["deep voice"] === 1 ? '🔴 Deepfake' : '🟢 Normal';
      document.getElementById('phishScore').textContent =
        d["voice phishing"] === 1 ? '🔴 Phishing' : '🟢 Safe';
      document.getElementById('reason').textContent =
        d.reason;
      document.getElementById('callDateTime').textContent =
        `📅 ${d.call_datetime || '--'}`;
      document.getElementById('phoneNum').textContent =
        `📞 ${d.phone_num}`;
      document.getElementById('deepScore').style.color =
        d["deep voice"]===1 ? '#ff6b6b' : '#a8ff60';
      document.getElementById('phishScore').style.color =
        d["voice phishing"]===1 ? '#ff6b6b' : '#a8ff60';
    };
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def ui():
    return display_html


@app.websocket("/ws")
async def ingest(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            msg = await ws.receive_text()
            for c in clients:
                if c.client_state.value == 1:
                    await c.send_text(msg)
    except WebSocketDisconnect:
        clients.remove(ws)

@app.websocket("/ws/ingest")
async def ingest(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            msg = await ws.receive_text()
            for c in clients:
                if c.client_state.value == 1:
                    await c.send_text(msg)
    except WebSocketDisconnect:
        clients.remove(ws)

if __name__ == "__main__":
    uvicorn.run("relay:app", host="0.0.0.0", port=8001)
