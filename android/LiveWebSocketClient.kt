package com.quicinc.chatapp

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject

class LiveWebSocketClient(
    private val serverUrl: String
) {

    companion object {
        private const val TAG = "LiveWebSocketClient"
    }

    private var webSocket: WebSocket? = null
    private val client = OkHttpClient()

    fun connect() {
        val request = Request.Builder()
            .url(serverUrl)
            .addHeader("Origin", "http://localhost")  // 🔧 변경됨: WebSocket 연결 시 Origin 헤더 명시
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: okhttp3.Response) {
                Log.d(TAG, "✅ WebSocket 연결 성공")
            }

            override fun onMessage(ws: WebSocket, text: String) {
                Log.d(TAG, "📥 수신된 메시지: $text")
            }

            override fun onMessage(ws: WebSocket, bytes: ByteString) {
                Log.d(TAG, "📥 수신된 바이너리 메시지: ${bytes.hex()}")
            }

            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "⚠️ WebSocket 연결 종료 중: $code / $reason")
                ws.close(code, reason)
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "❌ WebSocket 연결 종료됨: $code / $reason")
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: okhttp3.Response?) {
                Log.e(TAG, "🚨 WebSocket 오류 발생: ${t.message}", t)
            }
        })
    }

    fun sendMessage(message: String) {
        if (webSocket != null) {
            webSocket?.send(message)
            Log.d(TAG, "📤 메시지 전송됨: $message")
        } else {
            Log.w(TAG, "❗ WebSocket 연결이 아직 안 됨")
        }
    }

    fun sendBinary(data: ByteArray) {
        if (webSocket != null) {
            webSocket?.send(ByteString.of(*data))
            Log.d(TAG, "📤 바이너리 데이터 전송됨 (크기: ${data.size})")
        } else {
            Log.w(TAG, "❗ WebSocket 연결이 아직 안 됨")
        }
    }

    fun sendJsonMessage(
        phoneNum: String,
        timestamp: String,
        inputText: String,
        deepVoice: Int,
        voicePhishing: Int,
        reason: String
    ) {
        if (webSocket != null) {
            val json = JSONObject().apply {
                put("phone_num", phoneNum)
                put("call_datetime", timestamp)
                put("input_text", inputText)
                put("deep voice", deepVoice)         // ✅ 스페이스 있음!
                put("voice phishing", voicePhishing) // ✅ 스페이스 있음!
                put("reason", reason)
            }
            val jsonString = json.toString()
            webSocket?.send(jsonString)
            Log.d(TAG, "📤 JSON 메시지 전송됨: $jsonString")
        } else {
            Log.w(TAG, "❗ WebSocket 연결이 아직 안 됨")
        }
    }

    fun disconnect() {
        webSocket?.close(1000, "종료")
        webSocket = null
    }
}
