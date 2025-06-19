package com.quicinc.chatapp

import android.content.Context
import android.os.Environment
import android.os.FileObserver
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import okhttp3.*
import okio.ByteString
import org.json.JSONObject

import com.google.gson.Gson
import com.google.gson.JsonSyntaxException

import android.content.Intent
import org.json.JSONException
import java.nio.file.Paths

class CallRecordingObserver(private val context: Context) {

    companion object {
        private const val TAG = "CallRecordingObserver"
        private const val UPLOAD_INTERVAL = 10000L // 10초
    }

    private var fileObserver: FileObserver? = null
    private val handler = Handler(Looper.getMainLooper())

    private var currentM4aFile: String? = null
    private var isUploading = false
    private var lastUploadedSize: Long = 0
    private var lastSentTime: Long = 0L  // 마지막 전송 시간 기록 변수

    private var prefix: String = ""
    private var callId: String = ""
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient()


    //private val modelDir = Environment.getExternalStorageDirectory().absolutePath + "/models"
    //private val htpExtensionsDir = ""

    //private val llmController = LLMController()
    //private val genieWrapper = GenieWrapper(modelDir, htpExtensionsDir)
    //private val openAIClient = OpenAIClient("bla bla bla")

    private var chunkInsertedListener: OnChunkInsertedListener? = null

    // 시연용 서버와 통신
    private var liveWSManager: LiveWebSocketClient? = null

    fun setOnChunkInsertedListener(listener: OnChunkInsertedListener) {
        this.chunkInsertedListener = listener
    }


    data class OpenAIRequest(
        val model: String,
        val messages: List<Message>,
        val temperature: Double = 0.7,
        val top_p: Double = 1.0,
        val n: Int = 1
    )

    // 온디바이스 LLM 관련
    private lateinit var genieWrapper: GenieWrapper
    private val llmResponseBuffer = StringBuilder()
    private val externalCacheDir = context.externalCacheDir?.absolutePath ?: ""
    private val modelDir = Paths.get(externalCacheDir, "models", modelName).toString()


    fun startWatching() {
        val path = Environment.getExternalStorageDirectory().absolutePath + "/Recordings/Call"
        Log.d(TAG, "Attempting to watch path: $path")

        // LiveWebSocketClient
        //liveWSManager = LiveWebSocketClient("ws://54.180.145.14:8001/ws")
        liveWSManager = LiveWebSocketClient("ws://54.180.145.14:8001/ws/ingest")
        liveWSManager?.connect()

        // LLMManager

        genieWrapper = LLMManager.getWrapper() ?: run {
            Log.e(TAG, "GenieWrapper not initialized!")
            return
        }


        try {
            Log.d(TAG, "GenieWrapper 초기화 전: $modelDir / $htpConfigPath")
            Log.d(TAG, "모델 디렉토리 존재 여부: ${File(modelDir).exists()}")
            Log.d(TAG, "HTP 설정 파일 존재 여부: ${File(htpConfigPath).exists()}")
            if (!File(modelDir).exists() || !File(htpConfigPath).exists()) {
                Log.e(TAG, "모델 파일 또는 설정 파일 없음. 초기화 중단")
                return
            }
            genieWrapper = GenieWrapper(modelDir, htpConfigPath)
            Log.d(TAG, "$modelName 모델 로드됨")
        } catch (e: Exception) {
            Log.e(TAG, "GenieWrapper 초기화 실패", e)
            return
        }

        fileObserver = object : FileObserver(path, CREATE) {
            override fun onEvent(event: Int, file: String?) {
                if (event == CREATE && file != null && file.endsWith(".m4a")) {
                    val appDir = context.getExternalFilesDir("FinalResult")
                    if (appDir != null && !appDir.exists()) {
                        appDir.mkdirs()
                    }

                    val fullPath = "$path/$file"
                    Log.d(TAG, "New .m4a file detected: $file")

                    val timestamp = SimpleDateFormat("yyMMddHHmmss", Locale.getDefault()).format(Date())
                    val cleaned = file.replace(Regex("^통화[ _]?녹음[ _]?"), "")
                    val phoneNumber = cleaned.replace(Regex("[^0-9]"), "")
                    val baseName = File(cleaned).nameWithoutExtension.replace(" ", "_")

                    prefix = "${baseName}_$timestamp"
                    callId = timestamp
                    lastUploadedSize = 0
                    lastSentTime = 0L  // 새 파일 감지 시 초기화

                    DBManager.insertCall(CallModel(callId, phoneNumber))
                    Log.d(TAG, "CallModel inserted: $callId, $phoneNumber")

                    connectWebSocket(callId)
                    startPartialUploadLoop(fullPath)
                }
            }
        }

        fileObserver?.startWatching()
        Log.d(TAG, "Started watching directory: $path")
    }

    fun stopWatching() {
        fileObserver?.stopWatching()
        Log.d(TAG, "Stopped watching directory")
    }

    fun parseScamResultWithRegex(raw: String?): Pair<String, String>? {
        if (raw == null) return null

        val scamRegex = Regex("\"is_scam\"\\s*:\\s*\"(.*?)\"")
        val reasonRegex = Regex("\"reason\"\\s*:\\s*\"(.*?)\"")

        val scamMatch = scamRegex.find(raw)
        val reasonMatch = reasonRegex.find(raw)

        return if (scamMatch != null && reasonMatch != null) {
            val isScam = scamMatch.groupValues[1]
            val reason = reasonMatch.groupValues[1]
            Pair(isScam, reason)
        } else {
            null
        }
    }

    fun parseScamResultWithJson(raw: String?): Pair<String, String>? {
        return try {
            val json = JSONObject(raw)
            val segments = json.getJSONArray("segments")
            if (segments.length() > 0) {
                val obj = segments.getJSONObject(0)
                val isScam = obj.getString("is_scam")
                val reason = obj.getString("reason")
                Pair(isScam, reason)
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "❌ JSON 파싱 실패", e)
            null
        }
    }




    private fun connectWebSocket(callId: String) {
        val url = "ws://54.180.162.170:5000/ws/$callId"  //"ws://141.148.197.55:5000/ws/$callId"
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "WebSocket 연결 성공")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket 연결 실패: ${t.message}")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket 종료됨: $reason")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "서버로부터 메시지 수신: $text")

                val json = JSONObject(text)

                // 추론 결과 메시지인 경우
                if (json.has("inference_result")) {
                    val callId = json.getString("call_id")
                    val status = json.getString("status")
                    val result = json.getString("inference_result")
                    val result2 = json.getString("inference_result2")
                    val chunkNumber = json.getInt("chunk_number")
                    val retrievedArray = json.getJSONArray("retrieved_chunks")

                    val retrievedChunks = mutableListOf<String>()
                    for (i in 0 until retrievedArray.length()) {
                        retrievedChunks.add(retrievedArray.getString(i))
                        Log.d(TAG, "retrievedChunks: $retrievedChunks")
                    }

                    //val llmResult = llmController.runLLM(result)
                    //Log.d(TAG, "LLM 결과: $llmResult")
                    val messages = """
                        You are a voice phishing detection expert.



                        [Reference examples]
                        $retrievedChunks

                        [Call to analyze]
                        $result2

                        If the call shows any phishing clue, answer “Phishing”. If not, answer “Normal”.

                        You Must answer in JSON format:
                        {
                          "segments": [
                            {
                              "is_scam": "Phishing" or "Normal",
                              "reason": "Reason: Brief explanation"
                            }
                          ]
                        }
                        """.trimIndent()

                    genieWrapper.getResponseForPrompt(messages) { response ->
                        llmResponseBuffer.append(response)
                        val current = llmResponseBuffer.toString().replace("```", "").trim()
                        //Log.d(TAG, "LLM 응답 수신: $current")

                        try {
                            val json = JSONObject(current)  // 이 시점에 성공해야만 파싱됨
                            val segments = json.getJSONArray("segments")
                            if (segments.length() > 0) {
                                val obj = segments.getJSONObject(0)
                                val isScam = obj.getString("is_scam")
                                val reason = obj.getString("reason")

                                Log.d(TAG, "✅ Llama3 판단: $isScam")
                                Log.d(TAG, "📝 Llama3 이유: $reason")
                                /*
                                DBManager.insertChunkResult(
                                    ChunkResultModel(
                                        callId = callId,
                                        chunkNumber = chunkNumber,
                                        isDeepfake = result,
                                        isPhishing = isScam,
                                        reason = reason
                                    )
                                )
                                chunkInsertedListener?.onChunkInserted(callId, chunkNumber)
                                Log.d(TAG, "DBManager.insertChunkResult 완료 $callId, $chunkNumber")
                                */
                                llmResponseBuffer.clear()
                    /*
                    val messages2 = listOf(
                        Message("system", "너는 절대 설명 없이 JSON만 반환하는 보이스피싱 판단기야."),
                        Message("user", """
                                다음 통화 내용을 분석하고 반드시 한국어를 써서 아래 JSON 형식으로만 응답해.
                                {
                                  "segments": [
                                    {
                                      "is_scam": "보이스피싱" 또는 "정상",
                                      "reason": "판단 이유"
                                    }
                                  ]
                                }
                                유사한 사례: $retrievedChunks
                               
                                통화 내용:
                                $result2
                                """.trimIndent())
                    )

                    val llmResult = openAIClient.requestChat(
                        messages = messages2,
                        onResult = { response ->
                            val result3 = parseScamResultWithRegex(response)
                            if (result3 != null) {
                                val (isScam, reason) = result3
                                Log.d(TAG,"✅ GPT 판단: $isScam")
                                Log.d(TAG, "📝 GPT 이유: $reason")

                                DBManager.insertChunkResult(
                                    ChunkResultModel(
                                        callId = callId,
                                        chunkNumber = chunkNumber,
                                        isDeepfake = result,
                                        isPhishing = isScam,
                                        reason = reason
                                    )
                                )

                                chunkInsertedListener?.onChunkInserted(callId, chunkNumber)
                                Log.d(TAG, "DBManager.insertChunkResult 완료 $callId, $chunkNumber")


                                // 시연용 서버로 전송
                                liveWSManager?.sendJsonMessage(
                                    phoneNum = callId,
                                    timestamp = DBManager.getCallerByCallId(callId).toString(),
                                    inputText = result2,
                                    deepVoice = if(result=="fake") 1 else 0,
                                    voicePhishing = if(isScam=="보이스피싱") 1 else 0,
                                    reason = reason
                                )


                            } else {
                                Log.d(TAG, "⚠️ GPT 정규식 파싱 실패")
                            }
                        },
                        onError = { error -> println("오류: $error") }
                    )*/
                }


            }
        })
    }

    private fun startPartialUploadLoop(filePath: String) {
        currentM4aFile = filePath
        isUploading = true

        val uploadRunnable = object : Runnable {
            override fun run() {
                if (!isUploading || currentM4aFile == null) return
                sendChunkOverWebSocket(currentM4aFile!!)
                handler.postDelayed(this, UPLOAD_INTERVAL)
            }
        }

        handler.postDelayed(uploadRunnable, UPLOAD_INTERVAL) // [수정] 최초 전송도 10초 이후부터 시작하도록 지연 설정
    }

    fun stopPartialUpload() {
        isUploading = false
        currentM4aFile = null
        webSocket?.close(1000, "Stop Upload")
    }

    private fun sendChunkOverWebSocket(filePath: String) {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastSentTime < UPLOAD_INTERVAL) {
            Log.d(TAG, "전송 주기 미도달, 생략 (${(currentTime - lastSentTime)}ms)")
            return
        }

        val file = File(filePath)
        if (!file.exists()) {
            Log.e(TAG, "파일 존재하지 않음: $filePath")
            return
        }

        val fileSize = file.length()
        if (fileSize == lastUploadedSize) {
            Log.d(TAG, "변경된 내용 없음. 전송 생략")
            return
        }

        lastUploadedSize = fileSize
        lastSentTime = currentTime  // 전송 직후 시간 갱신

        try {
            val seconds = System.currentTimeMillis() / 1000 % 100000
            val filename = "${prefix}_at${seconds}s.m4a"
            val tempFile = File(Environment.getExternalStorageDirectory(), "temp_$filename")

            file.copyTo(tempFile, overwrite = true)
            val bytes = tempFile.readBytes()
            webSocket?.send(ByteString.of(*bytes))
            Log.d(TAG, "WebSocket으로 전송 완료: $filename")
            tempFile.delete()

        } catch (e: Exception) {
            Log.e(TAG, "WebSocket 전송 실패", e)
        }
    }
}
