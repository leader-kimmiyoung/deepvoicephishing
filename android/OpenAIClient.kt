package com.quicinc.chatapp

import okhttp3.Interceptor
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST




interface OpenAIApi {
    @POST("v1/chat/completions")
    fun getChatCompletion(@Body request: OpenAIRequest): Call<OpenAIResponse>
}

// 데이터 클래스들
data class Message(val role: String, val content: String)
data class OpenAIRequest(
    val model: String,
    val messages: List<Message>,
    val temperature: Double = 0.7
)
data class Choice(val message: Message)
data class OpenAIResponse(val choices: List<Choice>)
data class ScamSegment(
    val is_scam: String,
    val reason: String
)
data class ScamAnalysisResponse(
    val segments: List<ScamSegment>
)


class OpenAIClient(private val apiKey: String) {

    private val retrofit: Retrofit
    private val api: OpenAIApi


    init {
        val httpClient = OkHttpClient.Builder()
            .addInterceptor(Interceptor { chain ->
                val newRequest = chain.request().newBuilder()
                    .addHeader("Authorization", "Bearer $apiKey")
                    .addHeader("Content-Type", "application/json")
                    .build()
                chain.proceed(newRequest)
            })
            .build()

        retrofit = Retrofit.Builder()
            .baseUrl("https://api.openai.com/")
            .client(httpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        api = retrofit.create(OpenAIApi::class.java)
    }

    fun makePrompt(result: String): String {
        val promptTemplate = """
다음 통화 내용이 보이스피싱에 해당하는지 논리적으로 판단해주세요.

[보이스피싱 판단 기준]
1. 상대방이 긴박하거나 위급한 상황을 조성하며 행동을 재촉함
2. 본인의 신분을 숨기거나 사칭(경찰, 검사, 금융기관 등)함
3. 금융정보(계좌번호, OTP, 보안카드 등)나 금전 전달을 요구함
4. 대화를 통해 심리적 압박이나 겁을 줌
5. 개인정보를 수집하거나 유도함

[통화 내용 분석 순서]
1. 통화 내용의 핵심 내용을 요약하세요.
2. 보이스피싱 판단 기준 중 어떤 항목과 유사한지 항목별로 비교하세요.
3. 최종적으로 보이스피싱 여부를 판단하고 그 이유를 설명하세요.

[분석할 통화 내용]
$result

[응답 형식]
다음 JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요:
{{
    "segments": [
        {{
            "is_scam": "보이스피싱" 또는 "정상",
            "reason": "판단 이유"
        }}
    ]
}}
"""
        return promptTemplate
    }


    fun requestChat(
        messages: List<Message>,
        model: String = "gpt-4o",
        temperature: Double = 0.2,
        onResult: (String?) -> Unit,
        onError: (String) -> Unit
    ) {
        val request = OpenAIRequest(
            model = model,
            messages = messages,
            temperature = temperature
        )

        api.getChatCompletion(request).enqueue(object : Callback<OpenAIResponse> {
            override fun onResponse(call: Call<OpenAIResponse>, response: Response<OpenAIResponse>) {
                if (response.isSuccessful) {
                    val reply = response.body()?.choices?.firstOrNull()?.message?.content
                    onResult(reply)
                } else {
                    onError("HTTP ${response.code()} - ${response.errorBody()?.string()}")
                }
            }

            override fun onFailure(call: Call<OpenAIResponse>, t: Throwable) {
                onError("Network Error: ${t.message}")
            }
        })
    }

}
