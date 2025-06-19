package com.quicinc.chatapp

import android.os.Environment
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

class LLMController {

    // 지니 번들 위치
    private val geniePath: String = "/data/local/tmp/genie_bundle"

    // 프롬프트 실행 명령어 생성
    private fun buildCommand(prompt: String): List<String> {
        val shellPrompt =
            "'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n$prompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>'"

        return listOf(
            "sh", "-c",
            """
        cd $geniePath && \
        export LD_LIBRARY_PATH=$geniePath && \
        export ADSP_LIBRARY_PATH=$geniePath && \
        ./genie-t2t-run -c genie-config.json -p $shellPrompt
        """.trimIndent()
        )
    }
    /*
    class LLMController {

        // 지니 번들 위치
        private val geniePath: String = "/data/local/tmp/genie_bundle"

        // 프롬프트 실행 명령어 생성
        private fun buildCommand(prompt: String): List<String> {
            val shellPrompt = "'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n$prompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>'"

            return listOf(
                "sh", "-c",
                """
        cd $geniePath && \
        export LD_LIBRARY_PATH=\${'$'}PWD && \
        export ADSP_LIBRARY_PATH=\${'$'}PWD && \
        ./genie-t2t-run -c genie-config.json -p $shellPrompt
        """.trimIndent()
            )
        }*/


    // 명령어 실행
    fun runLLM(prompt: String): String {
        return try {
            val process = ProcessBuilder(buildCommand(prompt))
                .redirectErrorStream(true)
                .start()

            val reader = BufferedReader(InputStreamReader(process.inputStream))
            val output = reader.readLines().joinToString("\n")
            Log.e("LLMController", output)
            process.waitFor()
            output
        } catch (e: Exception) {
            Log.e("LLMController", "Genie 실행 실패", e)
            "runLLM error"
        }
    }
}