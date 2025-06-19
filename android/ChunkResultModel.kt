package com.quicinc.chatapp

data class ChunkResultModel(
    val callId: String,
    val chunkNumber: Int,
    val isDeepfake: String,
    val isPhishing: String,
    val reason: String
)
