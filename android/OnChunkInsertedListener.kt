package com.quicinc.chatapp

interface OnChunkInsertedListener {
    fun onChunkInserted(callId: String, chunkNumber: Int)
}