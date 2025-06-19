package com.quicinc.chatapp

data class CallWithChunks(
    val call: CallModel,
    val chunkResults: List<ChunkResultModel>,
    var isExpanded: Boolean = false
) {

    // Java 호환용 copy 함수
    fun copyWithNewChunks(newChunks: List<ChunkResultModel>): CallWithChunks {
        return CallWithChunks(call, newChunks, isExpanded)
    }
}
