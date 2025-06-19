package com.quicinc.chatapp

import android.util.Log

object LLMManager {
    private var genieWrapper: GenieWrapper? = null

    fun initialize(modelDir: String, htpPath: String) {
        if (genieWrapper == null) {
            genieWrapper = GenieWrapper(modelDir, htpPath)
            Log.d("LLMManager", "GenieWrapper initialized")
        }
    }

    fun getWrapper(): GenieWrapper? {
        return genieWrapper
    }
}
