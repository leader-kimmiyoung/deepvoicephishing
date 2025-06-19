package com.quicinc.chatapp

import android.app.Application

class App : Application() {
    override fun onCreate() {
        super.onCreate()

        // 싱글톤 DBManager 초기화
        DBManager.init(applicationContext)
    }
}
