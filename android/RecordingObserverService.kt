package com.quicinc.chatapp

import android.app.*
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat

class RecordingObserverService : Service() {

    companion object {
        private const val TAG = "RecordingObserverService"
    }

    private lateinit var observer: CallRecordingObserver2

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")

        observer = CallRecordingObserver2(context = applicationContext)
        observer.startWatching()
        Log.d(TAG, "CallRecordingObserver started")

        startForegroundServiceWithNotification()
    }

    override fun onDestroy() {
        super.onDestroy()
        observer.stopWatching()
        Log.d(TAG, "Service destroyed and observer stopped")
    }

    private fun startForegroundServiceWithNotification() {
        val channelId = "record_observer_channel"
        Log.d(TAG, "Starting foreground service with channelId: $channelId")

        // Android 8+ 채널 생성
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Recording Observer",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager?.createNotificationChannel(channel)
            Log.d(TAG, "Notification channel created")
        }

        // 알림 생성
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Recording Observer Running")
            .setContentText("Monitoring call recording folder...")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .build()

        startForeground(1, notification)
        Log.d(TAG, "Foreground notification started")
    }

    override fun onBind(intent: Intent?): IBinder? {
        Log.d(TAG, "onBind called (not used)")
        return null
    }
}
