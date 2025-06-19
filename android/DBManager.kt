package com.quicinc.chatapp

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase

object DBManager {
    private lateinit var dbHelper: DBHelper

    fun init(context: Context) {
        dbHelper = DBHelper(context.applicationContext)
    }

    fun insertCall(call: CallModel) {
        val db: SQLiteDatabase = dbHelper.writableDatabase
        val values = ContentValues().apply {
            put("call_id", call.callId)
            put("caller", call.caller)
        }
        db.insert("Call", null, values)
        db.close()
    }

    fun insertChunkResult(result: ChunkResultModel, onInserted: (() -> Unit)? = null) {
        val db: SQLiteDatabase = dbHelper.writableDatabase
        val sql = """
        INSERT OR IGNORE INTO ChunkResult (call_id, chunk_number, is_deepfake, is_phishing, reason)
        VALUES (?, ?, ?, ?, ?)
    """.trimIndent()

        val args = arrayOf(
            result.callId,
            result.chunkNumber,
            result.isDeepfake,
            result.isPhishing,
            result.reason
        )

        db.execSQL(sql, args)
        db.close()

        onInserted?.invoke()
    }


    fun getChunkResults(callId: String): List<ChunkResultModel> {
        val db = dbHelper.readableDatabase
        val cursor = db.rawQuery("SELECT * FROM ChunkResult WHERE call_id = ?", arrayOf(callId))
        val results = mutableListOf<ChunkResultModel>()

        while (cursor.moveToNext()) {
            results.add(
                ChunkResultModel(
                    callId = cursor.getString(cursor.getColumnIndexOrThrow("call_id")),
                    chunkNumber = cursor.getInt(cursor.getColumnIndexOrThrow("chunk_number")),
                    isDeepfake = cursor.getString(cursor.getColumnIndexOrThrow("is_deepfake")),
                    isPhishing = cursor.getString(cursor.getColumnIndexOrThrow("is_phishing")),
                    reason = cursor.getString(cursor.getColumnIndexOrThrow("reason"))
                )
            )
        }

        cursor.close()
        db.close()
        return results
    }

    fun getAllCalls(): List<CallModel> {
        val db = dbHelper.readableDatabase
        val cursor = db.rawQuery("SELECT * FROM Call ORDER BY received_time DESC", null)
        val result = mutableListOf<CallModel>()

        while (cursor.moveToNext()) {
            val callId = cursor.getString(cursor.getColumnIndexOrThrow("call_id"))
            val caller = cursor.getString(cursor.getColumnIndexOrThrow("caller"))
            result.add(CallModel(callId, caller))
        }

        cursor.close()
        db.close()
        return result
    }

    // 추가된 함수
    fun getCallerByCallId(callId: String): String? {
        val db = dbHelper.readableDatabase
        val cursor = db.rawQuery("SELECT caller FROM Call WHERE call_id = ?", arrayOf(callId))

        var caller: String? = null
        if (cursor.moveToFirst()) {
            caller = cursor.getString(cursor.getColumnIndexOrThrow("caller"))
        }

        cursor.close()
        db.close()
        return caller
    }


}
