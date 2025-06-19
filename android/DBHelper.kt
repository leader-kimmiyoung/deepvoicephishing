package com.quicinc.chatapp

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class DBHelper(context: Context) : SQLiteOpenHelper(context, "VoicePhishing.db", null, 1) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE Call (
                call_id TEXT PRIMARY KEY,
                caller TEXT
            );
        """.trimIndent())

        db.execSQL("""
            CREATE TABLE ChunkResult (
                call_id TEXT,
                chunk_number INTEGER,
                is_deepfake TEXT,
                is_phishing TEXT,
                reason TEXT,
                PRIMARY KEY(call_id, chunk_number),
                FOREIGN KEY(call_id) REFERENCES Call(call_id)
            );
        """.trimIndent())
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS ChunkResult")
        db.execSQL("DROP TABLE IF EXISTS Call")
        onCreate(db)
    }
}
