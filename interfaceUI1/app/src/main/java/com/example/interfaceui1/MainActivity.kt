package com.example.interfaceui1

import android.content.Intent
import android.os.Bundle
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File

class  MainActivity : AppCompatActivity() {

    private lateinit var signEngine: SignEngine
    private var muted = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Create the folder if it doesn't exist
        val folder = File(filesDir, "MyFiles")
        if (!folder.exists()) {
            folder.mkdirs()
        }



        val caption = findViewById<TextView>(R.id.captionText)
        val muteButton = findViewById<ImageButton>(R.id.muteButton)
        val flipButton = findViewById<ImageButton>(R.id.flipButton)
        val previewView = findViewById<PreviewView>(R.id.previewView)
        val fileButton = findViewById<ImageButton>(R.id.folderButton)


        //folder button
        fileButton.setOnClickListener {
            startActivity(Intent(this, FileInterfaceActivity::class.java))
        }

        // Camera + sign recognition + speech
        signEngine = SignEngine(
            activity = this,
            previewView = previewView,
            onSpoken = { phrase -> caption.text = phrase },
            onError = { msg -> Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }
        )

        previewView.post {
            signEngine.start()
        }

        // Mute button: turns spoken output on/off (dimmed = muted)
        muteButton.setOnClickListener {
            muted = !muted
            signEngine.setMuted(muted)
            muteButton.alpha = if (muted) 0.4f else 1f
        }

        // Flip camera button
        flipButton?.setOnClickListener {
            signEngine.toggleCamera()
        }

        // Text button: opens the Text to FSL screen (it has its own Back button)
        findViewById<ImageButton>(R.id.txtButton).setOnClickListener {
            startActivity(Intent(this, TextToFslActivity::class.java))
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        signEngine.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    override fun onDestroy() {
        signEngine.release()
        super.onDestroy()
    }
}