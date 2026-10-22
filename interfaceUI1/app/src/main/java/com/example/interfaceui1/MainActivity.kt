package com.example.interfaceui1

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.WindowManager
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

    private val pickVideoLauncher = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.GetContent()
    ) { uri: android.net.Uri? ->
        uri?.let {
            val intent = Intent(this, VideoAnnotationActivity::class.java).apply {
                data = it
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(intent)
        }
    }

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


        // Folder button: Tap to view saved recordings, Long-press to import external video
        fileButton.setOnClickListener {
            startActivity(Intent(this, FileInterfaceActivity::class.java))
        }
        fileButton.setOnLongClickListener {
            pickVideoLauncher.launch("video/*")
            true
        }

        // Camera + sign recognition + speech
        signEngine = SignEngine(
            activity = this,
            previewView = previewView,
            onSpoken = { phrase -> caption.text = phrase },
            onError = { msg -> Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }
        )

        // Light: tap the bulb to open Off / Auto / On. Back cam = flashlight, front cam = screen fill light.
        val lightButton = findViewById<ImageButton>(R.id.lightButton)
        val lightMenu = findViewById<View>(R.id.lightMenu)
        val fillGlow = findViewById<View>(R.id.fillGlow)
        val lightChoices = mapOf(
            findViewById<TextView>(R.id.lightOff) to SignEngine.LightMode.OFF,
            findViewById<TextView>(R.id.lightAuto) to SignEngine.LightMode.AUTO,
            findViewById<TextView>(R.id.lightOn) to SignEngine.LightMode.ON
        )

        fun showLightMode(mode: SignEngine.LightMode) {
            lightChoices.forEach { (view, m) ->
                view.setBackgroundResource(
                    if (m == mode) R.drawable.bg_sight_button_primary else R.drawable.bg_sight_button
                )
            }
            lightButton.alpha = if (mode == SignEngine.LightMode.OFF) 0.5f else 1f
        }
        showLightMode(SignEngine.LightMode.OFF)

        signEngine.onLightModeChanged = { showLightMode(it) }
        signEngine.onFillLight = { lit ->
            fillGlow.visibility = if (lit) View.VISIBLE else View.GONE
            window.attributes = window.attributes.apply {
                screenBrightness = if (lit) 1f else WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
            }
        }

        lightButton.setOnClickListener {
            lightMenu.visibility = if (lightMenu.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        lightChoices.forEach { (view, mode) ->
            view.setOnClickListener {
                signEngine.setLightMode(mode)
                lightMenu.visibility = View.GONE
            }
        }

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

        // Play/Record button: Records video with real-time sign recognition and saves to Pictures/S.I.G.H.T
        val playButton = findViewById<ImageButton>(R.id.playButton)
        playButton?.setOnClickListener {
            signEngine.toggleRecording { isRecording ->
                playButton.alpha = if (isRecording) 0.5f else 1.0f
            }
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

    override fun onStop() {
        signEngine.setLightMode(SignEngine.LightMode.OFF)
        super.onStop()
    }

    override fun onDestroy() {
        signEngine.release()
        super.onDestroy()
    }
}