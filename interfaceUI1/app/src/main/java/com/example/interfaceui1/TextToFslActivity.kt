package com.example.interfaceui1

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebView
import android.widget.Button
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/**
 * Screen opened by txtButton on the main UI.
 * Shows the offline "Text to FSL" page (bundled in assets/fsl) and a Back button
 * that returns to the main camera UI.
 */
class TextToFslActivity : AppCompatActivity() {

    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_text_to_fsl)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.textRoot)) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime()
            )
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        // Back button: close this screen and return to the main UI.
        findViewById<Button>(R.id.backButton).setOnClickListener { finish() }

        webView = findViewById(R.id.fslWebView)
        webView.settings.javaScriptEnabled = true   // the page is all JavaScript
        // The page is fully offline, so keep the WebView away from other files and the network.
        webView.settings.allowFileAccess = false
        webView.settings.allowContentAccess = false
        webView.setBackgroundColor(0)               // no white flash before the page paints

        if (savedInstanceState == null) {
            webView.loadUrl("file:///android_asset/fsl/index.html")
        } else {
            webView.restoreState(savedInstanceState)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        webView.saveState(outState)
    }

    override fun onResume() {
        super.onResume()
        webView.onResume()
    }

    override fun onPause() {
        webView.onPause()   // stops the avatar animation while the screen is hidden
        super.onPause()
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }
}
