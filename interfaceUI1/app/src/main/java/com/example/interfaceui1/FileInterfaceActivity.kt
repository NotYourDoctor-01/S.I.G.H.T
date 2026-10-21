package com.example.interfaceui1

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File

class FileInterfaceActivity : AppCompatActivity() {

    private lateinit var adapter: FileAdapter
    private lateinit var deleteButton: TextView
    private lateinit var selectAllButton: TextView

    private val pickVideo = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { importVideo(it) }
    }

    private fun importVideo(uri: Uri) {
        Thread {
            var dest: File? = null
            try {
                val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                    ?.use { if (it.moveToFirst()) it.getString(0) else null }
                    ?: "import_${System.currentTimeMillis()}.mp4"
                val folder = File(filesDir, "MyFiles")
                // File(name).name strips any path pieces the picker might send
                var target = File(folder, File(name).name)
                if (target.exists()) target = File(folder, "${System.currentTimeMillis()}_${target.name}")
                dest = target
                contentResolver.openInputStream(uri)!!.use { input ->
                    target.outputStream().use { input.copyTo(it) }
                }
                runOnUiThread {
                    adapter.addFile(target)
                    Toast.makeText(this, "Video imported", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                dest?.delete() // don't leave a half-copied file behind
                runOnUiThread { Toast.makeText(this, "Import failed", Toast.LENGTH_LONG).show() }
            }
        }.start()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.file_interface)
        WindowInsetsControllerCompat(window, window.decorView)
            .isAppearanceLightNavigationBars = false
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(
                systemBars.left,
                systemBars.top,
                systemBars.right,
                systemBars.bottom
            )
            insets
        }

        val backButton = findViewById<ImageButton>(R.id.fileupload)
        val cameraButton = findViewById<ImageButton>(R.id.imageButton7)

        backButton?.setOnClickListener { pickVideo.launch("video/*") }
        cameraButton?.setOnClickListener { finish() }

        val myFilesFolder = File(filesDir, "MyFiles")
        if (!myFilesFolder.exists()) {
            myFilesFolder.mkdirs()
        }

        val sightPicturesFolder = File(
            android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_PICTURES),
            "S.I.G.H.T"
        )
        if (!sightPicturesFolder.exists()) {
            sightPicturesFolder.mkdirs()
        }

        // Get the files stored in both folders
        val files = (myFilesFolder.listFiles()?.filter { it.isFile } ?: emptyList()) +
                (sightPicturesFolder.listFiles()?.filter { it.isFile } ?: emptyList())
        val sortedFiles = files.sortedByDescending { it.lastModified() }.toMutableList()

        deleteButton = findViewById(R.id.deleteButton)
        selectAllButton = findViewById(R.id.selectAllButton)

        adapter = FileAdapter(sortedFiles) { updateButtons() }

        // Display the files in the RecyclerView
        val fileList = findViewById<RecyclerView>(R.id.fileList)
        fileList.layoutManager = LinearLayoutManager(this)
        fileList.adapter = adapter

        // Delete: 1st tap enters selection mode, 2nd tap confirms deletion
        deleteButton.setOnClickListener {
            when {
                !adapter.selectionMode -> adapter.selectionMode = true
                adapter.selected.isEmpty() ->
                    Toast.makeText(this, "Select at least one file", Toast.LENGTH_SHORT).show()
                else -> confirmDelete()
            }
        }

        selectAllButton.setOnClickListener { adapter.toggleAll() }

        // Back press leaves selection mode first, then closes the screen
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (adapter.selectionMode) {
                    adapter.selectionMode = false
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })
    }

    private fun updateButtons() {
        val mode = adapter.selectionMode
        selectAllButton.visibility = if (mode) View.VISIBLE else View.GONE
        selectAllButton.text = if (adapter.allSelected()) "Deselect all" else "Select all"
        deleteButton.text = if (mode && adapter.selected.isNotEmpty()) "Delete (${adapter.selected.size})" else "Delete"
    }

    private fun confirmDelete() {
        val count = adapter.selected.size
        AlertDialog.Builder(this)
            .setTitle("Delete files")
            .setMessage("Permanently delete $count file(s)? This can't be undone.")
            .setPositiveButton("Delete") { _, _ ->
                val failed = adapter.deleteSelected()
                if (failed > 0) {
                    Toast.makeText(this, "$failed file(s) couldn't be deleted", Toast.LENGTH_LONG).show()
                }
                adapter.selectionMode = false
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}