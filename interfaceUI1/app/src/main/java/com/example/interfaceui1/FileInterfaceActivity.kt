package com.example.interfaceui1

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.CheckBox
import android.widget.ImageButton
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
    private lateinit var deleteButton: ImageButton
    private lateinit var selectAllButton: CheckBox

    private val pickVideo = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            importVideo(it)
        }
    }

    private fun importVideo(uri: Uri) {
        Thread {
            var dest: File? = null

            try {
                val name = contentResolver.query(
                    uri,
                    arrayOf(OpenableColumns.DISPLAY_NAME),
                    null,
                    null,
                    null
                )?.use {
                    if (it.moveToFirst()) it.getString(0) else null
                } ?: "import_${System.currentTimeMillis()}.mp4"

                val folder = File(filesDir, "MyFiles")

                // Remove any path pieces from the filename
                var target = File(folder, File(name).name)

                // Prevent duplicate filenames
                if (target.exists()) {
                    target = File(
                        folder,
                        "${System.currentTimeMillis()}_${target.name}"
                    )
                }

                dest = target

                contentResolver.openInputStream(uri)!!.use { input ->
                    target.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }

                runOnUiThread {
                    adapter.addFile(target)
                    Toast.makeText(
                        this,
                        "Video imported",
                        Toast.LENGTH_SHORT
                    ).show()
                }

            } catch (e: Exception) {

                // Delete incomplete file if import failed
                dest?.delete()

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Import failed",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }.start()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContentView(R.layout.file_interface)

        // Make navigation bar black with white navigation buttons
        WindowInsetsControllerCompat(
            window,
            window.decorView
        ).isAppearanceLightNavigationBars = false

        ViewCompat.setOnApplyWindowInsetsListener(
            findViewById(R.id.main)
        ) { v, insets ->

            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars()
            )

            v.setPadding(
                systemBars.left,
                systemBars.top,
                systemBars.right,
                systemBars.bottom
            )

            insets
        }

        // Import video button
        val backButton = findViewById<ImageButton>(
            R.id.fileupload
        )

        // Home/camera button
        val cameraButton = findViewById<ImageButton>(
            R.id.imageButton7
        )

        backButton.setOnClickListener {
            pickVideo.launch("video/*")
        }

        cameraButton.setOnClickListener {
            finish()
        }

        // MyFiles folder
        val myFilesFolder = File(filesDir, "MyFiles")

        if (!myFilesFolder.exists()) {
            myFilesFolder.mkdirs()
        }

        // S.I.G.H.T Pictures folder
        val sightPicturesFolder = File(
            android.os.Environment.getExternalStoragePublicDirectory(
                android.os.Environment.DIRECTORY_PICTURES
            ),
            "S.I.G.H.T"
        )

        if (!sightPicturesFolder.exists()) {
            sightPicturesFolder.mkdirs()
        }

        // Get files from both folders
        val files =
            (myFilesFolder.listFiles()?.filter { it.isFile }
                ?: emptyList()) +
                    (sightPicturesFolder.listFiles()?.filter { it.isFile }
                        ?: emptyList())

        // Sort newest first
        val sortedFiles = files
            .sortedByDescending { it.lastModified() }
            .toMutableList()

        // Connect XML buttons
        deleteButton = findViewById(R.id.deleteButton)
        selectAllButton = findViewById(R.id.selectAllButton)

        // Create adapter
        adapter = FileAdapter(sortedFiles) {
            updateButtons()
        }

        // RecyclerView
        val fileList = findViewById<RecyclerView>(
            R.id.fileList
        )

        fileList.layoutManager = LinearLayoutManager(this)
        fileList.adapter = adapter

        // Delete button
        // First tap = selection mode
        // Second tap = delete selected files
        deleteButton.setOnClickListener {

            when {

                // Enter selection mode
                !adapter.selectionMode -> {
                    adapter.selectionMode = true
                    updateButtons()
                }

                // Selection mode but nothing selected
                adapter.selected.isEmpty() -> {
                    Toast.makeText(
                        this,
                        "Select at least one file",
                        Toast.LENGTH_SHORT
                    ).show()
                }

                // Files selected
                else -> {
                    confirmDelete()
                }
            }
        }

        // Select all checkbox
        selectAllButton.setOnClickListener {

            adapter.toggleAll()

            updateButtons()
        }

        // Back button behavior
        // First back = leave selection mode
        // Second back = close screen
        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {

                override fun handleOnBackPressed() {

                    if (adapter.selectionMode) {

                        adapter.selectionMode = false
                        updateButtons()

                    } else {

                        isEnabled = false
                        onBackPressedDispatcher.onBackPressed()
                    }
                }
            }
        )
    }

    private fun updateButtons() {

        val mode = adapter.selectionMode

        // Show checkbox only in selection mode
        selectAllButton.visibility =
            if (mode) View.VISIBLE else View.GONE

        // Checkbox checked when every file is selected
        selectAllButton.isChecked =
            mode && adapter.allSelected()

        // Delete button becomes brighter when files are selected
        deleteButton.alpha =
            if (mode && adapter.selected.isNotEmpty()) {
                1f
            } else {
                0.6f
            }
    }

    private fun confirmDelete() {

        val count = adapter.selected.size

        AlertDialog.Builder(this)

            .setTitle("Delete files")

            .setMessage(
                "Permanently delete $count file(s)? This can't be undone."
            )

            .setPositiveButton("Delete") { _, _ ->

                val failed = adapter.deleteSelected()

                if (failed > 0) {
                    Toast.makeText(
                        this,
                        "$failed file(s) couldn't be deleted",
                        Toast.LENGTH_LONG
                    ).show()
                }

                adapter.selectionMode = false

                updateButtons()
            }

            .setNegativeButton("Cancel", null)

            .show()
    }
}