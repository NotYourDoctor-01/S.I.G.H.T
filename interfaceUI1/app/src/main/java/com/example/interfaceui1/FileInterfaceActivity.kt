package com.example.interfaceui1


import android.os.Bundle
import android.widget.ImageButton
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File

class FileInterfaceActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.file_interface)


        val backButton = findViewById<ImageButton>(R.id.imageButton6)
        val cameraButton = findViewById<ImageButton>(R.id.imageButton7)

        backButton?.setOnClickListener { finish() }
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
        val sortedFiles = files.sortedByDescending { it.lastModified() }

        // Display the files in the RecyclerView
        val fileList = findViewById<RecyclerView>(R.id.fileList)
        fileList.layoutManager = LinearLayoutManager(this)
        fileList.adapter = FileAdapter(sortedFiles)
    }
}
