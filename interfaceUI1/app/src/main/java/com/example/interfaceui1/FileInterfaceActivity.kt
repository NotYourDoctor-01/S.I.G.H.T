package com.example.interfaceui1

import android.content.Intent
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

        backButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }

        // Create the folder if it doesn't exist
        val folder = File(filesDir, "MyFiles")
        if (!folder.exists()) {
            folder.mkdirs()
        }

        // Get the files stored in the folder
        val files = folder.listFiles()
            ?.filter { it.isFile }
            ?.sortedByDescending { it.lastModified() }
            ?: emptyList()

        // Display the files in the RecyclerView
        val fileList = findViewById<RecyclerView>(R.id.fileList)
        fileList.layoutManager = LinearLayoutManager(this)
        fileList.adapter = FileAdapter(files)
    }
}
