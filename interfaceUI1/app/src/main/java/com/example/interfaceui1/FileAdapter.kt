package com.example.interfaceui1

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class FileAdapter(private val files: List<File>) :
    RecyclerView.Adapter<FileAdapter.FileViewHolder>() {

    class FileViewHolder(itemView: View) :
        RecyclerView.ViewHolder(itemView) {

        val fileName: TextView = itemView.findViewById(R.id.fileName)
        val fileDetails: TextView = itemView.findViewById(R.id.fileDetails)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FileViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_file, parent, false)

        return FileViewHolder(view)
    }

    override fun onBindViewHolder(holder: FileViewHolder, position: Int) {
        val file = files[position]

        holder.fileName.text = file.name

        val sizeKB = file.length() / 1024
        val date = SimpleDateFormat(
            "MMM dd, yyyy",
            Locale.getDefault()
        ).format(Date(file.lastModified()))

        holder.fileDetails.text = "$sizeKB KB • $date"
    }

    override fun getItemCount(): Int {
        return files.size
    }
}
