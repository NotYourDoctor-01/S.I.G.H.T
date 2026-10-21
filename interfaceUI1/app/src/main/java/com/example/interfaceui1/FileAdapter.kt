package com.example.interfaceui1

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class FileAdapter(
    private val files: MutableList<File>,
    private val onSelectionChanged: () -> Unit
) : RecyclerView.Adapter<FileAdapter.FileViewHolder>() {

    val selected = mutableSetOf<File>()

    var selectionMode = false
        set(value) {
            field = value
            selected.clear()
            notifyDataSetChanged()
            onSelectionChanged()
        }

    fun allSelected() = files.isNotEmpty() && selected.size == files.size

    fun toggleAll() {
        if (allSelected()) selected.clear() else selected.addAll(files)
        notifyDataSetChanged()
        onSelectionChanged()
    }

    fun addFile(file: File) {
        files.add(0, file)
        notifyItemInserted(0)
    }

    /** Deletes the selected files, returns how many failed to delete. */
    fun deleteSelected(): Int {
        var failed = 0
        selected.toList().forEach { if (it.delete()) files.remove(it) else failed++ }
        selected.clear()
        notifyDataSetChanged()
        onSelectionChanged()
        return failed
    }

    class FileViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val fileName: TextView = itemView.findViewById(R.id.fileName)
        val fileDetails: TextView = itemView.findViewById(R.id.fileDetails)
        val playIcon: View = itemView.findViewById(R.id.playIcon)
        val fileCheck: CheckBox = itemView.findViewById(R.id.fileCheck)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FileViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_file, parent, false)

        return FileViewHolder(view)
    }

    override fun onBindViewHolder(holder: FileViewHolder, position: Int) {
        val file = files[position]

        // Display the actual file name
        holder.fileName.text = file.name

        // Read the file size
        val sizeKB = file.length() / 1024

        // Read the last modified date
        val date = SimpleDateFormat(
            "MMM dd, yyyy",
            Locale.getDefault()
        ).format(Date(file.lastModified()))

        // Display file type, size, and date
        val type = when (file.extension.lowercase()) {
            "mp4" -> "Video recording"
            "jpg", "jpeg", "png" -> "Image"
            "txt" -> "Text file"
            else -> "File"
        }

        holder.fileDetails.text = "$type • ${sizeKB} KB • $date"

        // Selection mode swaps the play icon for a checkbox
        holder.playIcon.visibility = if (selectionMode) View.GONE else View.VISIBLE
        holder.fileCheck.visibility = if (selectionMode) View.VISIBLE else View.GONE
        holder.fileCheck.isChecked = file in selected

        holder.itemView.setOnClickListener {
            if (selectionMode) {
                if (!selected.add(file)) selected.remove(file)
                notifyItemChanged(holder.bindingAdapterPosition)
                onSelectionChanged()
            } else if (file.extension.lowercase() in listOf("mp4", "mov", "avi", "mkv", "webm")) {
                val context = holder.itemView.context
                val intent = android.content.Intent(context, VideoAnnotationActivity::class.java).apply {
                    data = android.net.Uri.fromFile(file)
                }
                context.startActivity(intent)
            }
        }
    }

    override fun getItemCount(): Int = files.size
}