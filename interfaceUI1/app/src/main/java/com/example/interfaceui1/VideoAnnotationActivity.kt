package com.example.interfaceui1

import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.VideoView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import kotlin.math.sqrt

class VideoAnnotationActivity : AppCompatActivity() {

    private data class SubtitleSegment(
        val startTimeMs: Long,
        val endTimeMs: Long,
        val text: String
    )

    private data class FrameEntry(
        val timestampSec: Float,
        val features: FloatArray,
        val hasHands: Boolean
    )

    private lateinit var videoView: VideoView
    private lateinit var subtitleText: TextView
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar

    private var videoUri: Uri? = null
    @Volatile private var destroyed = false
    private val subtitleSegments = ArrayList<SubtitleSegment>()

    private val handler = Handler(Looper.getMainLooper())
    private val updateSubtitleRunnable = object : Runnable {
        override fun run() {
            if (videoView.isPlaying) {
                val currentPos = videoView.currentPosition.toLong()
                val activeSegment = subtitleSegments.firstOrNull {
                    currentPos >= it.startTimeMs && currentPos <= it.endTimeMs
                }

                if (activeSegment != null) {
                    subtitleText.text = activeSegment.text
                    subtitleText.visibility = View.VISIBLE
                } else {
                    subtitleText.visibility = View.GONE
                }
            }
            handler.postDelayed(this, 100)
        }
    }

    private var handLandmarker: HandLandmarker? = null
    private var tfliteInterpreter: Interpreter? = null
    private var labels = listOf<String>()
    private var numFrames = 30
    private var numFeatures = 132
    private var clipSeconds = 4.07f

    companion object {
        private const val TAG = "VideoAnnotation"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_video_annotation)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.annotationRoot)) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime()
            )
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        findViewById<Button>(R.id.backButton).setOnClickListener { finish() }

        videoView = findViewById(R.id.videoView)
        subtitleText = findViewById(R.id.subtitleText)
        statusText = findViewById(R.id.statusText)
        progressBar = findViewById(R.id.progressBar)

        videoUri = intent.data

        if (videoUri == null) {
            statusText.text = "No video selected."
            progressBar.visibility = View.GONE
            return
        }

        // Already analyzed before? Load the saved subtitles and skip the analysis.
        val cache = cacheFileFor(videoUri!!)
        if (cache != null && loadCache(cache)) {
            progressBar.visibility = View.GONE
            statusText.text = "Loaded saved annotations. ${subtitleSegments.size} subtitle segment(s)."
            startVideoPlayback(videoUri!!)
            return
        }

        loadModelAndLabels()
        setupHandLandmarker()

        analyzeVideoAndPlay(videoUri!!, cache)
    }

    private fun loadModelAndLabels() {
        try {
            val jsonString = assets.open("model_info.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            if (jsonObject.has("num_frames")) numFrames = jsonObject.getInt("num_frames")
            if (jsonObject.has("num_features")) numFeatures = jsonObject.getInt("num_features")
            if (jsonObject.has("clip_seconds")) clipSeconds = jsonObject.getDouble("clip_seconds").toFloat()

            labels = assets.open("labels.txt").bufferedReader().useLines { lines ->
                lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }

            val fileDescriptor = assets.openFd("sign_model.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val modelBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )

            val options = Interpreter.Options().apply { setNumThreads(4) }
            tfliteInterpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model or labels", e)
        }
    }

    private fun setupHandLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setNumHands(2)
                .setRunningMode(RunningMode.IMAGE)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(this, options)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing HandLandmarker", e)
        }
    }

    private fun analyzeVideoAndPlay(uri: Uri, cache: File?) {
        val executor = Executors.newSingleThreadExecutor()
        executor.execute {
            val retriever = MediaMetadataRetriever()
            var complete = false // only a full, successful pass gets saved
            try {
                retriever.setDataSource(applicationContext, uri)
                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                val durationMs = durationStr?.toLongOrNull() ?: 0L

                val rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                val videoRotation = rotationStr?.toIntOrNull() ?: 0

                if (durationMs > 0) {
                    val frameIntervalMs = 100L // Process 10 fps
                    val totalSteps = (durationMs / frameIntervalMs).toInt().coerceAtLeast(1)

                    val timeBuffer = ArrayDeque<FrameEntry>()
                    val videoDurationSec = durationMs / 1000f
                    val requiredSpan = (0.7f * videoDurationSec).coerceAtMost(0.8f * clipSeconds).coerceAtLeast(0.5f)

                    for (step in 0..totalSteps) {
                        if (destroyed) return@execute // screen closed: don't save a half result
                        val currentTimeMs = step * frameIntervalMs
                        val frameUs = currentTimeMs * 1000L

                        val rawBitmap = retriever.getFrameAtTime(frameUs, MediaMetadataRetriever.OPTION_CLOSEST)
                        if (rawBitmap != null) {
                            val bitmap = if (videoRotation != 0) {
                                val matrix = android.graphics.Matrix().apply { postRotate(videoRotation.toFloat()) }
                                android.graphics.Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                            } else {
                                rawBitmap
                            }

                            val features = processBitmap(bitmap)
                            val hasHands = features.any { it != 0f }
                            val timestampSec = currentTimeMs / 1000f

                            timeBuffer.addLast(FrameEntry(timestampSec, features, hasHands))
                            while (timeBuffer.isNotEmpty() && (timestampSec - timeBuffer.first.timestampSec > clipSeconds)) {
                                timeBuffer.removeFirst()
                            }

                            val span = if (timeBuffer.size > 1) timeBuffer.last.timestampSec - timeBuffer.first.timestampSec else 0f
                            val handCount = timeBuffer.count { it.hasHands }
                            val handFrac = if (timeBuffer.isNotEmpty()) handCount.toFloat() / timeBuffer.size else 0f

                            if ((span >= requiredSpan || step == totalSteps) && handFrac >= 0.2f) {
                                val label = predictFromBuffer(timeBuffer)
                                if (label != null && !label.equals("none", ignoreCase = true)) {
                                    val phrase = label.replace("_", " ")
                                    val startMs = (currentTimeMs - (clipSeconds * 1000).toLong()).coerceAtLeast(0L)
                                    val endMs = currentTimeMs + 1200L

                                    val prev = subtitleSegments.lastOrNull()
                                    if (prev != null && prev.text == phrase && (startMs - prev.endTimeMs) < 1500L) {
                                        subtitleSegments[subtitleSegments.size - 1] = SubtitleSegment(prev.startTimeMs, endMs, phrase)
                                    } else {
                                        subtitleSegments.add(SubtitleSegment(startMs, endMs, phrase))
                                    }
                                }
                            }
                        }

                        val progress = (step * 100) / totalSteps
                        runOnUiThread {
                            progressBar.progress = progress
                        }
                    }
                    complete = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error analyzing video", e)
            } finally {
                try { retriever.release() } catch (ignored: Exception) {}
            }

            // Don't save if the model failed to load (the empty result would be wrong)
            if (complete && cache != null && handLandmarker != null && tfliteInterpreter != null) {
                saveCache(cache)
            }

            runOnUiThread {
                progressBar.visibility = View.GONE
                statusText.text = if (subtitleSegments.isEmpty()) {
                    "Video analysis complete. No sign phrases detected."
                } else {
                    "Video analysis complete. ${subtitleSegments.size} subtitle segment(s) annotated."
                }

                startVideoPlayback(uri)
            }
        }
    }

    private fun processBitmap(bitmap: android.graphics.Bitmap): FloatArray {
        val landmarker = handLandmarker ?: return FloatArray(numFeatures)
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result: HandLandmarkerResult = landmarker.detect(mpImage)
            extractFeaturesFromResult(result)
        } catch (e: Exception) {
            FloatArray(numFeatures)
        }
    }

    private fun extractFeaturesFromResult(result: HandLandmarkerResult): FloatArray {
        val features = FloatArray(numFeatures)
        val landmarksList = result.landmarks()
        val handednessList = result.handedness()

        for (i in landmarksList.indices) {
            val landmarks = landmarksList[i]
            val categories = if (i < handednessList.size) handednessList[i] else emptyList()
            val categoryName = if (categories.isNotEmpty()) categories[0].categoryName() else ""

            val slot = if (categoryName.equals("Left", ignoreCase = true)) 0 else 1
            val handFeats = computeHandFeatures(landmarks)
            val offset = slot * 66

            System.arraycopy(handFeats, 0, features, offset, 66)
        }
        return features
    }

    private fun computeHandFeatures(landmarks: List<NormalizedLandmark>): FloatArray {
        if (landmarks.isEmpty()) return FloatArray(66)

        val wristX = landmarks[0].x()
        val wristY = landmarks[0].y()
        val wristZ = landmarks[0].z()

        val rel9X = landmarks[9].x() - wristX
        val rel9Y = landmarks[9].y() - wristY
        val rel9Z = landmarks[9].z() - wristZ

        val norm = sqrt(rel9X * rel9X + rel9Y * rel9Y + rel9Z * rel9Z) + 1e-6f

        val features = FloatArray(66)
        for (i in 0 until minOf(21, landmarks.size)) {
            val lm = landmarks[i]
            features[i * 3] = (lm.x() - wristX) / norm
            features[i * 3 + 1] = (lm.y() - wristY) / norm
            features[i * 3 + 2] = (lm.z() - wristZ) / norm
        }

        features[63] = wristX
        features[64] = wristY
        features[65] = wristZ

        return features
    }

    private fun predictFromBuffer(timeBuffer: ArrayDeque<FrameEntry>): String? {
        val interpreter = tfliteInterpreter ?: return null
        if (labels.isEmpty() || timeBuffer.isEmpty()) return null

        val snapshot = ArrayList(timeBuffer)
        if (snapshot.size < 5) return null

        val sampledSequence = Array(1) { Array(numFrames) { FloatArray(numFeatures) } }
        val maxIdx = snapshot.size - 1

        for (i in 0 until numFrames) {
            val idx = (i * maxIdx.toFloat() / (numFrames - 1)).roundToInt().coerceIn(0, maxIdx)
            sampledSequence[0][i] = snapshot[idx].features
        }

        val numClasses = labels.size
        val output = Array(1) { FloatArray(numClasses) }

        return try {
            interpreter.run(sampledSequence, output)
            val predictions = output[0]
            var bestIdx = -1
            var bestScore = -1.0f

            for (i in predictions.indices) {
                if (predictions[i] > bestScore) {
                    bestScore = predictions[i]
                    bestIdx = i
                }
            }

            if (bestIdx in labels.indices && bestScore >= 0.35f) {
                labels[bestIdx]
            } else null
        } catch (e: Exception) {
            null
        }
    }

    // Saved results live in cacheDir. The name changes if the video changes or the app is
    // updated (a new model would give different subtitles). Only works for file:// videos.
    private fun cacheFileFor(uri: Uri): File? {
        if (uri.scheme != "file") return null
        val video = File(uri.path ?: return null)
        val appStamp = packageManager.getPackageInfo(packageName, 0).lastUpdateTime
        val folder = File(cacheDir, "annotations").apply { mkdirs() }
        return File(folder, "${video.absolutePath.hashCode()}_${video.lastModified()}_${video.length()}_$appStamp.json")
    }

    private fun loadCache(cache: File): Boolean {
        if (!cache.exists()) return false
        return try {
            val arr = JSONArray(cache.readText())
            val loaded = ArrayList<SubtitleSegment>()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                loaded.add(SubtitleSegment(o.getLong("start"), o.getLong("end"), o.getString("text")))
            }
            subtitleSegments.addAll(loaded)
            true
        } catch (e: Exception) {
            cache.delete() // unreadable, so analyze again
            false
        }
    }

    private fun saveCache(cache: File) {
        try {
            val arr = JSONArray()
            subtitleSegments.forEach {
                arr.put(JSONObject().put("start", it.startTimeMs).put("end", it.endTimeMs).put("text", it.text))
            }
            cache.writeText(arr.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Error saving annotations", e)
        }
    }

    private fun startVideoPlayback(uri: Uri) {
        videoView.setVideoURI(uri)
        videoView.setOnPreparedListener { mp: MediaPlayer ->
            mp.isLooping = true
            videoView.start()
            handler.post(updateSubtitleRunnable)
        }
    }

    override fun onDestroy() {
        destroyed = true
        handler.removeCallbacks(updateSubtitleRunnable)
        try {
            handLandmarker?.close()
            handLandmarker = null

            tfliteInterpreter?.close()
            tfliteInterpreter = null
        } catch (e: Exception) {
            Log.e(TAG, "Error closing resources", e)
        }
        super.onDestroy()
    }
}