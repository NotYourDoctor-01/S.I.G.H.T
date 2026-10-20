package com.example.interfaceui1

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import kotlin.math.sqrt

class SignEngine(
    private val activity: AppCompatActivity,
    private val previewView: PreviewView,
    private val onSpoken: (String) -> Unit,
    private val onError: (String) -> Unit
) : TextToSpeech.OnInitListener {

    private data class FrameEntry(
        val timestamp: Float,
        val features: FloatArray,
        val hasHands: Boolean
    )

    private data class PredictionVote(
        val labelIdx: Int,
        val confidence: Float
    )

    private var muted = false
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    private var handLandmarker: HandLandmarker? = null
    private var tfliteInterpreter: Interpreter? = null

    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private var clipSeconds = 4.07f
    private var numFrames = 30
    private var numFeatures = 132
    private var labels = listOf<String>()

    private val timeBuffer = ArrayDeque<FrameEntry>()
    private val votes = ArrayDeque<PredictionVote>()
    private var frameCount = 0

    private var lastSpokenPhrase: String? = null
    private var lastSpokenTime: Long = 0L
    private val cooldownMs = 3000L

    companion object {
        private const val TAG = "SignEngine"
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
        private const val CONFIDENCE_THRESHOLD = 0.70f
        private const val VOTE_WINDOW = 8
        private const val REQUIRED_VOTES = 5
    }

    init {
        initTts()
        loadModelAndLabels()
        setupHandLandmarker()
    }

    private fun initTts() {
        tts = TextToSpeech(activity.applicationContext, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG, "TTS Language not supported")
            } else {
                ttsReady = true
            }
        } else {
            Log.e(TAG, "TTS Initialization failed")
        }
    }

    private fun loadModelAndLabels() {
        try {
            val assetManager = activity.assets

            // Load model_info.json
            try {
                val jsonString = assetManager.open("model_info.json").bufferedReader().use { it.readText() }
                val jsonObject = JSONObject(jsonString)
                if (jsonObject.has("num_frames")) numFrames = jsonObject.getInt("num_frames")
                if (jsonObject.has("num_features")) numFeatures = jsonObject.getInt("num_features")
                if (jsonObject.has("clip_seconds")) clipSeconds = jsonObject.getDouble("clip_seconds").toFloat()
            } catch (e: Exception) {
                Log.w(TAG, "Could not load model_info.json, using defaults", e)
            }

            // Load labels.txt
            labels = assetManager.open("labels.txt").bufferedReader().useLines { lines ->
                lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }

            // Load sign_model.tflite
            val modelBuffer = loadModelFile("sign_model.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            tfliteInterpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model or labels", e)
            activity.runOnUiThread { onError("Failed to load sign recognition model: ${e.message ?: "Unknown error"}") }
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
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
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, _ ->
                    processHandLandmarks(result)
                }
                .setErrorListener { error ->
                    Log.e(TAG, "MediaPipe HandLandmarker error: ${error.message}")
                }
                .build()

            handLandmarker = HandLandmarker.createFromOptions(activity, options)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing HandLandmarker", e)
            activity.runOnUiThread { onError("Failed to initialize hand tracking: ${e.message ?: "Unknown error"}") }
        }
    }

    fun start() {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                activity,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                activity.runOnUiThread {
                    onError("Camera permission is required to display camera preview and recognize signs.")
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(activity)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }

                val cameraSelector = when {
                    provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) -> CameraSelector.DEFAULT_FRONT_CAMERA
                    provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) -> CameraSelector.DEFAULT_BACK_CAMERA
                    else -> throw IllegalStateException("No available camera found on this device")
                }

                provider.unbindAll()
                provider.bindToLifecycle(
                    activity,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start camera", e)
                activity.runOnUiThread { onError("Failed to start camera: ${e.message ?: "Unknown error"}") }
            }
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val landmarker = handLandmarker
        if (landmarker == null) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees

            val matrix = Matrix()
            matrix.postRotate(rotation.toFloat())
            matrix.postScale(-1f, 1f) // Mirror horizontally like Python --flip

            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

            val mpImage = BitmapImageBuilder(rotatedBitmap).build()
            val frameTimestampMs = SystemClock.uptimeMillis()

            landmarker.detectAsync(mpImage, frameTimestampMs)
        } catch (e: Exception) {
            Log.e(TAG, "Error processing camera frame", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun computeHandFeatures(landmarks: List<NormalizedLandmark>): FloatArray {
        if (landmarks.isEmpty()) return FloatArray(66)

        val wristX = landmarks[0].x()
        val wristY = landmarks[0].y()
        val wristZ = landmarks[0].z()

        // Landmark 9 is middle finger MCP joint (middle finger base)
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

        // Wrist absolute coordinates
        features[63] = wristX
        features[64] = wristY
        features[65] = wristZ

        return features
    }

    private fun processHandLandmarks(result: HandLandmarkerResult) {
        val features = FloatArray(numFeatures) // 132 floats

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

        val now = SystemClock.uptimeMillis() / 1000.0f
        val hasHands = landmarksList.isNotEmpty()

        synchronized(timeBuffer) {
            timeBuffer.addLast(FrameEntry(now, features, hasHands))
            while (timeBuffer.isNotEmpty() && (now - timeBuffer.first.timestamp > clipSeconds)) {
                timeBuffer.removeFirst()
            }

            val span = if (timeBuffer.size > 1) timeBuffer.last.timestamp - timeBuffer.first.timestamp else 0f
            val handCount = timeBuffer.count { it.hasHands }
            val handFrac = if (timeBuffer.isNotEmpty()) handCount.toFloat() / timeBuffer.size else 0f

            frameCount++
            if (frameCount % 3 == 0 && span >= 0.8f * clipSeconds && handFrac >= 0.3f) {
                runInferenceFromBuffer()
            }
        }
    }

    private fun runInferenceFromBuffer() {
        val interpreter = tfliteInterpreter ?: return
        if (labels.isEmpty()) return

        val snapshot: List<FrameEntry>
        synchronized(timeBuffer) {
            if (timeBuffer.isEmpty()) return
            snapshot = ArrayList(timeBuffer)
        }

        if (snapshot.size < 5) return

        // Sample 30 frames evenly across the time window (matching Python np.linspace)
        val sampledSequence = Array(1) { Array(numFrames) { FloatArray(numFeatures) } }
        val maxIdx = snapshot.size - 1

        for (i in 0 until numFrames) {
            val idx = (i * maxIdx.toFloat() / (numFrames - 1)).roundToInt().coerceIn(0, maxIdx)
            sampledSequence[0][i] = snapshot[idx].features
        }

        val numClasses = labels.size
        val output = Array(1) { FloatArray(numClasses) }

        try {
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

            if (bestIdx in labels.indices) {
                votes.addLast(PredictionVote(bestIdx, bestScore))
                while (votes.size > VOTE_WINDOW) {
                    votes.removeFirst()
                }

                if (votes.size == VOTE_WINDOW) {
                    val counts = HashMap<Int, Int>()
                    for (vote in votes) {
                        counts[vote.labelIdx] = (counts[vote.labelIdx] ?: 0) + 1
                    }

                    var topLabelIdx = -1
                    var topCount = 0
                    for ((lbl, cnt) in counts) {
                        if (cnt > topCount) {
                            topCount = cnt
                            topLabelIdx = lbl
                        }
                    }

                    var sumConf = 0.0f
                    for (vote in votes) {
                        if (vote.labelIdx == topLabelIdx) sumConf += vote.confidence
                    }
                    val avgConf = sumConf / topCount

                    if (topCount >= REQUIRED_VOTES && avgConf >= CONFIDENCE_THRESHOLD) {
                        val name = labels[topLabelIdx]
                        val currentTime = System.currentTimeMillis()
                        val heldLong = cooldownMs > 0 && (currentTime - lastSpokenTime) > cooldownMs

                        if (name.equals("none", ignoreCase = true)) {
                            lastSpokenPhrase = null
                        } else if (name != lastSpokenPhrase || heldLong) {
                            lastSpokenPhrase = name
                            lastSpokenTime = currentTime
                            val phrase = name.replace("_", " ")

                            activity.runOnUiThread {
                                onSpoken(phrase)
                                speakText(phrase)
                            }
                            votes.clear()
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error running TFLite inference", e)
        }
    }

    private fun speakText(text: String) {
        if (!muted && ttsReady && tts != null) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "SignEngineTTS")
        }
    }

    fun setMuted(muted: Boolean) {
        this.muted = muted
        if (muted) {
            tts?.stop()
        }
    }

    fun release() {
        try {
            cameraProvider?.unbindAll()
            cameraExecutor.shutdown()

            handLandmarker?.close()
            handLandmarker = null

            tfliteInterpreter?.close()
            tfliteInterpreter = null

            tts?.stop()
            tts?.shutdown()
            tts = null
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing SignEngine resources", e)
        }
    }
}
