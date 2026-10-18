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

class SignEngine(
    private val activity: AppCompatActivity,
    private val previewView: PreviewView,
    private val onSpoken: (String) -> Unit,
    private val onError: (String) -> Unit
) : TextToSpeech.OnInitListener {

    private var muted = false
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    private var handLandmarker: HandLandmarker? = null
    private var tfliteInterpreter: Interpreter? = null

    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private val frameBuffer = ArrayDeque<FloatArray>()
    private var numFrames = 30
    private var numFeatures = 132
    private var labels = listOf<String>()

    private var lastSpokenPhrase: String? = null
    private var lastSpokenTime: Long = 0L
    private val cooldownMs = 2000L

    companion object {
        private const val TAG = "SignEngine"
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
        private const val CONFIDENCE_THRESHOLD = 0.4f
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

            // Load model_info.json if available
            try {
                val jsonString = assetManager.open("model_info.json").bufferedReader().use { it.readText() }
                val jsonObject = JSONObject(jsonString)
                if (jsonObject.has("num_frames")) numFrames = jsonObject.getInt("num_frames")
                if (jsonObject.has("num_features")) numFeatures = jsonObject.getInt("num_features")
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
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
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

    private fun processHandLandmarks(result: HandLandmarkerResult) {
        val features = FloatArray(numFeatures) // 132 floats

        val landmarksList = result.landmarks()
        val handednessList = result.handedness()

        var leftHandIndex = -1
        var rightHandIndex = -1

        for (i in handednessList.indices) {
            val categories = handednessList[i]
            if (categories.isNotEmpty()) {
                val categoryName = categories[0].categoryName()
                if (categoryName.equals("Left", ignoreCase = true) && leftHandIndex == -1) {
                    leftHandIndex = i
                } else if (categoryName.equals("Right", ignoreCase = true) && rightHandIndex == -1) {
                    rightHandIndex = i
                }
            }
        }

        // If handedness isn't specified explicitly, assign detected hands in order
        if (leftHandIndex == -1 && rightHandIndex == -1) {
            if (landmarksList.size > 0) leftHandIndex = 0
            if (landmarksList.size > 1) rightHandIndex = 1
        } else if (leftHandIndex == -1 && landmarksList.size > 1 && rightHandIndex != 0) {
            leftHandIndex = 0
        } else if (rightHandIndex == -1 && landmarksList.size > 1 && leftHandIndex != 1) {
            rightHandIndex = 1
        }

        // Fill Left Hand Slot (indices 0..65)
        if (leftHandIndex != -1 && leftHandIndex < landmarksList.size) {
            val landmarks = landmarksList[leftHandIndex]
            val score = if (leftHandIndex < handednessList.size && handednessList[leftHandIndex].isNotEmpty()) {
                handednessList[leftHandIndex][0].score()
            } else 1.0f

            for (j in 0 until minOf(21, landmarks.size)) {
                val lm = landmarks[j]
                features[j * 3] = lm.x()
                features[j * 3 + 1] = lm.y()
                features[j * 3 + 2] = lm.z()
            }
            features[63] = 1.0f  // is_present
            features[64] = score // confidence score
            features[65] = 0.0f  // handedness (0.0 = Left)
        }

        // Fill Right Hand Slot (indices 66..131)
        if (rightHandIndex != -1 && rightHandIndex < landmarksList.size) {
            val landmarks = landmarksList[rightHandIndex]
            val score = if (rightHandIndex < handednessList.size && handednessList[rightHandIndex].isNotEmpty()) {
                handednessList[rightHandIndex][0].score()
            } else 1.0f

            val offset = 66
            for (j in 0 until minOf(21, landmarks.size)) {
                val lm = landmarks[j]
                features[offset + j * 3] = lm.x()
                features[offset + j * 3 + 1] = lm.y()
                features[offset + j * 3 + 2] = lm.z()
            }
            features[offset + 63] = 1.0f  // is_present
            features[offset + 64] = score // confidence score
            features[offset + 65] = 1.0f  // handedness (1.0 = Right)
        }

        // Add to frame buffer
        synchronized(frameBuffer) {
            frameBuffer.addLast(features)
            while (frameBuffer.size > numFrames) {
                frameBuffer.removeFirst()
            }

            if (frameBuffer.size == numFrames) {
                runInference()
            }
        }
    }

    private fun runInference() {
        val interpreter = tfliteInterpreter ?: return
        if (labels.isEmpty()) return

        // Prepare input: [1, 30, 132]
        val input = Array(1) { Array(numFrames) { FloatArray(numFeatures) } }
        synchronized(frameBuffer) {
            var idx = 0
            for (frame in frameBuffer) {
                input[0][idx] = frame
                idx++
            }
        }

        val numClasses = labels.size
        val output = Array(1) { FloatArray(numClasses) }

        try {
            interpreter.run(input, output)

            val predictions = output[0]
            var maxIdx = -1
            var maxScore = -1.0f

            for (i in predictions.indices) {
                if (predictions[i] > maxScore) {
                    maxScore = predictions[i]
                    maxIdx = i
                }
            }

            if (maxIdx in labels.indices && maxScore >= CONFIDENCE_THRESHOLD) {
                val recognizedPhrase = labels[maxIdx]
                val currentTime = System.currentTimeMillis()

                if (recognizedPhrase != lastSpokenPhrase || (currentTime - lastSpokenTime) > cooldownMs) {
                    lastSpokenPhrase = recognizedPhrase
                    lastSpokenTime = currentTime

                    activity.runOnUiThread {
                        onSpoken(recognizedPhrase)
                        speakText(recognizedPhrase)
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
