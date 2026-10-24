package com.echolens.app.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import com.echolens.app.MainActivity
import com.echolens.app.R
import com.echolens.app.audio.AudioEngine
import com.echolens.app.audio.SampleLibrary
import com.echolens.app.audio.neural.NeuralModels
import com.echolens.app.audio.neural.NeuralTextureProvisioner
import com.echolens.app.audio.neural.TextureCache
import com.echolens.app.core.HeadPose
import com.echolens.app.core.Material
import com.echolens.app.core.RuText
import com.echolens.app.core.TrackSnapshot
import com.echolens.app.head.HeadTracker
import com.echolens.app.util.Angles
import com.echolens.app.vision.DepthEstimator
import com.echolens.app.vision.FrameConverter
import com.echolens.app.vision.MaterialClassifier
import com.echolens.app.vision.SceneTracker
import com.echolens.app.vision.YoloDetector
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * Foreground service that owns the whole pipeline:
 * CameraX → TFLite detector → world-anchored scene tracker → binaural audio engine,
 * plus TTS announcements and haptics for newly detected objects.
 */
class AudioSceneService : LifecycleService() {

    companion object {
        const val ACTION_START = "com.echolens.app.START"
        const val ACTION_STOP = "com.echolens.app.STOP"

        const val PREFS = "echolens_settings"
        const val KEY_VOICE = "voice_announce"
        const val KEY_HAPTIC = "haptic"
        const val KEY_SCREEN = "keep_screen"
        const val KEY_INVERT = "invert_azimuth"
        const val KEY_SPEAKER = "speaker_mode"
        const val KEY_VOLUME = "master_volume"
        const val KEY_CONE_DEG = "cone_deg"
        const val KEY_STREAMS = "max_streams"
        const val KEY_NEURAL = "neural_textures"
        const val KEY_MODEL_ACCURATE = "detector_accurate"

        private const val CHANNEL_ID = "session"
        private const val NOTIFICATION_ID = 42
        private const val TAG = "EchoLensService"

        const val DEFAULT_VOLUME = 85
        const val DEFAULT_CONE_DEG = 30
        const val DEFAULT_STREAMS = 10
    }

    class UiState(
        val running: Boolean,
        val headYawDeg: Float,
        val fps: Float,
        val detectorMs: Float,
        val activeStreams: Int,
        val tracks: List<TrackSnapshot>,
        val frames: Long,
        val camError: String?,
        val depthMs: Int,
        val depthOn: Boolean,
        val neuralStatus: String,
        val neuralAvailable: Boolean,
        val materialClfOn: Boolean,
        val sampleCount: Int,
        val detectorModel: String
    )

    inner class LocalBinder : Binder() {
        val service: AudioSceneService get() = this@AudioSceneService
    }

    private val binder = LocalBinder()

    private var headTracker: HeadTracker? = null
    private var detector: YoloDetector? = null
    private var depthEstimator: DepthEstimator? = null
    private var materialClassifier: MaterialClassifier? = null
    private var sceneTracker: SceneTracker? = null
    private var audioEngine: AudioEngine? = null
    private var sampleLibrary: SampleLibrary? = null
    private var textureCache: TextureCache? = null
    private var provisioner: NeuralTextureProvisioner? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService? = null
    private var frameConverter: FrameConverter? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var tts: TextToSpeech? = null
    private var audioManager: AudioManager? = null
    private var audioFocusRequest: AudioFocusRequest? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile private var sessionRunning = false
    @Volatile private var ttsReady = false
    @Volatile private var lastTtsMs = 0L
    @Volatile private var fps = 0f
    @Volatile private var detectorMs = 0f
    @Volatile var camError: String? = null
        private set
    @Volatile private var frameCount = 0L
    private var rebindAttempts = 0
    @Volatile private var detectorModelName = ""

    // Analyzer-thread only
    private val frameTimes = ArrayDeque<Long>()

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
    }

    override fun onBind(intent: Intent): IBinder {
        super.onBind(intent)
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        when (intent?.action) {
            ACTION_STOP -> {
                stopSession()
                stopSelf()
            }
            else -> if (!sessionRunning) {
                rebindAttempts = 0
                frameCount = 0L
                camError = null
                startSession()
            }
        }
        return START_NOT_STICKY
    }

    // ------------------------------------------------------------ session

    private fun startSession() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "No camera permission — stopping")
            stopSelf()
            return
        }
        startInForeground()
        sessionRunning = true
        frameTimes.clear()

        requestAudioFocus()

        tts = TextToSpeech(applicationContext) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                val res = tts?.setLanguage(Locale.forLanguageTag("ru"))
                if (res == TextToSpeech.LANG_MISSING_DATA || res == TextToSpeech.LANG_NOT_SUPPORTED) {
                    // Fall back to the default locale — speech is a bonus, not a requirement.
                    tts?.language = Locale.getDefault()
                }
            }
        }

        headTracker = HeadTracker(this).also { it.start() }
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val accurate = prefs.getBoolean(KEY_MODEL_ACCURATE, true)
        try {
            detector = YoloDetector.create(this, accurate)
            detectorModelName = when {
                accurate -> "YOLOE-26s@640 · 426 классов"
                else -> "YOLO26-N@480 · COCO"
            }
            camError = null
        } catch (e: Exception) {
            Log.e(TAG, "Detector load failed", e)
            camError = "детектор: " + (e.message ?: "ошибка загрузки")
        }
        depthEstimator = try {
            DepthEstimator(this)
        } catch (e: Exception) {
            Log.w(TAG, "Depth model unavailable", e)
            null
        }
        sceneTracker = SceneTracker()
        frameConverter = FrameConverter()

        // Bundled sound texture library (no downloads needed).
        val samples = SampleLibrary(this)
        sampleLibrary = samples

        // Neural texture pipeline: cache + background on-device generation.
        val cache = TextureCache(this)
        textureCache = cache
        val prov = NeuralTextureProvisioner.get(this, cache)
        provisioner = prov

        // Optional visual material classifier (MINC-23).
        materialClassifier = try {
            MaterialClassifier.tryCreate(this)
        } catch (e: Exception) {
            Log.w(TAG, "Material classifier unavailable", e)
            null
        }

        val engine = AudioEngine(
            headProvider = { HeadPose(headTracker?.yaw ?: 0f, headTracker?.pitch ?: 0f) },
            tracksProvider = {
                sceneTracker?.snapshot(SystemClock.elapsedRealtime()) ?: emptyList()
            },
            onObjectActivated = { ruName -> onObjectActivated(ruName) },
            sampleLibrary = samples,
            textureCache = cache,
            provisioner = prov
        )
        audioEngine = engine
        applySettings()
        engine.startEngine()
        samples.preloadEvents(Material.values().toList())

        acquireWakeLock()
        startCamera()
    }

    fun applySettings() {
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        audioEngine?.settings?.apply {
            masterVolume = (prefs.getInt(KEY_VOLUME, DEFAULT_VOLUME) / 100f)
                .coerceIn(0.05f, 1f)
            coneWidthRad = Angles.rad(prefs.getInt(KEY_CONE_DEG, DEFAULT_CONE_DEG).toFloat())
            maxSources = prefs.getInt(KEY_STREAMS, DEFAULT_STREAMS).coerceIn(2, 16)
            invertAzimuth = prefs.getBoolean(KEY_INVERT, false)
            speakerMode = prefs.getBoolean(KEY_SPEAKER, true)
            neuralTextures = prefs.getBoolean(KEY_NEURAL, false)
        }
        provisioner?.enabled = prefs.getBoolean(KEY_NEURAL, false)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider
                // Default YUV_420_888 output and default resolution — the most
                // universally compatible camera configuration.
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                val executor = Executors.newSingleThreadExecutor()
                cameraExecutor = executor
                analysis.setAnalyzer(executor, ::analyzeFrame)
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                // Watchdog: if no frames arrive, rebind once, then report an error.
                mainHandler.postDelayed({
                    if (sessionRunning && frameCount == 0L) {
                        if (rebindAttempts < 1) {
                            rebindAttempts++
                            Log.w(TAG, "No camera frames — rebinding")
                            runCatching { cameraProvider?.unbindAll() }
                            startCamera()
                        } else {
                            camError = "камера не выдаёт кадры — закройте другие приложения с камерой"
                        }
                    }
                }, 3500)
            } catch (e: Exception) {
                Log.e(TAG, "Camera init failed", e)
                camError = "камера: " + (e.message ?: "ошибка запуска")
                stopSession()
                stopSelf()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            val detector = detector ?: return
            val scene = sceneTracker ?: return
            val converter = frameConverter ?: return
            val head = headTracker

            val t0 = SystemClock.elapsedRealtime()
            val bitmap = converter.toUprightBitmap(image)
            frameCount++

            // Relative depth on every 5th frame (~2-3 Hz).
            val depth = depthEstimator
            if (depth != null && frameCount % 5 == 0L) {
                depth.estimate(bitmap)
            }

            val detections = detector.detect(bitmap)
            val now = SystemClock.elapsedRealtime()

            // Attach relative depth to each detection.
            if (depth != null && depth.hasMap) {
                for (idx in detections.indices) {
                    val d = detections[idx]
                    val rd = depth.sampleAt(d.cxNorm, d.cyNorm)
                    if (rd != null) detections[idx] = d.withDepth(rd)
                }
            }

            // Optional material refinement — round-robin over the top-3
            // detections (~2 Hz each) instead of only the single best box.
            val clf = materialClassifier
            if (clf != null && frameCount % 10 == 0L && detections.isNotEmpty()) {
                val slot = ((frameCount / 10) % minOf(3, detections.size)).toInt()
                val d = detections[slot]
                val res = clf.classify(bitmap, d.cxNorm, d.cyNorm, d.hFrac)
                if (res != null) {
                    detections[slot] = d.withSpec(clf.refine(d.spec, res))
                }
            }

            if (head != null) {
                scene.onDetections(detections, head.yaw, head.pitch, now)
            }

            detectorMs = detectorMs * 0.85f + (now - t0) * 0.15f
            frameTimes.addLast(now)
            while (frameTimes.size > 30) frameTimes.removeFirst()
            val span = now - frameTimes.first()
            fps = if (span > 0) frameTimes.size * 1000f / span else 0f
        } catch (t: Throwable) {
            Log.w(TAG, "Frame analysis failed", t)
        } finally {
            image.close()
        }
    }

    private fun onObjectActivated(ruName: String) {
        // Called from the audio thread — hop to main for TTS / vibrator.
        mainHandler.post {
            if (!sessionRunning) return@post
            val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val now = SystemClock.elapsedRealtime()
            if (prefs.getBoolean(KEY_VOICE, true) && ttsReady && now - lastTtsMs > 1400L) {
                lastTtsMs = now
                tts?.speak(ruName, TextToSpeech.QUEUE_ADD, null, "obj_$now")
            }
            if (prefs.getBoolean(KEY_HAPTIC, true)) {
                vibrateShort()
            }
        }
    }

    private fun vibrateShort() {
        val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= 31) {
            (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
        vibrator.vibrate(VibrationEffect.createOneShot(40, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    // ------------------------------------------------------------- speech

    fun speakScene() {
        if (!sessionRunning) return
        val tracks = sceneTracker?.snapshot(SystemClock.elapsedRealtime()) ?: return
        if (tracks.isEmpty()) {
            speakNow(getString(R.string.no_tracks))
            return
        }
        val yaw = headTracker?.yaw ?: 0f
        val sorted = tracks.sortedBy { abs(Angles.normalize(it.azWorld - yaw)) }.take(5)
        val sb = StringBuilder()
        for (t in sorted) {
            val azRel = Angles.deg(Angles.normalize(t.azWorld - yaw))
            sb.append(t.ruName)
                .append(" — ")
                .append(RuText.direction(azRel))
                .append(", ")
                .append(RuText.distance(t.distance))
                .append(". ")
        }
        speakNow(sb.toString())
    }

    private fun speakNow(text: String) {
        if (!ttsReady) return
        val now = SystemClock.elapsedRealtime()
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "scene_$now")
    }

    // ------------------------------------------------------------ public API

    fun isRunning(): Boolean = sessionRunning

    fun startTestSound() {
        audioEngine?.startTest(10)
    }

    fun uiState(): UiState {
        val prov = provisioner
        return UiState(
            running = sessionRunning,
            headYawDeg = Angles.deg(headTracker?.yaw ?: 0f),
            fps = fps,
            detectorMs = detectorMs,
            activeStreams = audioEngine?.activeStreams ?: 0,
            tracks = sceneTracker?.snapshot(SystemClock.elapsedRealtime()) ?: emptyList(),
            frames = frameCount,
            camError = camError,
            depthMs = depthEstimator?.lastMs?.toInt() ?: 0,
            depthOn = depthEstimator != null,
            neuralStatus = prov?.status ?: "",
            neuralAvailable = NeuralModels.status(this).available,
            materialClfOn = materialClassifier != null,
            sampleCount = sampleLibrary?.sampleCount ?: 0,
            detectorModel = detectorModelName
        )
    }

    // ---------------------------------------------------------------- stop

    fun stopSession() {
        sessionRunning = false
        runCatching { cameraProvider?.unbindAll() }
        cameraExecutor?.shutdown()
        cameraExecutor = null
        cameraProvider = null
        audioEngine?.stopEngine()
        audioEngine = null
        provisioner?.clearPending()
        provisioner?.releaseEngine()
        sampleLibrary?.close()
        sampleLibrary = null
        headTracker?.stop()
        headTracker = null
        detector?.close()
        detector = null
        depthEstimator?.close()
        depthEstimator = null
        materialClassifier?.close()
        materialClassifier = null
        sceneTracker?.clear()
        sceneTracker = null
        frameConverter = null
        textureCache = null
        tts?.stop()
        tts?.shutdown()
        tts = null
        ttsReady = false
        releaseWakeLock()
        abandonAudioFocus()
    }

    override fun onDestroy() {
        stopSession()
        super.onDestroy()
    }

    // --------------------------------------------------------- plumbing

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel),
            NotificationManager.IMPORTANCE_LOW
        )
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(channel)
    }

    private fun startInForeground() {
        val openIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stopIntent = PendingIntent.getService(
            this, 1,
            Intent(this, AudioSceneService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_notify)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setContentIntent(openIntent)
            .setOngoing(true)
            .addAction(0, getString(R.string.action_stop), stopIntent)
            .build()
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun requestAudioFocus() {
        val am = audioManager ?: return
        val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .build()
        audioFocusRequest = req
        am.requestAudioFocus(req)
    }

    private fun abandonAudioFocus() {
        audioFocusRequest?.let { audioManager?.abandonAudioFocusRequest(it) }
        audioFocusRequest = null
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "echolens:session").apply {
            setReferenceCounted(false)
            acquire(6 * 60 * 60 * 1000L)
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }
}
