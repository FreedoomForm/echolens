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
import com.echolens.app.core.HeadPose
import com.echolens.app.core.RuText
import com.echolens.app.core.TrackSnapshot
import com.echolens.app.head.HeadTracker
import com.echolens.app.util.Angles
import com.echolens.app.vision.DetectorEngine
import com.echolens.app.vision.FrameConverter
import com.echolens.app.vision.SceneTracker
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
        const val KEY_VOLUME = "master_volume"
        const val KEY_CONE_DEG = "cone_deg"
        const val KEY_STREAMS = "max_streams"

        private const val CHANNEL_ID = "session"
        private const val NOTIFICATION_ID = 42
        private const val TAG = "EchoLensService"

        const val DEFAULT_VOLUME = 85
        const val DEFAULT_CONE_DEG = 30
        const val DEFAULT_STREAMS = 4
    }

    class UiState(
        val running: Boolean,
        val headYawDeg: Float,
        val fps: Float,
        val detectorMs: Float,
        val activeStreams: Int,
        val tracks: List<TrackSnapshot>
    )

    inner class LocalBinder : Binder() {
        val service: AudioSceneService get() = this@AudioSceneService
    }

    private val binder = LocalBinder()

    private var headTracker: HeadTracker? = null
    private var detector: DetectorEngine? = null
    private var sceneTracker: SceneTracker? = null
    private var audioEngine: AudioEngine? = null
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
            else -> if (!sessionRunning) startSession()
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
        detector = DetectorEngine(this)
        sceneTracker = SceneTracker()
        frameConverter = FrameConverter()

        val engine = AudioEngine(
            headProvider = { HeadPose(headTracker?.yaw ?: 0f, headTracker?.pitch ?: 0f) },
            tracksProvider = {
                sceneTracker?.snapshot(SystemClock.elapsedRealtime()) ?: emptyList()
            },
            onObjectActivated = { ruName -> onObjectActivated(ruName) }
        )
        audioEngine = engine
        applySettings()
        engine.startEngine()

        acquireWakeLock()
        startCamera()
    }

    fun applySettings() {
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        audioEngine?.settings?.apply {
            masterVolume = (prefs.getInt(KEY_VOLUME, DEFAULT_VOLUME) / 100f)
                .coerceIn(0.05f, 1f)
            coneWidthRad = Angles.rad(prefs.getInt(KEY_CONE_DEG, DEFAULT_CONE_DEG).toFloat())
            maxSources = prefs.getInt(KEY_STREAMS, DEFAULT_STREAMS).coerceIn(2, 6)
            invertAzimuth = prefs.getBoolean(KEY_INVERT, false)
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                androidx.camera.core.resolutionselector.ResolutionStrategy(
                                    android.util.Size(640, 480),
                                    androidx.camera.core.resolutionselector.ResolutionStrategy
                                        .FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                                )
                            )
                            .build()
                    )
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                val executor = Executors.newSingleThreadExecutor()
                cameraExecutor = executor
                analysis.setAnalyzer(executor, ::analyzeFrame)
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "Camera init failed", e)
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
            val detections = detector.detect(bitmap)
            val now = SystemClock.elapsedRealtime()

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
        return UiState(
            running = sessionRunning,
            headYawDeg = Angles.deg(headTracker?.yaw ?: 0f),
            fps = fps,
            detectorMs = detectorMs,
            activeStreams = audioEngine?.activeStreams ?: 0,
            tracks = sceneTracker?.snapshot(SystemClock.elapsedRealtime()) ?: emptyList()
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
        headTracker?.stop()
        headTracker = null
        detector?.close()
        detector = null
        sceneTracker?.clear()
        sceneTracker = null
        frameConverter = null
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
