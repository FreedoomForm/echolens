package com.echolens.app

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.view.WindowManager
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.echolens.app.core.RuText
import com.echolens.app.audio.neural.NeuralModels
import com.echolens.app.service.AudioSceneService
import com.echolens.app.util.Angles
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvTracks: TextView
    private lateinit var btnToggle: MaterialButton
    private lateinit var btnSpeak: MaterialButton
    private lateinit var btnTest: MaterialButton
    private lateinit var swVoice: MaterialSwitch
    private lateinit var swHaptic: MaterialSwitch
    private lateinit var swScreen: MaterialSwitch
    private lateinit var swInvert: MaterialSwitch
    private lateinit var swSpeaker: MaterialSwitch
    private lateinit var swModel: MaterialSwitch
    private lateinit var tvModelStatus: TextView
    private lateinit var swNeural: MaterialSwitch
    private lateinit var tvNeuralStatus: TextView
    private lateinit var tvDiag: TextView
    private lateinit var tvVolumeValue: TextView
    private lateinit var sbVolume: SeekBar
    private lateinit var tvConeValue: TextView
    private lateinit var sbCone: SeekBar
    private lateinit var tvStreamsValue: TextView
    private lateinit var sbStreams: SeekBar

    private var serviceBinder: AudioSceneService.LocalBinder? = null
    private var bound = false
    private val handler = Handler(Looper.getMainLooper())

    private val poller = object : Runnable {
        override fun run() {
            updateUi()
            if (bound) handler.postDelayed(this, 500)
        }
    }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            serviceBinder = service as? AudioSceneService.LocalBinder
            bound = true
            updateUi()
            handler.postDelayed(poller, 500)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            serviceBinder = null
            bound = false
            handler.removeCallbacks(poller)
            updateUi()
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        if (grants[Manifest.permission.CAMERA] == true) {
            maybeStartService()
        } else {
            Toast.makeText(this, R.string.permission_needed, Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tv_status)
        tvTracks = findViewById(R.id.tv_tracks)
        btnToggle = findViewById(R.id.btn_toggle)
        btnSpeak = findViewById(R.id.btn_speak)
        btnTest = findViewById(R.id.btn_test)
        swVoice = findViewById(R.id.sw_voice)
        swHaptic = findViewById(R.id.sw_haptic)
        swScreen = findViewById(R.id.sw_screen)
        swInvert = findViewById(R.id.sw_invert)
        swSpeaker = findViewById(R.id.sw_speaker)
        swModel = findViewById(R.id.sw_model)
        tvModelStatus = findViewById(R.id.tv_model_status)
        swNeural = findViewById(R.id.sw_neural)
        tvNeuralStatus = findViewById(R.id.tv_neural_status)
        tvDiag = findViewById(R.id.tv_diag)
        tvVolumeValue = findViewById(R.id.tv_volume_value)
        sbVolume = findViewById(R.id.sb_volume)
        tvConeValue = findViewById(R.id.tv_cone_value)
        sbCone = findViewById(R.id.sb_cone)
        tvStreamsValue = findViewById(R.id.tv_streams_value)
        sbStreams = findViewById(R.id.sb_streams)

        loadSettings()
        setupListeners()
    }

    override fun onStart() {
        super.onStart()
        bindService(
            Intent(this, AudioSceneService::class.java),
            connection,
            Context.BIND_AUTO_CREATE
        )
    }

    override fun onStop() {
        handler.removeCallbacks(poller)
        if (bound) {
            unbindService(connection)
            bound = false
            serviceBinder = null
        }
        updateUi()
        super.onStop()
    }

    // ------------------------------------------------------------- actions

    private fun maybeStartService() {
        val intent = Intent(this, AudioSceneService::class.java)
            .setAction(AudioSceneService.ACTION_START)
        ContextCompat.startForegroundService(this, intent)
    }

    private fun stopServiceByUser() {
        val intent = Intent(this, AudioSceneService::class.java)
            .setAction(AudioSceneService.ACTION_STOP)
        startService(intent)
    }

    private fun setupListeners() {
        btnToggle.setOnClickListener {
            val service = serviceBinder?.service
            if (service != null && service.isRunning()) {
                stopServiceByUser()
                handler.postDelayed({ updateUi() }, 250)
            } else {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
                ) {
                    maybeStartService()
                } else {
                    permissionLauncher.launch(
                        arrayOf(
                            Manifest.permission.CAMERA,
                            Manifest.permission.POST_NOTIFICATIONS
                        )
                    )
                }
            }
        }

        btnSpeak.setOnClickListener { serviceBinder?.service?.speakScene() }

        btnTest.setOnClickListener {
            val service = serviceBinder?.service
            if (service != null && service.isRunning()) {
                service.startTestSound()
                Toast.makeText(this, R.string.test_hint, Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, R.string.start_first, Toast.LENGTH_SHORT).show()
            }
        }

        val prefs = getSharedPreferences(AudioSceneService.PREFS, Context.MODE_PRIVATE)

        fun persistAndApply() {
            serviceBinder?.service?.applySettings()
        }

        swVoice.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(AudioSceneService.KEY_VOICE, checked).apply()
        }
        swHaptic.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(AudioSceneService.KEY_HAPTIC, checked).apply()
        }
        swScreen.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(AudioSceneService.KEY_SCREEN, checked).apply()
            applyKeepScreenOn(checked)
        }
        swInvert.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(AudioSceneService.KEY_INVERT, checked).apply()
            persistAndApply()
        }
        swSpeaker.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(AudioSceneService.KEY_SPEAKER, checked).apply()
            persistAndApply()
        }
        swModel.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(AudioSceneService.KEY_MODEL_ACCURATE, checked).apply()
        }
        swNeural.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(AudioSceneService.KEY_NEURAL, checked).apply()
            persistAndApply()
            refreshNeuralStatus()
        }

        sbVolume.max = 100
        sbVolume.setOnSeekBarChangeListener(object : SimpleSeekBar() {
            override fun onProgressChanged(bar: SeekBar?, progress: Int, fromUser: Boolean) {
                tvVolumeValue.text = getString(R.string.volume_label, progress)
                if (fromUser) {
                    prefs.edit().putInt(AudioSceneService.KEY_VOLUME, progress).apply()
                    persistAndApply()
                }
            }
        })

        sbCone.max = 40 // 20°..60°
        sbCone.setOnSeekBarChangeListener(object : SimpleSeekBar() {
            override fun onProgressChanged(bar: SeekBar?, progress: Int, fromUser: Boolean) {
                val deg = progress + 20
                tvConeValue.text = getString(R.string.cone_label, deg)
                if (fromUser) {
                    prefs.edit().putInt(AudioSceneService.KEY_CONE_DEG, deg).apply()
                    persistAndApply()
                }
            }
        })

        sbStreams.max = 14 // 2..16 voices
        sbStreams.setOnSeekBarChangeListener(object : SimpleSeekBar() {
            override fun onProgressChanged(bar: SeekBar?, progress: Int, fromUser: Boolean) {
                val n = progress + 2
                tvStreamsValue.text = getString(R.string.streams_label, n)
                if (fromUser) {
                    prefs.edit().putInt(AudioSceneService.KEY_STREAMS, n).apply()
                    persistAndApply()
                }
            }
        })
    }

    private open class SimpleSeekBar : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(bar: SeekBar?, progress: Int, fromUser: Boolean) {}
        override fun onStartTrackingTouch(bar: SeekBar?) {}
        override fun onStopTrackingTouch(bar: SeekBar?) {}
    }

    private fun loadSettings() {
        val prefs = getSharedPreferences(AudioSceneService.PREFS, Context.MODE_PRIVATE)
        swVoice.isChecked = prefs.getBoolean(AudioSceneService.KEY_VOICE, true)
        swHaptic.isChecked = prefs.getBoolean(AudioSceneService.KEY_HAPTIC, true)
        swScreen.isChecked = prefs.getBoolean(AudioSceneService.KEY_SCREEN, true)
        swInvert.isChecked = prefs.getBoolean(AudioSceneService.KEY_INVERT, false)
        swSpeaker.isChecked = prefs.getBoolean(AudioSceneService.KEY_SPEAKER, true)
        swModel.isChecked = prefs.getBoolean(AudioSceneService.KEY_MODEL_ACCURATE, true)
        tvModelStatus.text = getString(R.string.model_hint)
        swNeural.isChecked = prefs.getBoolean(AudioSceneService.KEY_NEURAL, false)
        refreshNeuralStatus()
        val volume = prefs.getInt(AudioSceneService.KEY_VOLUME, AudioSceneService.DEFAULT_VOLUME)
        sbVolume.progress = volume
        tvVolumeValue.text = getString(R.string.volume_label, volume)
        val cone = prefs.getInt(AudioSceneService.KEY_CONE_DEG, AudioSceneService.DEFAULT_CONE_DEG)
        sbCone.progress = (cone - 20).coerceIn(0, 40)
        tvConeValue.text = getString(R.string.cone_label, cone)
        val streams = prefs.getInt(AudioSceneService.KEY_STREAMS, AudioSceneService.DEFAULT_STREAMS)
        sbStreams.progress = (streams - 2).coerceIn(0, 14)
        tvStreamsValue.text = getString(R.string.streams_label, streams)
    }

    // ----------------------------------------------------------------- UI

    private fun refreshNeuralStatus() {
        val st = NeuralModels.status(this)
        tvNeuralStatus.text = when {
            !st.deviceCapable -> getString(
                R.string.neural_weak_device,
                (st.totalRamBytes / (1024L * 1024 * 1024)).toInt()
            )
            st.complete -> getString(R.string.neural_ready)
            st.bundledInApk -> getString(R.string.neural_bundled)
            else -> getString(R.string.neural_no_models, st.modelsDir?.absolutePath ?: "")
        }
    }

    private fun applyKeepScreenOn(enabled: Boolean) {
        if (enabled) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    private fun updateUi() {
        val service = serviceBinder?.service
        val running = service != null && service.isRunning()

        btnToggle.setText(if (running) R.string.btn_stop else R.string.btn_start)
        btnSpeak.isEnabled = running
        btnTest.isEnabled = running

        if (running && service != null) {
            val state = service.uiState()
            val headDeg = Math.round(state.headYawDeg).toInt()
            tvStatus.text = getString(R.string.status_fmt, headDeg, state.tracks.size, state.fps)

            val camState = state.camError ?: "OK"
            val depthStr = if (!state.depthOn) "выкл" else "${state.depthMs} мс"
            tvDiag.text = getString(
                R.string.diag_fmt, camState, state.frames, Math.round(state.detectorMs), depthStr
            ) + "\n" + getString(
                R.string.diag_lib_fmt, state.detectorModel, state.sampleCount, state.materialClfOn
            )

            if (state.neuralStatus.isNotEmpty()) {
                tvNeuralStatus.text = getString(R.string.neural_gen, state.neuralStatus)
            } else {
                refreshNeuralStatus()
            }

            if (state.tracks.isEmpty()) {
                tvTracks.text = getString(R.string.no_tracks)
            } else {
                val sb = StringBuilder()
                state.tracks
                    .sortedBy { it.distance }
                    .take(6)
                    .forEach { t ->
                        val azRel = Angles.deg(Angles.normalize(t.azWorld - Math.toRadians(state.headYawDeg.toDouble()).toFloat()))
                        sb.append(t.ruName)
                            .append(" · ")
                            .append(RuText.direction(azRel))
                            .append(" · ")
                            .append(RuText.distance(t.distance))
                            .append('\n')
                    }
                sb.append('\n').append(getString(R.string.streams_fmt, state.activeStreams))
                tvTracks.text = sb.toString()
            }

            applyKeepScreenOn(
                getSharedPreferences(AudioSceneService.PREFS, Context.MODE_PRIVATE)
                    .getBoolean(AudioSceneService.KEY_SCREEN, true)
            )
        } else {
            tvStatus.text = getString(R.string.status_idle)
            tvTracks.text = getString(R.string.no_tracks)
            tvDiag.text = ""
            applyKeepScreenOn(false)
        }
    }
}
