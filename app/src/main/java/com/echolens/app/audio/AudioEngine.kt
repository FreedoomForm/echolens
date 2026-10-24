package com.echolens.app.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import com.echolens.app.core.HeadPose
import com.echolens.app.core.MaterialSpec
import com.echolens.app.core.TrackSnapshot
import com.echolens.app.util.Angles
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sign

/**
 * Real-time binaural audio engine.
 *
 * Pipeline per 20 ms block:
 *  1. Pull the current head pose and the world-anchored track list.
 *  2. Rank tracks (importance × proximity × centrality), keep the top N,
 *     gate them with the attention cone (±cone width around the head
 *     direction — objects the user "looks at" sound; the rest is quiet).
 *  3. Render each active object's material texture.
 *  4. Spatialize: constant-power ILD panning + Woodworth ITD (inter-ear
 *     delay) + contralateral head-shadow low-pass + distance gain —
 *     all relative to the CURRENT head yaw, so sounds stay anchored in
 *     the world when the head turns (externalization).
 *  5. Add a light stereo reverb and a soft limiter, write to a low-latency
 *     AudioTrack.
 */
class AudioEngine(
    private val sampleRate: Int = 48000,
    private val blockFrames: Int = 960, // 20 ms @ 48 kHz
    private val headProvider: () -> HeadPose,
    private val tracksProvider: () -> List<TrackSnapshot>,
    private val onObjectActivated: ((ruName: String) -> Unit)? = null
) : Thread("EchoLensAudio") {

    class Settings {
        @Volatile var masterVolume = 0.85f
        @Volatile var coneWidthRad = 0.524f // 30°
        @Volatile var coneFloor = 0.12f
        @Volatile var maxSources = 4
        @Volatile var invertAzimuth = false
    }

    val settings = Settings()

    @Volatile private var running = false
    @Volatile private var testModeUntilMs = 0L
    @Volatile private var streamsCount = 0

    val activeStreams: Int get() = streamsCount

    private inner class Slot {
        var trackId = -1L
        var ruName = ""
        var spec: MaterialSpec? = null
        var generator: TextureGenerator? = null
        var azWorld = 0f
        var elevWorld = 0f
        var distance = 2f
        var targetGain = 0f
        var currentGain = 0f
        var silencingBlocks = 0
        var testSource = false

        // Spatial state (updated per block)
        var panL = 0.707f
        var panR = 0.707f
        var delayL = 0f
        var delayR = 0f
        var lpCoefL = 1f
        var lpCoefR = 1f
        var lpL = 0f
        var lpR = 0f
        var gainL = 0f
        var gainR = 0f
        val dry = FloatArray(blockFrames)
        val delayLineL = FloatArray(64)
        val delayLineR = FloatArray(64)
        var writePos = 0
    }

    private val slots = Array(8) { Slot() }
    private val mixL = FloatArray(blockFrames)
    private val mixR = FloatArray(blockFrames)
    private val out = FloatArray(blockFrames * 2)
    private val announcedAt = HashMap<Long, Long>()

    // Test tone state (sound check: a ping circles the head)
    private var testPhase = 0f
    private var testPingTimer = 0f
    private var testPingEnv = 0f
    private var testPingPhase = 0f

    fun startEngine() {
        if (!isAlive) {
            running = true
            start()
        }
    }

    fun stopEngine() {
        running = false
    }

    /** Sound check: a ping circles the head for [seconds]. */
    fun startTest(seconds: Int = 10) {
        testModeUntilMs = SystemClock.elapsedRealtime() + seconds * 1000L
        testPingTimer = 0f
        testPhase = 0f
    }

    override fun run() {
        priority = MAX_PRIORITY
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_FLOAT
        )
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(minBuf, (sampleRate / 10) * 8)) // ≥100 ms, stereo float
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .build()
        try {
            track.play()
            val reverb = SimpleReverb(sampleRate)
            while (running) {
                renderBlock(track, reverb)
            }
        } finally {
            try {
                track.stop()
            } catch (_: IllegalStateException) {
            }
            track.release()
        }
    }

    private fun renderBlock(track: AudioTrack, reverb: SimpleReverb) {
        java.util.Arrays.fill(mixL, 0f)
        java.util.Arrays.fill(mixR, 0f)
        val pose = headProvider()
        val nowMs = SystemClock.elapsedRealtime()

        if (nowMs < testModeUntilMs) {
            renderTestSource(pose)
        } else {
            assignSlots(tracksProvider(), pose, nowMs)
        }

        for (s in slots) {
            if (s.trackId != -1L) spatialize(s, pose)
        }

        reverb.process(mixL, mixR, blockFrames)

        val mv = settings.masterVolume
        var i = 0
        for (f in 0 until blockFrames) {
            val l = softClip(mixL[f] * mv)
            val r = softClip(mixR[f] * mv)
            out[i++] = l
            out[i++] = r
        }
        track.write(out, 0, out.size, AudioTrack.WRITE_BLOCKING)
        streamsCount = slots.count { it.trackId != -1L && it.currentGain > 0.05f }
    }

    // ---------------------------------------------------------------- slots

    private class Candidate(
        val t: TrackSnapshot,
        val azRel: Float,
        val score: Float,
        val gain: Float
    )

    private fun relAz(azWorld: Float, pose: HeadPose): Float {
        var az = Angles.normalize(azWorld - pose.yaw)
        if (settings.invertAzimuth) az = -az
        return az
    }

    private fun coneGain(azRel: Float): Float {
        val w = settings.coneWidthRad
        val floor = settings.coneFloor
        val inner = w * 0.55f
        val outer = w * 1.45f
        val t = 1f - Angles.smoothstep(inner, outer, abs(azRel))
        return floor + (1f - floor) * t
    }

    private fun assignSlots(tracks: List<TrackSnapshot>, pose: HeadPose, nowMs: Long) {
        // Release any test-mode slots first.
        for (s in slots) {
            if (s.testSource) {
                s.testSource = false
                s.trackId = -1L
                s.generator = null
                s.currentGain = 0f
                s.targetGain = 0f
                s.silencingBlocks = 0
            }
        }

        val max = settings.maxSources.coerceIn(1, 8)
        val cands = ArrayList<Candidate>(tracks.size)
        for (t in tracks) {
            val az = relAz(t.azWorld, pose)
            val centrality = 0.5f + 0.5f * cos(az)
            val proximity = t.spec.refDistanceM / (t.spec.refDistanceM + t.distance)
            val score = t.spec.importance * (0.35f + 0.65f * proximity) * (0.3f + 0.7f * centrality)
            cands.add(Candidate(t, az, score, coneGain(az)))
        }
        cands.sortByDescending { it.score }
        if (cands.size > max) cands.subList(max, cands.size).clear()

        // Update existing slots / mark the rest for fade-out.
        for (s in slots) {
            if (s.trackId == -1L) continue
            val c = cands.firstOrNull { it.t.id == s.trackId }
            if (c == null) {
                s.targetGain = 0f
                s.silencingBlocks++
            } else {
                s.azWorld = c.t.azWorld
                s.elevWorld = c.t.elevWorld
                s.distance = c.t.distance
                s.targetGain = c.gain
                s.silencingBlocks = 0
            }
        }

        // Assign new tracks to free slots.
        for (c in cands) {
            if (slots.any { it.trackId == c.t.id }) continue
            val s = slots.firstOrNull { it.trackId == -1L }
                ?: slots.firstOrNull { it.targetGain == 0f && it.currentGain < 0.02f }
                ?: break
            s.trackId = c.t.id
            s.ruName = c.t.ruName
            s.spec = c.t.spec
            s.generator = TextureGenerator(c.t.spec, sampleRate, c.t.id * 6364136223846793005L)
            s.azWorld = c.t.azWorld
            s.elevWorld = c.t.elevWorld
            s.distance = c.t.distance
            s.targetGain = c.gain
            s.currentGain = 0f
            s.silencingBlocks = 0
            maybeAnnounce(c.t.id, c.t.ruName, nowMs, c.gain)
        }

        // Retire fully faded-out slots.
        for (s in slots) {
            s.currentGain += (s.targetGain - s.currentGain) * 0.25f
            if (s.trackId != -1L && s.targetGain == 0f && s.silencingBlocks > 8 && s.currentGain < 0.01f) {
                s.trackId = -1L
                s.generator = null
            }
        }
    }

    private fun maybeAnnounce(id: Long, ruName: String, nowMs: Long, gain: Float) {
        if (gain < 0.25f) return
        val last = announcedAt[id] ?: 0L
        if (last != 0L && nowMs - last < 8000L) return
        announcedAt[id] = nowMs
        if (announcedAt.size > 64) announcedAt.clear()
        onObjectActivated?.invoke(ruName)
    }

    // ------------------------------------------------------------ rendering

    private fun spatialize(s: Slot, pose: HeadPose) {
        val spec = s.spec ?: return
        val az = relAz(s.azWorld, pose)
        val elev = Angles.clamp(s.elevWorld - pose.pitch, -1.25f, 1.25f)

        if (s.testSource) {
            // dry buffer already filled by renderTestSource
        } else {
            val gen = s.generator ?: return
            val distGain = (spec.refDistanceM / s.distance).pow(1.15f).coerceAtMost(1.7f)
            gen.process(s.dry, blockFrames, s.distance, s.targetGain * distGain)
        }

        // Constant-power panning.
        val pan = Angles.clamp(az / HALF_PI, -1f, 1f) * QUARTER_PI + QUARTER_PI
        val pl = cos(pan)
        val pr = sin(pan)

        // Woodworth ITD: only the contralateral (farther) ear is delayed.
        val theta = Angles.clamp(az, -HALF_PI, HALF_PI)
        val w = (theta + sin(theta)) * INV_WOODWORTH // -1..1
        val maxItdSamples = 0.00062f * sampleRate // ~0.62 ms
        s.delayL = (w * maxItdSamples).coerceIn(0f, 60f)
        s.delayR = (-w * maxItdSamples).coerceIn(0f, 60f)

        // Contralateral head-shadow low-pass + elevation brightness tilt.
        val shadowL = Angles.smoothstep(0f, 1.05f, if (az > 0f) az else 0f)
        val shadowR = Angles.smoothstep(0f, 1.05f, if (az < 0f) -az else 0f)
        val bright = 1f + 0.3f * sin(elev)
        s.lpCoefL = lpCoef((12000f / (1f + shadowL * 4.5f)) * bright)
        s.lpCoefR = lpCoef((12000f / (1f + shadowR * 4.5f)) * bright)

        var wp = s.writePos
        var gainL = s.gainL
        var gainR = s.gainR
        var lpL = s.lpL
        var lpR = s.lpR
        val tl = pl
        val tr = pr
        val coefL = s.lpCoefL
        val coefR = s.lpCoefR
        val dl = s.delayL
        val dr = s.delayR
        val lineL = s.delayLineL
        val lineR = s.delayLineR
        val dry = s.dry
        for (i in 0 until blockFrames) {
            val x = dry[i]
            lineL[wp] = x
            lineR[wp] = x
            val sl = readDelay(lineL, wp, dl)
            val sr = readDelay(lineR, wp, dr)
            lpL += coefL * (sl - lpL)
            lpR += coefR * (sr - lpR)
            gainL += 0.004f * (tl - gainL)
            gainR += 0.004f * (tr - gainR)
            mixL[i] += lpL * gainL
            mixR[i] += lpR * gainR
            wp = (wp + 1) and 63
        }
        s.writePos = wp
        s.gainL = gainL
        s.gainR = gainR
        s.lpL = lpL
        s.lpR = lpR
    }

    private fun readDelay(line: FloatArray, wp: Int, delay: Float): Float {
        if (delay < 1f) return line[wp]
        val base = wp - delay
        val idx0 = floor(base).toInt() and 63
        val idx1 = (idx0 + 1) and 63
        val frac = base - floor(base)
        return line[idx0] + (line[idx1] - line[idx0]) * frac
    }

    private fun lpCoef(fc: Float): Float =
        (1f - exp(-TWO_PI * fc / sampleRate)).coerceIn(0.0005f, 1f)

    private fun softClip(x: Float): Float {
        val a = abs(x)
        return if (a < 0.9f) x else sign(x) * (0.9f + 0.1f * (1f - exp(-(a - 0.9f) / 0.1f)))
    }

    // ----------------------------------------------------------- test mode

    private fun renderTestSource(pose: HeadPose) {
        // Fade normal sources out quickly.
        for (s in slots) {
            if (!s.testSource && s.trackId != -1L) {
                s.targetGain = 0f
                s.silencingBlocks++
                s.currentGain *= 0.7f
            }
        }
        for (s in slots) {
            if (!s.testSource && s.trackId != -1L && s.currentGain < 0.02f) {
                s.trackId = -1L
                s.generator = null
            }
        }

        val s = slots[0]
        if (!s.testSource) {
            s.testSource = true
            s.trackId = Long.MIN_VALUE
            s.spec = com.echolens.app.core.TextureLibrary.spec("clock")
            s.currentGain = 0f
            s.gainL = 0f
            s.gainR = 0f
            s.silencingBlocks = 0
        }
        // The ping orbits the head: azimuth sweeps 0..360° over ~9 s.
        s.azWorld = pose.yaw + testPhase
        s.elevWorld = pose.pitch
        s.distance = 2f
        s.targetGain = 1f
        s.currentGain += (1f - s.currentGain) * 0.25f
        testPhase += 0.7f * blockFrames / sampleRate
        if (testPhase > TWO_PI) testPhase -= TWO_PI

        // Synthesize the ping directly into the dry buffer.
        val dry = s.dry
        testPingTimer -= blockFrames
        var env = testPingEnv
        var phase = testPingPhase
        for (i in 0 until blockFrames) {
            if (testPingTimer <= 0f) {
                testPingTimer = 0.55f * sampleRate
                env = 1f
                phase = 0f
            }
            dry[i] = if (env > 0.001f) sin(TWO_PI * 880f * phase) * env * 0.5f else 0f
            env *= 0.99958f
            phase += 0.018333f
            if (phase >= 1f) phase -= 1f
        }
        testPingEnv = env
        testPingPhase = phase
    }

    private companion object {
        const val TWO_PI = 6.2831855f
        const val HALF_PI = 1.5707964f
        const val QUARTER_PI = 0.7853982f
        const val INV_WOODWORTH = 1f / (1.5707964f + 1f)
    }
}
