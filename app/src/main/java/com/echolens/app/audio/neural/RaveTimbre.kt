package com.echolens.app.audio.neural

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.FloatBuffer
import java.util.Random
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Optional texture backend: a per-material RAVE model (IRCAM, MIT) exported
 * to ONNX (nn~ compatible), stored as  rave_WOOD.onnx  etc. in the models dir.
 *
 * Two input conventions are auto-detected:
 *  - audio input  [1, N]      : timbre-transfer mode — the model re-voices a
 *    neutral noise/bed seed into its training material;
 *  - latent input [1, C, F]   : decoder-only mode — gaussian latents are
 *    sampled directly.
 *
 * Output is converted to a mono 48 kHz loop (crossfaded seam), peak-normalized.
 */
class RaveTimbre(
    private val env: OrtEnvironment,
    modelFile: File,
    private val numThreads: Int = 3
) : AutoCloseable {

    private val session: OrtSession
    private val inputName: String
    private val audioMode: Boolean // true = [1, N] audio input
    private val modelSr: Int
    private var pixelsSeed: FloatArray = FloatArray(0)

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(numThreads)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(modelFile.absolutePath, opts)
        inputName = session.inputNames.iterator().next()
        val info = session.getInputInfo()[inputName]!!.info as TensorInfo
        val shape = info.shape
        audioMode = shape.size == 2
        // Optional sidecar: rave_MATERIAL.json with {"sample_rate": 44100}
        val meta = File(modelFile.parentFile, modelFile.nameWithoutExtension + ".json")
        modelSr = if (meta.isFile) {
            Regex("\"sample_rate\"\\s*:\\s*(\\d+)").find(meta.readText())
                ?.groupValues?.get(1)?.toIntOrNull() ?: 44100
        } else 44100
    }

    /** Returns mono 48 kHz loop samples or null on failure. */
    fun generate(seed: Long, seconds: Float = 8f): FloatArray? = try {
        val rng = Random(seed)
        val frames = (seconds * modelSr).toInt()

        val input: OnnxTensor = if (audioMode) {
            if (pixelsSeed.size != frames) {
                pixelsSeed = FloatArray(frames)
            }
            val a = pixelsSeed
            for (i in a.indices) {
                // Slightly pink-ish noise seed: RAVE re-voices it into material.
                a[i] = (rng.nextGaussian() * 0.35).toFloat()
            }
            OnnxTensor.createTensor(env, FloatBuffer.wrap(a), longArrayOf(1, frames.toLong()))
        } else {
            val shape = (session.getInputInfo()[inputName]!!.info as TensorInfo).shape
            var elems = 1L
            for (d in shape) elems *= if (d > 0) d else 1L
            val lat = FloatArray(elems.toInt())
            for (i in lat.indices) lat[i] = rng.nextGaussian().toFloat()
            OnnxTensor.createTensor(env, FloatBuffer.wrap(lat), shape)
        }

        input.use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                val out = res[0].value as? Array<*> ?: return null
                @Suppress("UNCHECKED_CAST")
                val first = out[0] as? FloatArray ?: return null
                postProcess(first, modelSr)
            }
        }
    } catch (t: Throwable) {
        android.util.Log.w("RaveTimbre", "RAVE generation failed", t)
        null
    }

    private fun postProcess(raw: FloatArray, sr: Int): FloatArray {
        var peak = 1e-6f
        for (v in raw) {
            val a = abs(v)
            if (a > peak) peak = a
        }
        val norm = 0.85f / peak

        val outLen = (raw.size.toLong() * 48000L / sr).toInt()
        val out = FloatArray(outLen)
        val ratio = sr.toFloat() / 48000f
        for (i in out.indices) {
            val pos = i * ratio
            val i0 = pos.toInt().coerceIn(0, raw.size - 1)
            val i1 = (i0 + 1).coerceAtMost(raw.size - 1)
            val f = pos - i0
            out[i] = (raw[i0] + (raw[i1] - raw[i0]) * f) * norm
        }

        val fade = (0.25f * 48000f).roundToInt()
        if (fade in 1 until out.size / 2) {
            for (i in 0 until fade) {
                val u = i.toFloat() / fade
                val tail = out[out.size - fade + i]
                out[i] = out[i] * u + tail * (1f - u)
            }
            val keep = out.size - fade
            val trimmed = FloatArray(keep)
            System.arraycopy(out, 0, trimmed, 0, keep)
            return trimmed
        }
        return out
    }

    override fun close() {
        runCatching { session.close() }
    }
}
