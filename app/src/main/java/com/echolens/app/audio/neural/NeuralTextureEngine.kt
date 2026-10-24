package com.echolens.app.audio.neural

import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.roundToInt

/**
 * On-device neural texture generator: Stable Audio Open Small (341M) on LiteRT.
 *
 * Kotlin port of the Apache-2.0 reference pipeline from Arm ML-examples
 * (kleidiai-examples/audiogen, app/audiogen.cpp):
 *
 *   1. conditioners_float32.tflite — T5 text embeds + number conditioner
 *      inputs:  [0] ids int64, [1] attention mask int64, [2] seconds float32
 *      outputs: [0] cross-attention embeds, [2] global conditioning
 *   2. dit_model.tflite — diffusion transformer (dynamic int8), 8 steps of a
 *      ping-pong sampler with the sigmoid-sigma schedule
 *   3. autoencoder_model.tflite — VAE decoder (fp16) -> planar stereo 44.1 kHz
 *
 * All model files are provisioned outside the APK (see NeuralModels);
 * if anything is missing or fails, the caller silently falls back to
 * procedural synthesis.
 */
class NeuralTextureEngine private constructor(
    private val conditioners: Interpreter,
    private val dit: Interpreter,
    private val vae: Interpreter
) : AutoCloseable {

    companion object {
        private const val TAG = "NeuralTex"
        private const val LOGSNR_MAX = -6.0f
        private const val SIGMA_MAX = 1.0f
        const val DEFAULT_STEPS = 8
        private const val OUT_PEAK = 0.85f
        private const val RNG_OFFSET = 4564L
        private const val MODEL_SR = 44100
        private const val ENGINE_SR = 48000

        fun tryCreate(status: NeuralModels.Status, numThreads: Int = 4): NeuralTextureEngine? {
            val condF = status.conditioners ?: return null
            val ditF = status.dit ?: return null
            val vaeF = status.vae ?: return null
            var c: Interpreter? = null
            var d: Interpreter? = null
            var v: Interpreter? = null
            return try {
                c = Interpreter(condF, Interpreter.Options().setNumThreads(numThreads))
                d = Interpreter(ditF, Interpreter.Options().setNumThreads(numThreads))
                v = Interpreter(vaeF, Interpreter.Options().setNumThreads(numThreads))
                val engine = NeuralTextureEngine(c, d, v)
                engine.probe()
                engine
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "Engine init failed", t)
                runCatching { c?.close() }
                runCatching { d?.close() }
                runCatching { v?.close() }
                null
            }
        }
    }

    // Probed tensor geometry
    private var idsElems = 0
    private var crossElems = 0
    private var globalElems = 0
    private var latentElems = 0
    private var aeElems = 0

    // Reusable direct buffers
    private lateinit var idsBuf: ByteBuffer
    private lateinit var maskBuf: ByteBuffer
    private lateinit var secBuf: ByteBuffer
    private lateinit var crossBuf: ByteBuffer
    private lateinit var globalBuf: ByteBuffer
    private lateinit var ditOutBuf: ByteBuffer
    private lateinit var aeOutBuf: ByteBuffer
    private lateinit var tBuf: ByteBuffer

    @Synchronized
    private fun probe() {
        idsElems = numElems(conditioners.getInputTensor(0).shape(), fallback = 128)
        crossElems = numElems(conditioners.getOutputTensor(0).shape(), fallback = 98304)
        globalElems = numElems(conditioners.getOutputTensor(2).shape(), fallback = 768)
        latentElems = numElems(dit.getInputTensor(0).shape(), fallback = 0)
        aeElems = numElems(vae.getOutputTensor(0).shape(), fallback = 0)
        require(latentElems > 0 && aeElems > 0 && idsElems > 0) { "bad tensor geometry" }

        idsBuf = direct(idsElems * 8)
        maskBuf = direct(idsElems * 8)
        secBuf = direct(16)
        crossBuf = direct(crossElems * 4)
        globalBuf = direct(globalElems * 4)
        ditOutBuf = direct(latentElems * 4)
        aeOutBuf = direct(aeElems * 4)
        tBuf = direct(16)

        android.util.Log.i(
            TAG, "probed: ids=$idsElems cross=$crossElems global=$globalElems " +
                "latent=$latentElems ae=$aeElems"
        )
    }

    /**
     * Generates one ~10 s texture. Returns mono samples at 48 kHz,
     * peak-normalized and loop-crossfaded, or null on any failure.
     * Must be called from a background thread.
     */
    fun generate(
        promptIds: IntArray,
        seed: Long,
        steps: Int = DEFAULT_STEPS,
        secondsTotal: Float = 10f,
        onProgress: ((Int, Int) -> Unit)? = null
    ): FloatArray? = try {
        runPipeline(promptIds, seed, steps, secondsTotal, onProgress)
    } catch (t: Throwable) {
        android.util.Log.w(TAG, "Generation failed", t)
        null
    }

    // --------------------------------------------------------------- pipeline

    private fun runPipeline(
        promptIds: IntArray,
        seed: Long,
        steps: Int,
        secondsTotal: Float,
        onProgress: ((Int, Int) -> Unit)?
    ): FloatArray {
        // ---- 1. Conditioners ------------------------------------------------
        val idsLong = idsBuf.order(ByteOrder.nativeOrder()).asLongBuffer()
        val maskLong = maskBuf.order(ByteOrder.nativeOrder()).asLongBuffer()
        idsLong.rewind(); maskLong.rewind()
        val n = promptIds.size.coerceAtMost(idsElems)
        for (i in 0 until n) {
            idsLong.put(promptIds[i].toLong())
            maskLong.put(1L)
        }
        if (n > 0) {
            // EOS token (id 1) last, rest zero-padded.
            idsLong.put(n - 1, 1L)
        }
        idsLong.rewind(); maskLong.rewind()

        secBuf.rewind()
        secBuf.putFloat(secondsTotal)
        secBuf.rewind()

        val condIn = arrayOf<Any>(idsBuf, maskBuf, secBuf)
        val condOut = mapOf(0 to crossBuf, 2 to globalBuf)
        conditioners.runForMultipleInputsOutputs(condIn, condOut)

        // ---- 2. DiT ping-pong sampling --------------------------------------
        val x = FloatArray(latentElems)
        val rng = java.util.Random(seed)
        for (i in 0 until latentElems) x[i] = rng.nextGaussian().toFloat() * SIGMA_MAX

        val sigmas = FloatArray(steps + 1)
        fillSigmas(sigmas, LOGSNR_MAX, 2.0f)

        val ditIn = arrayOf<Any>(x, tBuf, crossBuf, globalBuf)
        val ditOut = mapOf(0 to ditOutBuf)

        for (i in 0 until steps) {
            val cur = sigmas[i]
            val next = sigmas[i + 1]
            tBuf.rewind()
            tBuf.putFloat(cur)
            tBuf.rewind()
            dit.runForMultipleInputsOutputs(ditIn, ditOut)

            ditOutBuf.rewind()
            val outF = ditOutBuf.order(ByteOrder.nativeOrder()).asFloatBuffer()
            val stepRng = java.util.Random(seed + i + RNG_OFFSET)
            for (j in 0 until latentElems) {
                val denoised = x[j] - cur * outF.get(j)
                x[j] = (1f - next) * denoised + next * stepRng.nextGaussian().toFloat()
            }
            onProgress?.invoke(i + 1, steps)
        }

        // ---- 3. VAE decode --------------------------------------------------
        val aeIn = arrayOf<Any>(x)
        val aeOut = mapOf(0 to aeOutBuf)
        vae.runForMultipleInputsOutputs(aeIn, aeOut)

        aeOutBuf.rewind()
        val audioF = aeOutBuf.order(ByteOrder.nativeOrder()).asFloatBuffer()
        val frames = (aeElems / 2).coerceAtMost(audioF.remaining() / 2)
        val mono = FloatArray(frames)
        // Planar layout: [left samples..., right samples...]
        for (i in 0 until frames) {
            mono[i] = 0.5f * (audioF.get(i) + audioF.get(frames + i))
        }
        return postProcess(mono, MODEL_SR, ENGINE_SR)
    }

    /**
     * Sigmoid schedule from the reference implementation: linspace over
     * log-SNR values squashed by 1/(1+e^x); endpoints pinned to sigma_max
     * and 0.
     */
    private fun fillSigmas(arr: FloatArray, start: Float, end: Float) {
        val sz = arr.size
        val step = (end - start) / (sz - 1)
        arr[0] = start
        arr[sz - 1] = end
        for (i in 1 until sz - 1) arr[i] = arr[i - 1] + step
        for (i in 0 until sz) arr[i] = 1f / (1f + exp(arr[i]))
        arr[0] = SIGMA_MAX
        arr[sz - 1] = 0f
    }

    /** Mono -> peak normalize -> linear resample -> circular loop crossfade. */
    private fun postProcess(mono: FloatArray, srcSr: Int, dstSr: Int): FloatArray {
        var peak = 1e-6f
        for (v in mono) {
            val a = abs(v)
            if (a > peak) peak = a
        }
        val norm = OUT_PEAK / peak

        val outLen = (mono.size.toLong() * dstSr / srcSr).toInt()
        val out = FloatArray(outLen)
        val ratio = srcSr.toFloat() / dstSr
        for (i in out.indices) {
            val pos = i * ratio
            val i0 = pos.toInt().coerceIn(0, mono.size - 1)
            val i1 = (i0 + 1).coerceAtMost(mono.size - 1)
            val f = pos - i0
            out[i] = (mono[i0] + (mono[i1] - mono[i0]) * f) * norm
        }

        val fade = (0.25f * dstSr).roundToInt()
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

    private fun numElems(shape: IntArray, fallback: Int): Int {
        var n = 1
        for (d in shape) n *= if (d > 0) d else return fallback
        return n
    }

    private fun direct(bytes: Int): ByteBuffer =
        ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

    override fun close() {
        runCatching { conditioners.close() }
        runCatching { dit.close() }
        runCatching { vae.close() }
    }
}
