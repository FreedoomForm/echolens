package com.echolens.app.audio.neural

import org.tensorflow.lite.DataType
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
 * (kleidiai-examples/audiogen, app/audiogen.cpp), verified against the real
 * Arm-converted .tflite signatures (probe on x86, golden run reproduced audio):
 *
 *   conditioners_float32.tflite
 *     inputs:  input_ids int64 (1,128) | attention_mask int64 (1,128)
 *              | seconds_total f32 (1)                      [4-byte tensor!]
 *     outputs: [0] cross embeds f32 (1,65,768)
 *              [1] cross mask f32 (1,65)   (unused)
 *              [2] global conditioning f32 (1,768)
 *
 *   dit_model.tflite
 *     inputs:  cross (1,65,768) | global (1,768) | x (1,64,256) | t f32 (1)
 *              The LiteRT export REORDERS arguments relative to the torch
 *              signature, so input roles are bound by SHAPE at probe time
 *              (v1.2.0 assumed the torch order and produced garbage).
 *     output:  denoised latent (1,64,256)
 *
 *   autoencoder_model.tflite
 *     input:   latent (1,64,256)
 *     output:  planar stereo f32 (2,524288) @ 44.1 kHz  [left..., right...]
 *
 * Sampling: 8-step ping-pong with the sigmoid-sigma schedule (log-SNR
 * linspace -6..2), initial noise from java.util.Random(seed), per-step noise
 * reseeded with seed+i+4564 — exactly as the Arm reference runner.
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

    // ------------------------------------------------------- probed geometry
    private var idsIdx = -1   // conditioners: input_ids (int64)
    private var maskIdx = -1  // conditioners: attention_mask (int64)
    private var secIdx = -1   // conditioners: seconds_total (f32, 4 bytes)

    private var idxCross = -1 // dit input: cross embeds  (== conditioners out0)
    private var idxGlobal = -1// dit input: global cond    (== conditioners out2)
    private var idxX = -1     // dit input: latent noise   (1,64,256)
    private var idxT = -1     // dit input: sigma scalar   (1)

    private var idsElems = 0
    private var crossElems = 0
    private var globalElems = 0
    private var latentElems = 0
    private var aeElems = 0

    // Reusable direct buffers (sizes MUST match the tensors byte-for-byte)
    private lateinit var idsBuf: ByteBuffer
    private lateinit var maskBuf: ByteBuffer
    private lateinit var secBuf: ByteBuffer
    private lateinit var crossBuf: ByteBuffer
    private lateinit var globalBuf: ByteBuffer
    private lateinit var xBuf: ByteBuffer
    private lateinit var ditOutBuf: ByteBuffer
    private lateinit var aeOutBuf: ByteBuffer
    private lateinit var tBuf: ByteBuffer

    @Synchronized
    private fun probe() {
        // ---- conditioners: bind inputs by dtype (ids/mask int64, seconds f32)
        for (i in 0 until conditioners.inputTensorCount) {
            val t = conditioners.getInputTensor(i)
            when (t.dataType()) {
                DataType.INT64 -> if (idsIdx < 0) idsIdx = i else maskIdx = i
                DataType.FLOAT32 -> if (secIdx < 0) secIdx = i
                else -> Unit
            }
        }
        require(idsIdx >= 0 && maskIdx >= 0 && secIdx >= 0) { "conditioners: inputs not found" }
        idsElems = elems(conditioners.getInputTensor(idsIdx).shape())
        require(idsElems > 0) { "conditioners: dynamic ids shape" }

        // outputs: [0] cross, [1] cross mask (unused), [2] global
        require(conditioners.outputTensorCount >= 3) { "conditioners: unexpected outputs" }
        crossElems = elems(conditioners.getOutputTensor(0).shape())
        globalElems = elems(conditioners.getOutputTensor(2).shape())
        require(crossElems > globalElems && globalElems > 0) { "conditioners: bad outputs" }

        // ---- dit: bind ALL inputs by shape (export reorders arguments!)
        var found = 0
        for (i in 0 until dit.inputTensorCount) {
            val e = elems(dit.getInputTensor(i).shape())
            when {
                e == crossElems -> { idxCross = i; found++ }
                e == globalElems -> { idxGlobal = i; found++ }
                e == 1 -> { idxT = i; found++ }
                e > 0 -> { idxX = i; found++ }
                else -> Unit // dynamic shape — will fail the require below
            }
        }
        require(found == 4 && idxCross != idxGlobal && idxX != idxT) { "dit: cannot bind inputs" }
        latentElems = elems(dit.getInputTensor(idxX).shape())
        require(latentElems > 0) { "dit: dynamic latent shape" }

        // ---- autoencoder: latent in, planar stereo out
        require(elems(vae.getInputTensor(0).shape()) == latentElems) { "vae: latent mismatch" }
        aeElems = elems(vae.getOutputTensor(0).shape())
        require(aeElems >= latentElems * 4) { "vae: bad output shape" }

        idsBuf = direct(idsElems * 8)
        maskBuf = direct(idsElems * 8)
        secBuf = direct(4)          // seconds_total is a (1) f32 tensor
        crossBuf = direct(crossElems * 4)
        globalBuf = direct(globalElems * 4)
        xBuf = direct(latentElems * 4)
        ditOutBuf = direct(latentElems * 4)
        aeOutBuf = direct(aeElems * 4)
        tBuf = direct(4)            // sigma is a (1) f32 tensor

        android.util.Log.i(
            TAG, "probed: ids=$idsElems cross=$crossElems global=$globalElems " +
                "latent=$latentElems ae=$aeElems | dit in-map " +
                "cross=$idxCross global=$idxGlobal x=$idxX t=$idxT"
        )
    }

    /**
     * Generates one ~12 s texture. Returns mono samples at 48 kHz,
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

        val condIn = arrayOfNulls<Any>(conditioners.inputTensorCount)
        condIn[idsIdx] = idsBuf
        condIn[maskIdx] = maskBuf
        condIn[secIdx] = secBuf
        val condOut = mapOf(0 to crossBuf, 2 to globalBuf)
        conditioners.runForMultipleInputsOutputs(condIn.requireNoNulls(), condOut)

        // ---- 2. DiT ping-pong sampling --------------------------------------
        val xf = FloatArray(latentElems)
        val rng = java.util.Random(seed)
        for (i in 0 until latentElems) xf[i] = rng.nextGaussian().toFloat() * SIGMA_MAX

        val sigmas = FloatArray(steps + 1)
        fillSigmas(sigmas, LOGSNR_MAX, 2.0f)

        val xF = xBuf.order(ByteOrder.nativeOrder()).asFloatBuffer()
        val dF = ditOutBuf.order(ByteOrder.nativeOrder()).asFloatBuffer()

        val ditIn = arrayOfNulls<Any>(dit.inputTensorCount)
        ditIn[idxCross] = crossBuf
        ditIn[idxGlobal] = globalBuf
        ditIn[idxX] = xBuf
        ditIn[idxT] = tBuf
        val ditOut = mapOf(0 to ditOutBuf)

        for (i in 0 until steps) {
            val cur = sigmas[i]
            val next = sigmas[i + 1]
            tBuf.rewind()
            tBuf.putFloat(cur)
            xF.rewind()
            xF.put(xf)
            dit.runForMultipleInputsOutputs(ditIn.requireNoNulls(), ditOut)

            dF.rewind()
            val stepRng = java.util.Random(seed + i + RNG_OFFSET)
            for (j in 0 until latentElems) {
                val denoised = xf[j] - cur * dF.get(j)
                xf[j] = (1f - next) * denoised + next * stepRng.nextGaussian().toFloat()
            }
            onProgress?.invoke(i + 1, steps)
        }

        // ---- 3. VAE decode --------------------------------------------------
        xF.rewind()
        xF.put(xf)
        vae.runForMultipleInputsOutputs(arrayOf(xBuf), mapOf(0 to aeOutBuf))

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

    private fun elems(shape: IntArray): Int {
        var n = 1
        for (d in shape) {
            if (d <= 0) return -1
            n *= d
        }
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
