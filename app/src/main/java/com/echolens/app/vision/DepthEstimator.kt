package com.echolens.app.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

/**
 * Monocular depth: Depth Anything V2 Small (INT8 ONNX, Apache-2.0).
 * Relative depth, higher value = closer to the camera.
 *
 * Runs on a fraction of frames (every ~5th) and provides the distance
 * signal that drives sound loudness — fused with the bbox-size prior
 * inside SceneTracker. Input is dynamic, so we run at 308×308 for speed
 * (~100-150 ms on a phone, ~50-70 ms at 224).
 */
class DepthEstimator(
    context: Context,
    modelAsset: String = "depth_v2_small.onnx",
    private val netSize: Int = 308
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private var scaledPixels = IntArray(netSize * netSize)
    private val net = FloatArray(3 * netSize * netSize)
    private var map: Array<FloatArray>? = null

    @Volatile var lastMs = 0f
        private set

    val hasMap: Boolean get() = map != null

    init {
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.iterator().next()
    }

    fun estimate(upright: Bitmap) {
        val scaled = Bitmap.createScaledBitmap(upright, netSize, netSize, true)
        scaled.getPixels(scaledPixels, 0, netSize, 0, 0, netSize, netSize)
        if (scaled !== upright) scaled.recycle()

        val plane = netSize * netSize
        for (i in 0 until plane) {
            val p = scaledPixels[i]
            net[i] = (((p shr 16) and 0xFF) * INV255 - MEAN[0]) / STD[0]
            net[plane + i] = (((p shr 8) and 0xFF) * INV255 - MEAN[1]) / STD[1]
            net[2 * plane + i] = ((p and 0xFF) * INV255 - MEAN[2]) / STD[2]
        }

        val t0 = System.currentTimeMillis()
        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(net), longArrayOf(1, 3, netSize.toLong(), netSize.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                map = (results[0].value as Array<Array<FloatArray>>)[0]
            }
        }
        lastMs = (System.currentTimeMillis() - t0).toFloat()
    }

    /** Median relative depth in a 3×3 patch at normalized image coords; higher = closer. */
    fun sampleAt(cxNorm: Float, cyNorm: Float): Float? {
        val m = map ?: return null
        val s = m.size
        if (s < 8) return null
        val cx = (cxNorm * s).toInt().coerceIn(1, s - 2)
        val cy = (cyNorm * s).toInt().coerceIn(1, s - 2)
        var k = 0
        val vals = FloatArray(9)
        for (dy in -1..1) {
            val row = m[cy + dy]
            for (dx in -1..1) {
                vals[k++] = row[cx + dx]
            }
        }
        vals.sort()
        return vals[4]
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val INV255 = 1f / 255f
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
