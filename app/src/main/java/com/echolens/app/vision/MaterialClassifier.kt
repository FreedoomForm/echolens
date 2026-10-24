package com.echolens.app.vision

import android.content.Context
import android.graphics.Bitmap
import com.echolens.app.audio.neural.NeuralModels
import com.echolens.app.core.Material
import com.echolens.app.core.MaterialSpec
import com.echolens.app.core.TextureLibrary
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Optional visual material classifier (MINC-23 fine-tune, MobileNetV3-Small,
 * uint8 TFLite) that refines what a detected object "sounds like".
 *
 * The detector says "bottle" (glass by default); the classifier looks at the
 * actual pixels and may say "plastic" — the material, and therefore the
 * sound texture, is corrected accordingly.
 *
 * The model file  minc23_mobilenet_v3_int8.tflite  is optional and lives in
 * the app models directory (see NeuralModels). If absent, the feature is
 * silently disabled and TextureLibrary defaults are used.
 */
class MaterialClassifier private constructor(
    private val interpreter: Interpreter
) : AutoCloseable {

    companion object {
        const val MODEL_FILE = "minc23_mobilenet_v3_int8.tflite"
        private const val TAG = "MaterialClf"

        /** MINC-23 label order — must match tools/train_minc_classifier.py. */
        val MINC_LABELS = arrayOf(
            "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
            "hair", "leather", "metal", "mirror", "other", "painted", "paper",
            "plastic", "polishedstone", "skin", "sky", "stone", "tile",
            "wallpaper", "water", "wood"
        )

        private val MINC_TO_MATERIAL = mapOf(
            "wood" to Material.WOOD,
            "metal" to Material.METAL,
            "glass" to Material.GLASS,
            "mirror" to Material.GLASS,
            "ceramic" to Material.CERAMIC,
            "tile" to Material.CERAMIC,
            "brick" to Material.CERAMIC,
            "stone" to Material.CERAMIC,
            "polishedstone" to Material.CERAMIC,
            "fabric" to Material.FABRIC,
            "carpet" to Material.FABRIC,
            "leather" to Material.FABRIC,
            "wallpaper" to Material.FABRIC,
            "foliage" to Material.PLANT,
            "paper" to Material.PAPER,
            "food" to Material.FOOD,
            "water" to Material.WATER,
            "skin" to Material.PERSON,
            "hair" to Material.PERSON,
            "plastic" to Material.NEUTRAL,
            "painted" to Material.NEUTRAL,
            "other" to Material.NEUTRAL,
            "sky" to Material.NEUTRAL
        )

        const val CONFIDENCE_GATE = 0.5f

        fun tryCreate(context: Context): MaterialClassifier? {
            val source: Any = try {
                val file = java.io.File(NeuralModels.modelsDir(context), MODEL_FILE)
                if (file.isFile && file.length() > 100_000L) {
                    file
                } else {
                    // Optional asset fallback (not bundled by default).
                    val inRoot = context.assets.list("")?.contains(MODEL_FILE) == true
                    val inModels = context.assets.list("models")?.contains(MODEL_FILE) == true
                    when {
                        inRoot -> {
                            val bytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
                            ByteBuffer.wrap(bytes)
                        }
                        inModels -> {
                            val bytes = context.assets.open("models/$MODEL_FILE").use { it.readBytes() }
                            ByteBuffer.wrap(bytes)
                        }
                        else -> return null
                    }
                }
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "Model file lookup failed", t)
                return null
            }
            return try {
                val opts = Interpreter.Options().setNumThreads(2)
                val interp = when (val src = source) {
                    is java.io.File -> Interpreter(src, opts)
                    is ByteBuffer -> Interpreter(src, opts)
                    else -> return null
                }
                MaterialClassifier(interp)
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "Material classifier unavailable", t)
                null
            }
        }
    }

    private var inputSize = 224
    private var uint8Input = true
    private var probed = false
    private var pixelBuf: ByteBuffer = ByteBuffer.allocateDirect(0)
    private var outBuf: ByteBuffer = ByteBuffer.allocateDirect(0)

    class Result(val mincLabel: String, val score: Float, val material: Material)

    @Synchronized
    fun classify(upright: Bitmap, cxNorm: Float, cyNorm: Float, hFrac: Float): Result? {
        return try {
            if (!probed) probe()
            val w = upright.width
            val h = upright.height
            // Square crop around the detection center, side = 1.2 x box height.
            val side = (hFrac * h * 1.2f).toInt().coerceIn(24, minOf(w, h))
            val left = (cxNorm * w - side / 2f).toInt().coerceIn(0, w - side)
            val top = (cyNorm * h - side / 2f).toInt().coerceIn(0, h - side)
            val crop = Bitmap.createBitmap(upright, left, top, side, side)
            val scaled = Bitmap.createScaledBitmap(crop, inputSize, inputSize, true)
            if (scaled !== crop) crop.recycle()

            if (uint8Input) {
                val pixels = IntArray(inputSize * inputSize)
                scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
                pixelBuf.rewind()
                for (p in pixels) {
                    pixelBuf.put(((p shr 16) and 0xFF).toByte())
                    pixelBuf.put(((p shr 8) and 0xFF).toByte())
                    pixelBuf.put((p and 0xFF).toByte())
                }
                pixelBuf.rewind()
            } else {
                val pixels = IntArray(inputSize * inputSize)
                scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
                val fb = pixelBuf.order(ByteOrder.nativeOrder()).asFloatBuffer()
                fb.rewind()
                for (p in pixels) {
                    fb.put(((p shr 16) and 0xFF) / 255f)
                    fb.put(((p shr 8) and 0xFF) / 255f)
                    fb.put((p and 0xFF) / 255f)
                }
                fb.rewind()
            }
            if (scaled !== upright) scaled.recycle()

            val outF = outBuf.order(ByteOrder.nativeOrder()).asFloatBuffer()
            outF.rewind()
            interpreter.run(pixelBuf, outBuf)

            var best = 0
            var bestScore = Float.NEGATIVE_INFINITY
            outF.rewind()
            for (i in 0 until MINC_LABELS.size) {
                val v = outF.get(i)
                if (v > bestScore) {
                    bestScore = v
                    best = i
                }
            }
            val probs = softmaxScores(outF)
            val score = if (probs != null) probs[best] else bestScore.coerceIn(0f, 1f)
            if (score < CONFIDENCE_GATE) return null
            val label = MINC_LABELS[best.coerceIn(0, MINC_LABELS.size - 1)]
            Result(label, score, MINC_TO_MATERIAL[label] ?: Material.NEUTRAL)
        } catch (t: Throwable) {
            android.util.Log.w(TAG, "classify failed", t)
            null
        }
    }

    /** If the output is logits, a softmax gives comparable confidence. */
    private fun softmaxScores(buf: java.nio.FloatBuffer): FloatArray? {
        if (buf.remaining() != MINC_LABELS.size) return null
        var max = Float.NEGATIVE_INFINITY
        for (i in 0 until buf.remaining()) max = maxOf(max, buf.get(i))
        var sum = 0f
        val e = FloatArray(buf.remaining())
        for (i in e.indices) {
            e[i] = kotlin.math.exp(buf.get(i) - max)
            sum += e[i]
        }
        if (sum <= 0f) return null
        for (i in e.indices) e[i] /= sum
        return e
    }

    @Synchronized
    private fun probe() {
        val inT = interpreter.getInputTensor(0)
        inputSize = if (inT.shape().size >= 4 && inT.shape()[1] > 0) inT.shape()[1] else 224
        uint8Input = inT.dataType() == org.tensorflow.lite.DataType.UINT8
        val inBytes = inputSize * inputSize * 3 * if (uint8Input) 1 else 4
        pixelBuf = ByteBuffer.allocateDirect(inBytes).order(ByteOrder.nativeOrder())
        val outCount = interpreter.outputTensorCount
        val outT = interpreter.getOutputTensor(outCount - 1)
        val outElems = outT.shape().let { s -> s.fold(1) { acc, d -> acc * if (d > 0) d else MINC_LABELS.size } }
        val outBytes = outElems * if (outT.dataType() == org.tensorflow.lite.DataType.UINT8) 1 else 4
        outBuf = ByteBuffer.allocateDirect(outBytes).order(ByteOrder.nativeOrder())
        probed = true
    }

    /** Convenience: override the spec's material, keeping names and geometry. */
    fun refine(spec: MaterialSpec, result: Result): MaterialSpec =
        if (result.material != spec.material) {
            MaterialSpec(result.material, spec.ruName, spec.realHeightM, spec.importance, spec.refDistanceM)
        } else spec

    override fun close() {
        runCatching { interpreter.close() }
    }
}
