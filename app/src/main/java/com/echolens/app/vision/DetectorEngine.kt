package com.echolens.app.vision

import android.content.Context
import android.graphics.Bitmap
import com.echolens.app.core.MaterialSpec
import com.echolens.app.core.TextureLibrary
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * On-device object detector: quantized SSD MobileNet v1 (COCO, 80 classes),
 * running on the plain TensorFlow Lite Interpreter.
 *
 * Model input : [1, 300, 300, 3] uint8 (RGB)
 * Model output: TFLite_Detection_PostProcess — boxes / classes / scores / count
 *               classes are 1-based COCO ids into the bundled labelmap.
 */
class DetectorEngine(
    context: Context,
    private val scoreThreshold: Float = 0.55f,
    modelAsset: String = "detect.tflite",
    labelsAsset: String = "labelmap.txt"
) : AutoCloseable {

    class Detection(
        val label: String,
        val ruName: String,
        val spec: MaterialSpec,
        val score: Float,
        /** BBox center X in the upright image, 0..1, left to right. */
        val cxNorm: Float,
        /** BBox center Y in the upright image, 0..1, top to bottom. */
        val cyNorm: Float,
        /** BBox height as a fraction of image height. */
        val hFrac: Float
    )

    private val interpreter: Interpreter
    private val labels: List<String>
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputBuffer: ByteBuffer
    private var pixels: IntArray

    // Recycled output arrays — the interpreter rewrites them on every call.
    private val outputBoxes = Array(1) { Array(MAX_DETECTIONS) { FloatArray(4) } }
    private val outputClasses = Array(1) { FloatArray(MAX_DETECTIONS) }
    private val outputScores = Array(1) { FloatArray(MAX_DETECTIONS) }
    private val outputCount = FloatArray(1)

    init {
        // Read the model into a direct buffer — works regardless of asset compression.
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val model = ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .put(bytes)
            .apply { rewind() }
        interpreter = Interpreter(model, Interpreter.Options().apply { numThreads = 2 })
        labels = context.assets.open(labelsAsset).use { stream ->
            stream.bufferedReader().readLines()
        }.map { it.trim() }

        val shape = interpreter.getInputTensor(0).shape() // [1, H, W, 3]
        inputHeight = shape[1]
        inputWidth = shape[2]
        inputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3)
            .order(ByteOrder.nativeOrder())
        pixels = IntArray(inputWidth * inputHeight)
    }

    /** Detects objects on an upright (portrait) bitmap of any size. */
    fun detect(upright: Bitmap): List<Detection> {
        val scaled = if (upright.width == inputWidth && upright.height == inputHeight) {
            upright
        } else {
            Bitmap.createScaledBitmap(upright, inputWidth, inputHeight, true)
        }
        scaled.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (scaled !== upright) scaled.recycle()

        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.put(((p shr 16) and 0xFF).toByte()) // R
            inputBuffer.put(((p shr 8) and 0xFF).toByte())  // G
            inputBuffer.put((p and 0xFF).toByte())          // B
        }
        inputBuffer.rewind()

        interpreter.run(inputBuffer, arrayOf(outputBoxes, outputClasses, outputScores, outputCount))

        val n = outputCount[0].toInt().coerceIn(0, MAX_DETECTIONS)
        val result = ArrayList<Detection>(n)
        for (i in 0 until n) {
            val score = outputScores[0][i]
            if (score < scoreThreshold) continue
            val classId = outputClasses[0][i].toInt()
            if (classId < 1 || classId > labels.size) continue
            val label = labels[classId - 1]
            if (label.length < 2) continue // skip "???" placeholder rows

            val box = outputBoxes[0][i]
            val ymin = box[0].coerceIn(0f, 1f)
            val xmin = box[1].coerceIn(0f, 1f)
            val ymax = box[2].coerceIn(0f, 1f)
            val xmax = box[3].coerceIn(0f, 1f)
            if (ymax <= ymin || xmax <= xmin) continue

            val spec = TextureLibrary.spec(label)
            result.add(
                Detection(
                    label = label,
                    ruName = spec.ruName,
                    spec = spec,
                    score = score,
                    cxNorm = (xmin + xmax) / 2f,
                    cyNorm = (ymin + ymax) / 2f,
                    hFrac = (ymax - ymin).coerceAtLeast(0.004f)
                )
            )
        }
        return result
    }

    override fun close() {
        interpreter.close()
    }

    private companion object {
        const val MAX_DETECTIONS = 10
    }
}
