package com.echolens.app.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import com.echolens.app.core.MaterialSpec
import com.echolens.app.core.TextureLibrary
import java.nio.FloatBuffer

/**
 * Real-time object detector: YOLO26-Nano (COCO-80), exported to ONNX,
 * running on ONNX Runtime. End-to-end (NMS-free) output: [1, 300, 6]
 * = x1, y1, x2, y2, score, class_id in letterboxed input pixels.
 *
 * COCO mAP50-95 ≈ 40.3 (YOLO26-N) vs ≈ 23-25 (SSD MobileNet v1) —
 * roughly +70% relative quality at similar on-device latency.
 */
class YoloDetector(
    context: Context,
    modelAsset: String = "yolo26n.onnx",
    private val scoreThreshold: Float = 0.35f
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
        val hFrac: Float,
        /** Relative depth from Depth Anything (higher = closer), may be null. */
        val relDepth: Float?
    ) {
        fun withDepth(rd: Float?): Detection =
            if (rd == null) this else Detection(label, ruName, spec, score, cxNorm, cyNorm, hFrac, rd)

        /** Material refinement (e.g. from the optional MINC material classifier). */
        fun withSpec(newSpec: MaterialSpec): Detection =
            Detection(label, newSpec.ruName, newSpec, score, cxNorm, cyNorm, hFrac, relDepth)
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val inputSize: Int
    private var pixels = IntArray(0)
    private var net = FloatArray(0)
    private var srcW = 0
    private var srcH = 0

    init {
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.iterator().next()
        val info = session.getInputInfo()[inputName]!!.info as TensorInfo
        val dims = info.shape
        inputSize = if (dims.size >= 4 && dims[2] > 0) dims[2].toInt() else 480
        net = FloatArray(3 * inputSize * inputSize)
    }

    fun detect(upright: Bitmap): MutableList<Detection> {
        val w = upright.width
        val h = upright.height
        if (w != srcW || h != srcH) {
            srcW = w
            srcH = h
            pixels = IntArray(w * h)
        }
        val s = inputSize
        val scale = minOf(s.toFloat() / w, s.toFloat() / h)
        val rw = (w * scale).toInt().coerceAtMost(s)
        val rh = (h * scale).toInt().coerceAtMost(s)
        val px = (s - rw) / 2
        val py = (s - rh) / 2

        upright.getPixels(pixels, 0, w, 0, 0, w, h)
        java.util.Arrays.fill(net, GRAY)
        val plane = s * s
        for (y in 0 until rh) {
            val srcOff = y * w
            val dstRow = (py + y) * s + px
            for (x in 0 until rw) {
                val p = pixels[srcOff + x]
                val di = dstRow + x
                net[di] = ((p shr 16) and 0xFF) * INV255
                net[plane + di] = ((p shr 8) and 0xFF) * INV255
                net[2 * plane + di] = (p and 0xFF) * INV255
            }
        }

        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(net), longArrayOf(1, 3, s.toLong(), s.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val rows = (results[0].value as Array<Array<FloatArray>>)[0] // [300][6]
                val out = ArrayList<Detection>(12)
                for (row in rows) {
                    val score = row[4]
                    if (score < scoreThreshold) continue
                    val cls = row[5].toInt().coerceIn(0, LABELS.size - 1)
                    val label = LABELS[cls]
                    val spec = TextureLibrary.spec(label)
                    // Un-letterbox back to upright pixel coords, then normalize.
                    val x1 = (row[0] - px) / scale
                    val x2 = (row[2] - px) / scale
                    val y1 = (row[1] - py) / scale
                    val y2 = (row[3] - py) / scale
                    if (x2 <= 0f || y2 <= 0f || x1 >= w || y1 >= h) continue
                    val cx = (((x1 + x2) / 2f) / w).coerceIn(0f, 1f)
                    val cy = (((y1 + y2) / 2f) / h).coerceIn(0f, 1f)
                    val hf = ((y2 - y1) / h).coerceAtLeast(0.004f)
                    out.add(Detection(label, spec.ruName, spec, score, cx, cy, hf, null))
                }
                out.sortByDescending { it.score }
                if (out.size > 12) out.subList(12, out.size).clear()
                return out
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val GRAY = 114f / 255f
        private const val INV255 = 1f / 255f

        /** Ultralytics COCO-80 class order. */
        val LABELS = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }
}
