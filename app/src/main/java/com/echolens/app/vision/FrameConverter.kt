package com.echolens.app.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

/**
 * Converts CameraX RGBA_8888 ImageProxy frames into upright Bitmaps.
 * All intermediate buffers are reused — zero steady-state allocations.
 */
class FrameConverter {

    private var srcBitmap: Bitmap? = null
    private var dstBitmap: Bitmap? = null
    private var canvas: Canvas? = null
    private val matrix = Matrix()
    private var packed: ByteBuffer? = null
    private var rowBuf: ByteArray? = null

    fun toUprightBitmap(image: ImageProxy): Bitmap {
        val w = image.width
        val h = image.height
        val plane = image.planes[0]
        val rowStride = plane.rowStride
        val direct = plane.buffer

        var src = srcBitmap
        if (src == null || src.width != w || src.height != h) {
            src = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            srcBitmap = src
            packed = ByteBuffer.allocate(w * h * 4)
            rowBuf = ByteArray(rowStride)
        }

        if (rowStride == w * 4) {
            direct.rewind()
            src.copyPixelsFromBuffer(direct)
        } else {
            // Stride-aware copy: strip row padding.
            val p = packed!!
            var r = rowBuf
            if (r == null || r.size < rowStride) {
                r = ByteArray(rowStride)
                rowBuf = r
            }
            p.rewind()
            for (y in 0 until h) {
                direct.position(y * rowStride)
                direct.get(r, 0, w * 4)
                p.put(r, 0, w * 4)
            }
            p.rewind()
            src.copyPixelsFromBuffer(p)
        }

        val rot = image.imageInfo.rotationDegrees
        if (rot == 0) return src

        val dw = if (rot == 90 || rot == 270) h else w
        val dh = if (rot == 90 || rot == 270) w else h
        var dst = dstBitmap
        if (dst == null || dst.width != dw || dst.height != dh) {
            dst = Bitmap.createBitmap(dw, dh, Bitmap.Config.ARGB_8888)
            dstBitmap = dst
            canvas = Canvas(dst)
        }
        matrix.reset()
        matrix.setRotate(rot.toFloat())
        val c = canvas!!
        c.setMatrix(matrix)
        c.drawBitmap(src, 0f, 0f, null)
        return dst
    }
}
