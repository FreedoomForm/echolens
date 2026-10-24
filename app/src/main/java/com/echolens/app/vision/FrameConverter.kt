package com.echolens.app.vision

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

/**
 * Converts CameraX YUV_420_888 frames into upright ARGB Bitmaps.
 *
 * YUV_420_888 is the camera's native output format — it works on every
 * device, unlike the emulated RGBA_8888 path. All buffers are reused,
 * zero steady-state allocations. Includes BT.601 color conversion and
 * rotation into the upright portrait frame.
 */
class FrameConverter {

    private var outBitmap: Bitmap? = null
    private var argb = IntArray(0)        // upright output pixels
    private var yArr = ByteArray(0)       // packed luma
    private var uArr = ByteArray(0)       // packed chroma U
    private var vArr = ByteArray(0)       // packed chroma V
    private var srcW = 0
    private var srcH = 0

    fun toUprightBitmap(image: ImageProxy): Bitmap {
        val w = image.width
        val h = image.height
        if (w != srcW || h != srcH) {
            srcW = w
            srcH = h
            yArr = ByteArray(w * h)
            val cw = (w + 1) / 2
            val ch = (h + 1) / 2
            uArr = ByteArray(cw * ch)
            vArr = ByteArray(cw * ch)
        }

        copyPlane(image.planes[0].buffer, image.planes[0].rowStride, image.planes[0].pixelStride,
            yArr, w, h)
        copyPlane(image.planes[1].buffer, image.planes[1].rowStride, image.planes[1].pixelStride,
            uArr, (w + 1) / 2, (h + 1) / 2)
        copyPlane(image.planes[2].buffer, image.planes[2].rowStride, image.planes[2].pixelStride,
            vArr, (w + 1) / 2, (h + 1) / 2)

        val rot = image.imageInfo.rotationDegrees
        val cw = (w + 1) / 2
        val dw: Int
        val dh: Int
        when (rot) {
            90 -> { dw = h; dh = w }
            270 -> { dw = h; dh = w }
            180 -> { dw = w; dh = h }
            else -> { dw = w; dh = h }
        }
        if (argb.size < dw * dh) argb = IntArray(dw * dh)

        var i = 0
        when (rot) {
            90 -> {
                // 90° clockwise: dst(x' = h-1-y, y' = x)
                for (y in 0 until h) {
                    val yOff = y * w
                    val uOff = (y shr 1) * cw
                    for (x in 0 until w) {
                        argb[x * dw + (h - 1 - y)] = yuvToArgb(yArr[yOff + x], uOff + (x shr 1))
                        i++
                    }
                }
            }
            270 -> {
                // 270° clockwise: dst(x' = y, y' = w-1-x)
                for (y in 0 until h) {
                    val yOff = y * w
                    val uOff = (y shr 1) * cw
                    for (x in 0 until w) {
                        argb[(w - 1 - x) * dw + y] = yuvToArgb(yArr[yOff + x], uOff + (x shr 1))
                        i++
                    }
                }
            }
            180 -> {
                for (y in 0 until h) {
                    val yOff = y * w
                    val uOff = (y shr 1) * cw
                    for (x in 0 until w) {
                        argb[(h - 1 - y) * dw + (w - 1 - x)] = yuvToArgb(yArr[yOff + x], uOff + (x shr 1))
                        i++
                    }
                }
            }
            else -> {
                for (y in 0 until h) {
                    val yOff = y * w
                    val uOff = (y shr 1) * cw
                    for (x in 0 until w) {
                        argb[yOff + x] = yuvToArgb(yArr[yOff + x], uOff + (x shr 1))
                        i++
                    }
                }
            }
        }

        var bmp = outBitmap
        if (bmp == null || bmp.width != dw || bmp.height != dh) {
            bmp = Bitmap.createBitmap(dw, dh, Bitmap.Config.ARGB_8888)
            outBitmap = bmp
        }
        bmp.setPixels(argb, 0, dw, 0, 0, dw, dh)
        return bmp
    }

    private fun copyPlane(
        buffer: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        dst: ByteArray,
        width: Int,
        height: Int
    ) {
        buffer.rewind()
        if (pixelStride == 1 && rowStride == width) {
            buffer.get(dst, 0, minOf(dst.size, buffer.remaining()))
        } else if (pixelStride == 1) {
            for (row in 0 until height) {
                buffer.position(row * rowStride)
                buffer.get(dst, row * width, width)
            }
        } else {
            // Interleaved (pixelStride == 2) or exotic layouts: absolute reads.
            for (row in 0 until height) {
                val base = row * rowStride
                val dstOff = row * width
                for (col in 0 until width) {
                    dst[dstOff + col] = buffer.get(base + col * pixelStride)
                }
            }
        }
    }

    private fun yuvToArgb(yByte: Byte, uvIndex: Int): Int {
        val yVal = (yByte.toInt() and 0xFF) - 16
        val uVal = (uArr[uvIndex].toInt() and 0xFF) - 128
        val vVal = (vArr[uvIndex].toInt() and 0xFF) - 128
        val y2 = 1192 * yVal
        var r = (y2 + 1634 * vVal) shr 10
        var g = (y2 - 833 * vVal - 400 * uVal) shr 10
        var b = (y2 + 2066 * uVal) shr 10
        if (r < 0) r = 0 else if (r > 255) r = 255
        if (g < 0) g = 0 else if (g > 255) g = 255
        if (b < 0) b = 0 else if (b > 255) b = 255
        return 0xFF000000.toInt() or (r shl 16) or (g shl 8) or b
    }
}
