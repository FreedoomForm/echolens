package com.echolens.app.audio

/**
 * Compact stereo Schroeder reverb (4 parallel combs + 1 allpass per channel).
 * A touch of uncorrelated late reverb helps the brain judge sounds as
 * coming from the outside world rather than from inside the head.
 */
class SimpleReverb(sampleRate: Int, private val wet: Float = 0.10f) {

    private class Comb(val size: Int) {
        val buf = FloatArray(size)
        var idx = 0
        var filter = 0f
        fun process(x: Float): Float {
            val y = buf[idx]
            filter += 0.28f * (y - filter)
            buf[idx] = x + filter * 0.78f
            if (++idx >= size) idx = 0
            return y
        }
    }

    private class Allpass(val size: Int) {
        val buf = FloatArray(size)
        var idx = 0
        fun process(x: Float): Float {
            val bufOut = buf[idx]
            val out = -x + bufOut
            buf[idx] = x + bufOut * 0.7f
            if (++idx >= size) idx = 0
            return out
        }
    }

    private val scale = sampleRate / 48000f
    private fun d(n: Int) = (n * scale).toInt().coerceAtLeast(4)

    // Slightly different tap lengths per channel → decorrelated stereo tail.
    private val combsL = arrayOf(Comb(d(729)), Comb(d(907)), Comb(d(1071)), Comb(d(1273)))
    private val combsR = arrayOf(Comb(d(752)), Comb(d(930)), Comb(d(1094)), Comb(d(1296)))
    private val apL = Allpass(d(189))
    private val apR = Allpass(d(211))

    fun process(l: FloatArray, r: FloatArray, frames: Int) {
        for (i in 0 until frames) {
            val inL = l[i] * 0.6f
            val inR = r[i] * 0.6f
            var wl = 0f
            var wr = 0f
            for (c in 0 until 4) {
                wl += combsL[c].process(inL)
                wr += combsR[c].process(inR)
            }
            l[i] += apL.process(wl * 0.25f) * wet
            r[i] += apR.process(wr * 0.25f) * wet
        }
    }
}
