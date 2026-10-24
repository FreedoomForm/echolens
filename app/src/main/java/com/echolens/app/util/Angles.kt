package com.echolens.app.util

import kotlin.math.PI

object Angles {

    /** Normalize angle to [-π, π]. */
    fun normalize(a: Float): Float {
        val twoPi = (2.0 * PI).toFloat()
        val pi = PI.toFloat()
        var x = a
        while (x > pi) x -= twoPi
        while (x < -pi) x += twoPi
        return x
    }

    fun clamp(v: Float, lo: Float, hi: Float): Float = if (v < lo) lo else if (v > hi) hi else v

    fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
        if (edge0 == edge1) return if (x < edge0) 0f else 1f
        val t = clamp((x - edge0) / (edge1 - edge0), 0f, 1f)
        return t * t * (3f - 2f * t)
    }

    fun deg(rad: Float): Float = Math.toDegrees(rad.toDouble()).toFloat()

    fun rad(deg: Float): Float = Math.toRadians(deg.toDouble()).toFloat()
}
