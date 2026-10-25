package com.echolens.app

import com.echolens.app.util.Angles
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI

/**
 * Unit tests for the angle math used by the binaural renderer and the
 * world-tracking pipeline. Pure JVM — no Android dependencies.
 */
class AnglesTest {

    @Test
    fun `normalize keeps result in -pi pi`() {
        assertEquals(0f, Angles.normalize(0f), 1e-6f)
        assertEquals(0f, Angles.normalize(2f * PI.toFloat()), 1e-4f)
        assertEquals(-0.1f, Angles.normalize(-0.1f), 1e-6f)
        // One full turn must land back at the same angle
        val a = 1.23f
        assertEquals(a, Angles.normalize(a + 2f * PI.toFloat()), 1e-4f)
        assertEquals(a, Angles.normalize(a - 2f * PI.toFloat()), 1e-4f)
        val n = Angles.normalize(37.5f)
        assertTrue("normalized angle out of range: $n", n >= -PI.toFloat() && n <= PI.toFloat())
    }

    @Test
    fun `clamp respects bounds`() {
        assertEquals(2f, Angles.clamp(5f, -2f, 2f), 1e-6f)
        assertEquals(-2f, Angles.clamp(-5f, -2f, 2f), 1e-6f)
        assertEquals(0.5f, Angles.clamp(0.5f, -2f, 2f), 1e-6f)
    }

    @Test
    fun `smoothstep is 0 before edge0 and 1 after edge1`() {
        assertEquals(0f, Angles.smoothstep(1f, 2f, 0.5f), 1e-6f)
        assertEquals(1f, Angles.smoothstep(1f, 2f, 2.5f), 1e-6f)
        assertEquals(0.5f, Angles.smoothstep(1f, 2f, 1.5f), 1e-6f)
    }

    @Test
    fun `smoothstep is monotonic inside the band`() {
        var prev = -1f
        var x = 0f
        while (x <= 1f) {
            val v = Angles.smoothstep(0f, 1f, x)
            assertTrue("smoothstep decreased at x=$x", v >= prev)
            prev = v
            x += 0.05f
        }
    }
}
