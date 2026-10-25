package com.echolens.app

import com.echolens.app.core.RuText
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The spoken scene description is a core accessibility surface for the
 * Russian-first audience — these tests pin the direction bands and Russian
 * plural forms (one/few/many meters).
 */
class RuTextTest {

    @Test
    fun `direction bands cover the full circle without gaps`() {
        // Every 5 degrees must yield a non-empty direction word.
        for (deg in -180..179 step 5) {
            assertTrue(RuText.direction(deg.toFloat()).isNotBlank())
        }
    }

    @Test
    fun `direction center and sides are stable`() {
        assertEquals("прямо", RuText.direction(0f))
        assertEquals("прямо", RuText.direction(10f))
        assertEquals("чуть справа", RuText.direction(30f))
        assertEquals("справа", RuText.direction(80f))
        assertEquals("слева", RuText.direction(-80f))
        assertEquals("позади", RuText.direction(180f))
    }

    @Test
    fun `distance uses russian plural forms`() {
        assertEquals("ближе метра", RuText.distance(0.7f))
        assertEquals("1 метр", RuText.distance(1.2f))
        assertEquals("2 метра", RuText.distance(2.1f))
        assertEquals("5 метров", RuText.distance(5.4f))
        assertEquals("21 метр", RuText.distance(21f))
        assertEquals("22 метра", RuText.distance(22f))
    }

    @Test
    fun `distance is capped at 99 meters`() {
        assertTrue(RuText.distance(150f).startsWith("99"))
    }
}
