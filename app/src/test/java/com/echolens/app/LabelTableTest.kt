package com.echolens.app

import com.echolens.app.core.LabelTable
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the structural invariants of the generated class table. If the
 * YOLOE vocabulary is ever regenerated, these tests catch a table that is
 * inconsistent with the model (wrong size, misaligned arrays, broken COCO
 * prefix shared with the fallback detector).
 */
class LabelTableTest {

    @Test
    fun `table is parallel and has the declared size`() {
        val n = LabelTable.EN.size
        assertEquals("LabelTable claims 426 classes", 426, n)
        assertEquals("RU size mismatch", n, LabelTable.RU.size)
        assertEquals("MATERIAL size mismatch", n, LabelTable.MATERIAL.size)
        assertEquals("HEIGHT_M size mismatch", n, LabelTable.HEIGHT_M.size)
        assertEquals("IMPORTANCE size mismatch", n, LabelTable.IMPORTANCE.size)
    }

    @Test
    fun `first 80 entries are COCO-80 for the fallback detector`() {
        // Canonical Ultralytics COCO order — spot-check fixed positions.
        assertEquals("person", LabelTable.EN[0])
        assertEquals("car", LabelTable.EN[2])
        assertEquals("bird", LabelTable.EN[14])
        assertEquals("cat", LabelTable.EN[15])
        assertEquals("dog", LabelTable.EN[16])
        assertEquals("toothbrush", LabelTable.EN[79])
    }

    @Test
    fun `spec builds for every class`() {
        for (i in LabelTable.EN.indices) {
            val s = LabelTable.spec(i)
            assertTrue("class $i has non-positive reference distance", s.refDistanceM > 0f)
            assertTrue("class $i importance out of range", s.importance in 0f..1f)
        }
    }
}
