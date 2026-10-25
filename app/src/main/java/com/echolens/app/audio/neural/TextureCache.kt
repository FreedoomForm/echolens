package com.echolens.app.audio.neural

import android.content.Context
import com.echolens.app.core.Material
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Persistent cache of generated neural textures.
 *
 * Each entry is a seamless ~10 s mono loop at the engine sample rate (48 kHz),
 * stored as raw little-endian float32 plus a small json-style meta file.
 * There is NO audio bank inside the APK — everything here is generated
 * on this very phone and reused across sessions.
 */
class TextureCache(private val context: Context) {

    class Entry(
        val file: File,
        /** Mono samples at [sampleRate]. */
        @Volatile var samples: FloatArray?,
        val sampleRate: Int,
        val durationSec: Float,
        val createdMs: Long
    )

    private val dir: File = File(context.filesDir, "neural_textures").apply { mkdirs() }
    private val mem = ConcurrentHashMap<String, Entry>()
    private var memBytes = 0L

    companion object {
        private const val EXT = ".pcm"
        private const val META = ".meta"
        private const val SR = 48000
        private const val MAX_MEM_BYTES = 96L * 1024 * 1024 // ~12 loops in RAM
    }

    fun key(material: Material, variant: Int): String = "${material.name.lowercase()}_v$variant"

    fun find(material: Material, variant: Int): Entry? {
        val k = key(material, variant)
        mem[k]?.let { return it }
        val f = File(dir, k + EXT)
        if (!f.isFile || f.length() < 4096L) return null
        return try {
            val samples = readPcm(f)
            val meta = File(dir, k + META)
            val created = if (meta.isFile) meta.readText().split("|").getOrNull(2)?.toLongOrNull() ?: 0L else 0L
            val e = Entry(f, samples, SR, samples.size.toFloat() / SR, created)
            remember(k, e)
            e
        } catch (_: Exception) {
            f.delete()
            null
        }
    }

    fun save(material: Material, variant: Int, samples: FloatArray) {
        val k = key(material, variant)
        try {
            val f = File(dir, k + EXT)
            writePcm(f, samples)
            File(dir, k + META).writeText("neural|48k|${System.currentTimeMillis()}")
            remember(k, Entry(f, samples, SR, samples.size.toFloat() / SR, System.currentTimeMillis()))
        } catch (_: Exception) {
            // Disk full or IO error — cache simply stays absent, procedural keeps playing.
        }
    }

    /** Drops the in-RAM copies of the OLDEST entries until 75% of the budget is free.
     *  On-disk files stay — only the RAM copy is released, so a later [find]
     *  transparently re-reads the loop from disk. */
    private fun remember(k: String, e: Entry) {
        mem[k]?.let { old -> memBytes -= (old.samples?.size ?: 0) * 4L }
        mem[k] = e
        memBytes += (e.samples?.size ?: 0) * 4L
        if (memBytes > MAX_MEM_BYTES) {
            val byAge = mem.entries
                .filter { it.value.samples != null }
                .sortedBy { it.value.createdMs }
                .iterator()
            while (memBytes > MAX_MEM_BYTES * 3 / 4 && byAge.hasNext()) {
                val en = byAge.next()
                val bytes = (en.value.samples?.size ?: 0) * 4L
                en.value.samples = null
                memBytes -= bytes
            }
        }
    }

    private fun readPcm(f: File): FloatArray {
        val size = (f.length() / 4L).toInt()
        val out = FloatArray(size)
        DataInputStream(f.inputStream().buffered()).use { d ->
            for (i in 0 until size) out[i] = java.lang.Float.intBitsToFloat(d.readInt())
        }
        return out
    }

    private fun writePcm(f: File, samples: FloatArray) {
        val tmp = File(f.parentFile, f.name + ".tmp")
        DataOutputStream(tmp.outputStream().buffered()).use { d ->
            for (v in samples) d.writeInt(java.lang.Float.floatToIntBits(v))
        }
        if (!tmp.renameTo(f)) {
            f.delete()
            tmp.renameTo(f)
        }
    }
}
