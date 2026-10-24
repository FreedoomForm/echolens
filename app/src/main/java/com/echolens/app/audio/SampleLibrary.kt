package com.echolens.app.audio

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import com.echolens.app.core.Material
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Bundled sound texture library (assets/sounds/, see tools/render_sound_library.py).
 *
 * The library ships inside the APK: ~880 seamless loops, one-shot events and
 * object-class signature textures, all synthesized offline at 48 kHz mono and
 * loudness-calibrated per material. The audio engine spatializes each active
 * object with its own loop + pitch variant, so even 15 simultaneous objects
 * sound distinct ("thousands" of distinct voices via variant x pitch x depth).
 *
 * Decoding uses MediaExtractor/MediaCodec directly on asset file descriptors
 * (assets are stored uncompressed via noCompress "ogg"). Samples are decoded
 * asynchronously on a worker pool and cached LRU — the realtime audio thread
 * never blocks on I/O.
 */
class SampleLibrary(private val context: Context) {

    class Entry internal constructor(val path: String) {
        override fun equals(other: Any?) = other is Entry && other.path == path
        override fun hashCode() = path.hashCode()
    }

    private class ClassPack(val material: Material, val loops: List<Entry>)

    private val loopsByMaterial = HashMap<Material, List<Entry>>(16)
    private val eventsByMaterial = HashMap<Material, List<Entry>>(16)
    private val classPacks = HashMap<String, ClassPack>(40)
    private val decoded = ConcurrentHashMap<String, FloatArray>()
    private val decodeQueue = ConcurrentHashMap<String, Boolean>()
    private val decodePool: ExecutorService = Executors.newFixedThreadPool(2) { r ->
        Thread(r, "SampleDecode").apply { priority = Thread.NORM_PRIORITY - 1 }
    }

    /** LRU order of decoded paths (approximate, for eviction bookkeeping). */
    private val lruOrder = java.util.LinkedHashMap<String, Long>(64, 0.75f, true)

    var sampleCount: Int = 0
        private set

    companion object {
        private const val MANIFEST = "sounds/manifest.json"
        private const val TARGET_SR = 48000
        private const val MAX_CACHED = 30 // ~30 x 2 s x 48 kHz x float ≈ 11 MB

        fun isAvailable(context: Context): Boolean = try {
            context.assets.open(MANIFEST).use { it.read() } >= 0
        } catch (_: Exception) {
            false
        }
    }

    init {
        try {
            parseManifest()
        } catch (t: Throwable) {
            android.util.Log.w("SampleLibrary", "manifest parse failed", t)
        }
    }

    private fun parseManifest() {
        val root = context.assets.open(MANIFEST).use { JSONObject(String(it.readBytes())) }
        val loopsObj = root.optJSONObject("loops")
        val eventsObj = root.optJSONObject("events")
        val classObj = root.optJSONObject("class")

        var count = 0
        if (loopsObj != null) {
            for (name in loopsObj.keys()) {
                val m = materialOrNull(name) ?: continue
                val list = loopsObj.getJSONArray(name)
                val entries = ArrayList<Entry>(list.length())
                for (i in 0 until list.length()) entries.add(Entry(list.getString(i)))
                loopsByMaterial[m] = entries
                count += entries.size
            }
        }
        if (eventsObj != null) {
            for (name in eventsObj.keys()) {
                val m = materialOrNull(name) ?: continue
                val list = eventsObj.getJSONArray(name)
                val entries = ArrayList<Entry>(list.length())
                for (i in 0 until list.length()) entries.add(Entry(list.getString(i)))
                eventsByMaterial[m] = entries
                count += entries.size
            }
        }
        if (classObj != null) {
            for (label in classObj.keys()) {
                val obj = classObj.getJSONObject(label)
                val m = materialOrNull(obj.optString("material")) ?: continue
                val list = obj.getJSONArray("loops")
                val entries = ArrayList<Entry>(list.length())
                for (i in 0 until list.length()) entries.add(Entry(list.getString(i)))
                classPacks[label] = ClassPack(m, entries)
                count += entries.size
            }
        }
        sampleCount = count
    }

    private fun materialOrNull(name: String): Material? = try {
        Material.valueOf(name)
    } catch (_: IllegalArgumentException) {
        null
    }

    /** Loop list for an object: class signature pack first, then material bed. */
    fun loopsFor(label: String, material: Material): List<Entry>? {
        classPacks[label]?.let { return it.loops }
        return loopsByMaterial[material]
    }

    fun eventFor(material: Material, seed: Long): Entry? {
        val list = eventsByMaterial[material] ?: return null
        if (list.isEmpty()) return null
        val idx = ((seed % list.size + list.size) % list.size).toInt()
        return list[idx]
    }

    fun hasMaterial(material: Material): Boolean = loopsByMaterial.containsKey(material)

    /** Non-blocking: returns decoded PCM if ready, otherwise queues a background decode. */
    fun getOrRequest(entry: Entry): FloatArray? {
        decoded[entry.path]?.let {
            touch(entry.path)
            return it
        }
        requestDecode(entry)
        return null
    }

    fun requestDecode(entry: Entry) {
        if (decoded.containsKey(entry.path)) return
        if (decodeQueue.putIfAbsent(entry.path, true) != null) return
        decodePool.execute {
            try {
                val pcm = decodeAsset(entry.path)
                if (pcm != null) {
                    decoded[entry.path] = pcm
                    touch(entry.path)
                    evictIfNeeded()
                }
            } catch (t: Throwable) {
                android.util.Log.w("SampleLibrary", "decode failed: ${entry.path}", t)
            } finally {
                decodeQueue.remove(entry.path)
            }
        }
    }

    /** Preload events for materials that are likely to appear soon. */
    fun preloadEvents(materials: Collection<Material>) {
        for (m in materials) {
            eventsByMaterial[m]?.firstOrNull()?.let { requestDecode(it) }
        }
    }

    @Synchronized
    private fun touch(path: String) {
        lruOrder[path] = System.nanoTime()
    }

    @Synchronized
    private fun evictIfNeeded() {
        while (lruOrder.size > MAX_CACHED) {
            var oldestPath: String? = null
            var oldest = Long.MAX_VALUE
            for ((p, t) in lruOrder) {
                if (t < oldest) {
                    oldest = t
                    oldestPath = p
                }
            }
            if (oldestPath == null) break
            lruOrder.remove(oldestPath)
            decoded.remove(oldestPath)
        }
    }

    fun close() {
        decodePool.shutdownNow()
        decoded.clear()
        lruOrder.clear()
    }

    // -------------------------------------------------------------- decode

    /** Decodes an asset OGG/PCM file to 48 kHz mono float. */
    private fun decodeAsset(path: String): FloatArray? {
        val afd = context.assets.openFd(path)
        val mex = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            mex.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
            var trackIdx = -1
            var format: MediaFormat? = null
            for (i in 0 until mex.trackCount) {
                val f = mex.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) {
                    trackIdx = i
                    format = f
                    break
                }
            }
            if (trackIdx < 0 || format == null) return null
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            mex.selectTrack(trackIdx)
            val srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            val pcm = ArrayList<FloatArray>(8)
            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            var lastRate = srcRate
            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val size = mex.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, size, mex.sampleTime, 0)
                            mex.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIdx >= 0) {
                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    val fmt = codec.getOutputFormat(outIdx)
                    lastRate = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    val outCh = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    val enc = fmt.getInteger(
                        MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT
                    )
                    val chunk: FloatArray = if (enc == AudioFormat.ENCODING_PCM_FLOAT) {
                        val fb = outBuf.order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer()
                        val n = fb.remaining() / outCh
                        FloatArray(n).also { dst ->
                            for (i in 0 until n) {
                                var acc = 0f
                                for (c in 0 until outCh) acc += fb.get()
                                dst[i] = acc / outCh
                            }
                        }
                    } else {
                        val sb = outBuf.order(java.nio.ByteOrder.nativeOrder()).asShortBuffer()
                        val n = sb.remaining() / outCh
                        FloatArray(n).also { dst ->
                            for (i in 0 until n) {
                                var acc = 0
                                for (c in 0 until outCh) acc += sb.get().toInt()
                                dst[i] = (acc / outCh) / 32768f
                            }
                        }
                    }
                    if (chunk.isNotEmpty()) pcm.add(chunk)
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                } else if (outIdx == MediaCodec.INFO_TRY_AGAIN_LATER && inputDone) {
                    // Drain a few more times before giving up.
                    var drained = false
                    for (k in 0 until 10) {
                        val idx2 = codec.dequeueOutputBuffer(info, 10_000)
                        if (idx2 >= 0) {
                            codec.releaseOutputBuffer(idx2, false)
                            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) drained = true
                        }
                    }
                    if (drained) outputDone = true else break
                }
            }

            var flat = FloatArray(pcm.sumOf { it.size })
            var off = 0
            for (c in pcm) {
                System.arraycopy(c, 0, flat, off, c.size)
                off += c.size
            }
            flat = resampleTo48k(flat, lastRate, channels)
            return if (flat.size > TARGET_SR / 10) flat else null
        } finally {
            try {
                codec?.stop()
            } catch (_: Exception) {
            }
            try {
                codec?.release()
            } catch (_: Exception) {
            }
            mex.release()
            runCatching { afd.close() }
        }
    }

    private fun resampleTo48k(x: FloatArray, srcRate: Int, @Suppress("UNUSED_PARAMETER") channels: Int): FloatArray {
        if (srcRate == TARGET_SR || x.size < 16) return x
        val outLen = (x.size.toLong() * TARGET_SR / srcRate).toInt()
        if (outLen < 16) return x
        val out = FloatArray(outLen)
        val ratio = srcRate.toDouble() / TARGET_SR
        var pos = 0.0
        for (i in 0 until outLen) {
            val i0 = pos.toInt()
            val i1 = (i0 + 1).coerceAtMost(x.size - 1)
            val f = pos - i0
            out[i] = (x[i0] * (1 - f) + x[i1] * f).toFloat()
            pos += ratio
        }
        return out
    }
}
