package com.echolens.app.audio.neural

import android.content.Context
import android.os.Process
import com.echolens.app.core.Material
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Background provisioner: turns "the engine wants a neural texture for
 * material X" into an actual on-device generation job and stores the result
 * in [TextureCache].
 *
 * Backend priority per material:
 *   1. RAVE ONNX model (rave_X.onnx) — fast, runs almost real-time
 *   2. Stable Audio Open Small LiteRT pipeline (~15-40 s per texture)
 *
 * While a texture is being generated the audio engine keeps playing the
 * procedural fallback and crossfades to the neural loop as soon as the
 * listener fires. Nothing is ever downloaded — everything is generated
 * on this phone and cached.
 *
 * Lifecycle: the singleton worker thread lives for the whole process and is
 * intentionally NOT stopped between sessions — idle polling costs one 400 ms
 * wake-up while a session is off, and re-creating the thread (plus ONNX
 * sessions) per session would be far more expensive. [shutdownProvisioner]
 * exists for tests and process teardown, not for session stop.
 */
class NeuralTextureProvisioner private constructor(
    private val context: Context,
    private val cache: TextureCache
) : Thread("EchoLensNeural") {

    interface Listener {
        /** Called on the neural worker thread when a texture becomes available. */
        fun onTextureReady(material: Material, variant: Int)
    }

    private class Job(val material: Material, val variant: Int, val seed: Long)

    private val queue = ConcurrentLinkedQueue<Job>()
    private val inFlight = java.util.Collections.newSetFromMap(java.util.concurrent.ConcurrentHashMap<Job, Boolean>())
    private val running = AtomicBoolean(false)
    private val cooldownMs = 90_000L
    private val lastFail = HashMap<String, Long>()

    @Volatile var listener: Listener? = null
    @Volatile var enabled = false
    @Volatile var status: String = ""
        private set

    private var engine: NeuralTextureEngine? = null
    private var raveCache = HashMap<String, RaveTimbre>()

    companion object {
        @Volatile private var instance: NeuralTextureProvisioner? = null

        fun get(context: Context, cache: TextureCache): NeuralTextureProvisioner {
            instance?.let { return it }
            synchronized(this) {
                instance?.let { return it }
                val p = NeuralTextureProvisioner(context.applicationContext, cache)
                p.start()
                instance = p
                return p
            }
        }
    }

    init {
        running.set(true)
    }

    /** Called from the audio thread — must be non-blocking. */
    fun request(material: Material, variant: Int, seed: Long) {
        if (!enabled) return
        val job = Job(material, variant, seed)
        if (inFlight.contains(job)) return
        val failedAt = synchronized(lastFail) { lastFail[job.key()] }
        if (failedAt != null && System.currentTimeMillis() - failedAt < cooldownMs) return
        if (cache.find(material, variant) != null) return
        inFlight.add(job)
        queue.add(job)
    }

    /** Drops pending work (e.g. session stopped). */
    fun clearPending() {
        queue.clear()
        inFlight.clear()
    }

    /** Frees the LiteRT interpreters (GBs of RAM) — call when the session stops. */
    fun releaseEngine() {
        synchronized(this) {
            engine?.close()
            engine = null
        }
    }

    fun shutdownProvisioner() {
        running.set(false)
        queue.clear()
        interrupt()
        synchronized(this) {
            engine?.close()
            engine = null
            raveCache.values.forEach { runCatching { it.close() } }
            raveCache.clear()
        }
        synchronized(Companion) { if (instance === this) instance = null }
    }

    private fun Job.key() = "${material.name}_v$variant"

    override fun run() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_LESS_FAVORABLE)
        while (running.get()) {
            val job = queue.poll()
            if (job == null) {
                try {
                    sleep(400)
                } catch (_: InterruptedException) {
                    break
                }
                continue
            }
            try {
                process(job)
            } finally {
                inFlight.remove(job)
            }
        }
    }

    private fun process(job: Job) {
        if (!enabled || running.get().not()) return
        if (cache.find(job.material, job.variant) != null) return

        var status0 = NeuralModels.status(context)

        // ---- First use: unpack models bundled in the APK (v1.3.0) --------
        if (!status0.available && status0.deviceCapable && status0.bundledInApk) {
            status = "первый запуск: распаковка встроенных моделей (~850 МБ)…"
            val err = NeuralModels.unpackBundledModels(context) { line -> status = line }
            if (err != null) {
                status = "нейро-текстуры: $err"
                markFail(job)
                return
            }
            status0 = NeuralModels.status(context)
        }

        // ---- Backend 1: RAVE (per-material, cheap) --------------------------
        val raveFile = status0.raveFor(job.material.name)
        if (raveFile != null) {
            status = "RAVE: ${job.material.name} v${job.variant}"
            val rave = synchronized(this) {
                raveCache[job.material.name] ?: run {
                    try {
                        RaveTimbre(ai.onnxruntime.OrtEnvironment.getEnvironment(), raveFile)
                            .also { raveCache[job.material.name] = it }
                    } catch (t: Throwable) {
                        android.util.Log.w("NeuralProvisioner", "RAVE load failed", t)
                        null
                    }
                }
            }
            if (rave != null) {
                val samples = rave.generate(job.seed + 17)
                if (samples != null) {
                    cache.save(job.material, job.variant, samples)
                    listener?.onTextureReady(job.material, job.variant)
                    status = "RAVE готов: ${job.material.name}"
                    return
                }
            }
        }

        // ---- Backend 2: Stable Audio Open Small (LiteRT) ---------------------
        if (!status0.available) {
            status = "модели не найдены"
            markFail(job)
            return
        }
        val eng = synchronized(this) {
            engine ?: NeuralTextureEngine.tryCreate(status0).also { engine = it }
        }
        if (eng == null) {
            status = "не удалось загрузить модели"
            markFail(job)
            return
        }
        val prompt = MaterialPrompts.prompt(job.material.name, job.variant)
        status = "генерация ${job.material.name}: 0/${NeuralTextureEngine.DEFAULT_STEPS}"
        val samples = eng.generate(
            promptIds = prompt.ids,
            seed = job.seed,
            steps = NeuralTextureEngine.DEFAULT_STEPS
        ) { step, total ->
            status = "генерация ${job.material.name}: $step/$total"
        }
        if (samples != null && samples.size > 48000) {
            cache.save(job.material, job.variant, samples)
            status = "готово: ${job.material.name}"
            listener?.onTextureReady(job.material, job.variant)
        } else {
            status = "сбой генерации: ${job.material.name}"
            markFail(job)
        }
    }

    private fun markFail(job: Job) {
        synchronized(lastFail) {
            if (lastFail.size > 64) lastFail.clear()
            lastFail[job.key()] = System.currentTimeMillis()
        }
    }
}
