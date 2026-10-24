package com.echolens.app.audio.neural

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.StatFs
import java.io.File

/**
 * Locates on-device neural texture model files and checks device capability.
 *
 * Since v1.3.0 the Stable Audio Open Small LiteRT models are BUNDLED inside
 * the APK as assets (assets/models/, weight-only int8 conditioner + int8 DiT
 * + fp16 VAE, ~850 MB total). On first use they are unpacked into the
 * app-external models directory:
 *
 *   /sdcard/Android/data/com.echolens.app/files/models/
 *
 * Manually provisioned files (or a previous install) keep working: existing
 * files in the models dir take precedence and are never overwritten.
 *
 * Expected files (Arm ML-examples "audiogen" naming, Apache-2.0 pipeline):
 *   conditioners_float32.tflite   — T5 text conditioner, weight-only int8
 *   dit_model.tflite              — diffusion transformer, dynamic int8
 *   autoencoder_model.tflite      — VAE decoder, fp16
 *
 * Optional:
 *   rave_WOOD.onnx, rave_METAL.onnx ... — per-material RAVE models (nn~ ONNX
 *   export), used as a cheaper real-time-ish texture backend.
 *   minc23_mobilenet_v3_int8.tflite — optional material classifier (vision).
 */
object NeuralModels {

    const val FILE_CONDITIONERS = "conditioners_float32.tflite"
    const val FILE_DIT = "dit_model.tflite"
    const val FILE_VAE = "autoencoder_model.tflite"

    private const val ASSET_DIR = "models"
    private val BUNDLED = mapOf(
        FILE_CONDITIONERS to 185_798_608L,
        FILE_DIT to 344_293_232L,
        FILE_VAE to 312_588_244L
    )

    /** Minimum total device RAM (bytes) for on-device diffusion to be sane. */
    private const val MIN_TOTAL_RAM = 6L * 1024 * 1024 * 1024

    class Status(
        val modelsDir: File?,
        val conditioners: File?,
        val dit: File?,
        val vae: File?,
        val totalRamBytes: Long,
        val lowRamDevice: Boolean,
        val raveMaterials: List<String>,
        val bundledInApk: Boolean = false
    ) {
        val complete: Boolean
            get() = conditioners != null && dit != null && vae != null

        val deviceCapable: Boolean
            get() = !lowRamDevice && totalRamBytes >= MIN_TOTAL_RAM

        val available: Boolean
            get() = complete && deviceCapable

        /** True when the full model set ships inside this APK as assets. */

        fun raveFor(material: String): File? =
            modelsDir?.let { d ->
                val f = File(d, "rave_$material.onnx")
                if (f.isFile && f.length() > 0) f else null
            }
    }

    fun modelsDir(context: Context): File {
        val ext = context.getExternalFilesDir(null)?.let { File(it, "models") }
        if (ext != null && ext.isDirectory) return ext
        val internal = File(context.filesDir, "models")
        if (internal.isDirectory) return internal
        return ext ?: internal
    }

    fun status(context: Context): Status {
        val dir = modelsDir(context)
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am?.getMemoryInfo(memInfo)
        val rave = MATERIAL_NAMES.mapNotNull { m ->
            val f = File(dir, "rave_$m.onnx")
            if (f.isFile) m else null
        }
        return Status(
            modelsDir = dir,
            conditioners = find(dir, FILE_CONDITIONERS),
            dit = find(dir, FILE_DIT),
            vae = find(dir, FILE_VAE),
            totalRamBytes = memInfo.totalMem,
            lowRamDevice = am?.isLowRamDevice ?: false || Build.VERSION.SDK_INT < 26,
            raveMaterials = rave,
            bundledInApk = hasBundledModels(context)
        )
    }

    /** True when this APK carries the full model set in assets/models/. */
    fun hasBundledModels(context: Context): Boolean = try {
        val names = context.assets.list(ASSET_DIR)?.toSet() ?: emptySet()
        BUNDLED.keys.all { it in names }
    } catch (_: Exception) {
        false
    }

    /**
     * Copies the bundled .tflite models from APK assets into the models dir.
     * Existing valid files are never overwritten. Blocking I/O (~850 MB) —
     * call only from a background thread.
     *
     * @param onProgress called per file with a short human-readable line
     * @return null on success, otherwise a short failure reason
     */
    fun unpackBundledModels(context: Context, onProgress: (String) -> Unit = {}): String? {
        val dir = modelsDir(context)
        if (!dir.isDirectory && !dir.mkdirs()) return "не удалось создать ${dir.path}"
        val missing = BUNDLED.filter { (name, _) -> find(dir, name) == null }.keys.toList()
        if (missing.isEmpty()) return null

        val need = missing.sumOf { BUNDLED[it] ?: 0L }
        val free = try {
            StatFs(dir.path).availableBytes
        } catch (_: Exception) {
            Long.MAX_VALUE
        }
        if (free < need + 64_000_000L) {
            return "мало места на диске: нужно ~${need / 1_000_000} МБ, свободно ${free / 1_000_000} МБ"
        }

        for (name in missing) {
            onProgress("распаковка $name (${BUNDLED[name]!! / 1_000_000} МБ)")
            val tmp = File(dir, "$name.tmp")
            try {
                context.assets.open("$ASSET_DIR/$name").use { inp ->
                    tmp.outputStream().use { out -> inp.copyTo(out, 262_144) }
                }
                val dst = File(dir, name)
                if (!tmp.renameTo(dst)) {
                    tmp.copyTo(dst, overwrite = true)
                    tmp.delete()
                }
            } catch (t: Throwable) {
                tmp.delete()
                return "ошибка распаковки $name: ${t.message}"
            }
        }
        return null
    }

    fun freeCacheDiskBytes(context: Context): Long {
        val dir = context.filesDir
        return try {
            StatFs(dir.path).availableBytes
        } catch (_: Exception) {
            Long.MAX_VALUE
        }
    }

    private fun find(dir: File, name: String): File? {
        val f = File(dir, name)
        return if (f.isFile && f.length() > 1_000_000L) f else null
    }

    /** The 15 material names used by the sound engine (same order as core.Material). */
    val MATERIAL_NAMES = listOf(
        "WOOD", "METAL", "GLASS", "CERAMIC", "ENGINE", "WATER", "ELECTRONICS",
        "FABRIC", "PLANT", "PAPER", "PERSON", "ANIMAL", "BIRD", "FOOD", "NEUTRAL"
    )
}
