package com.echolens.app.audio.neural

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.StatFs
import java.io.File

/**
 * Locates on-device neural texture model files and checks device capability.
 *
 * The Stable Audio Open Small LiteRT models (~2-3 GB on disk) are NOT shipped
 * inside the APK. They are provisioned by the user (or a companion script)
 * into the app-external models directory:
 *
 *   /sdcard/Android/data/com.echolens.app/files/models/
 *
 * Expected files (Arm ML-examples "audiogen" naming, Apache-2.0 pipeline):
 *   conditioners_float32.tflite   — T5 text conditioner + number conditioner
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

    /** Minimum total device RAM (bytes) for on-device diffusion to be sane. */
    private const val MIN_TOTAL_RAM = 6L * 1024 * 1024 * 1024

    class Status(
        val modelsDir: File?,
        val conditioners: File?,
        val dit: File?,
        val vae: File?,
        val totalRamBytes: Long,
        val lowRamDevice: Boolean,
        val raveMaterials: List<String>
    ) {
        val complete: Boolean
            get() = conditioners != null && dit != null && vae != null

        val deviceCapable: Boolean
            get() = !lowRamDevice && totalRamBytes >= MIN_TOTAL_RAM

        val available: Boolean
            get() = complete && deviceCapable

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
            raveMaterials = rave
        )
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
