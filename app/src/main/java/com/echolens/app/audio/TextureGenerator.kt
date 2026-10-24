package com.echolens.app.audio

import com.echolens.app.core.Material
import com.echolens.app.core.MaterialSpec
import java.util.Random
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin

/**
 * Procedural sound-texture synthesizer for one object stream.
 *
 * Every detected object class maps to a "material" whose characteristic
 * sound — wooden knocks, metallic ringing, glass clinks, engine rumble,
 * water, rustling, birds — is synthesized from scratch, so the APK needs
 * no audio files at all.
 *
 * The sound gets louder (and slightly denser) as the object approaches,
 * and a distance low-pass filter models air absorption.
 */
class TextureGenerator(
    private val spec: MaterialSpec,
    private val sampleRate: Int,
    seed: Long,
    /**
     * Phone speakers roll off below ~350-400 Hz, so in speaker mode all
     * textures are shifted into the 200 Hz+ range and their "body" is
     * carried by harmonics the speaker can actually reproduce.
     */
    speakerMode: Boolean = true
) {
    private val invSr = 1f / sampleRate
    private val rng = Random(seed)

    /** Time until the next strike/burst event, in samples. */
    private var timeToEvent = (0.25f + rng.nextFloat() * 0.6f) * sampleRate

    // Frequency/tonal adaptation
    private val fMul: Float
    private val bedCoefEngine: Float
    private val bedCoefPerson: Float
    private val bedCoefWater: Float
    private val bedCoefFabric: Float
    private val bedCoefPlant: Float
    private val bedCoefFood: Float
    private val neutralFreq: Float

    // Continuous "bed" state
    private var phase = 0f
    private var phase2 = 0f
    private var lfo1 = 0f
    private var lfo2 = 0f
    private var bedLp = 0f
    private var bedHp = 0f
    private val f0: Float

    // Output chain
    private var distLp = 0f
    private var gainSmooth = 0f
    private val outTrim: Float

    private class Voice {
        var active = false
        var age = 0f
        var maxAge = 1f
        var partials = 0
        val freqs = FloatArray(5)
        val amps = FloatArray(5)
        val envs = FloatArray(5)
        val envCoefs = FloatArray(5)
        val phases = FloatArray(5)
        var noiseAmp = 0f
        var noiseEnv = 0f
        var noiseEnvCoef = 1f
        var noiseState = 0f
        var noiseLpCoef = 0.4f
        var chirp = false
        var chirpF0 = 0f
        var chirpF1 = 0f
        var chirpDur = 0.05f
        var chirpPhase = 0f
    }

    private val voices = Array(5) { Voice() }
    private var voiceCursor = 0

    init {
        outTrim = when (spec.material) {
            Material.WOOD -> 0.50f
            Material.METAL -> 0.34f
            Material.GLASS -> 0.30f
            Material.CERAMIC -> 0.38f
            Material.ENGINE -> 0.55f
            Material.WATER -> 0.42f
            Material.ELECTRONICS -> 0.30f
            Material.FABRIC -> 0.42f
            Material.PLANT -> 0.36f
            Material.PAPER -> 0.46f
            Material.PERSON -> 0.30f
            Material.ANIMAL -> 0.42f
            Material.BIRD -> 0.38f
            Material.FOOD -> 0.36f
            Material.NEUTRAL -> 0.30f
        }
        f0 = when (spec.material) {
            Material.ENGINE -> if (speakerMode) 105f + rng.nextFloat() * 45f else 44f + rng.nextFloat() * 36f
            Material.PERSON -> if (speakerMode) 190f + rng.nextFloat() * 50f else 150f + rng.nextFloat() * 40f
            else -> 0f
        }
        if (speakerMode) {
            fMul = 1.35f
            bedCoefEngine = 0.045f
            bedCoefPerson = 0.05f
            bedCoefWater = 0.13f
            bedCoefFabric = 0.09f
            bedCoefPlant = 0.14f
            bedCoefFood = 0.075f
            neutralFreq = 440f
        } else {
            fMul = 1f
            bedCoefEngine = 0.013f
            bedCoefPerson = 0.02f
            bedCoefWater = 0.09f
            bedCoefFabric = 0.058f
            bedCoefPlant = 0.11f
            bedCoefFood = 0.038f
            neutralFreq = 340f
        }
    }

    /** Renders one block. [gain] combines cone and distance attenuation (0..~1.7). */
    fun process(out: FloatArray, frames: Int, distanceM: Float, gain: Float) {
        val m = spec.material
        val prox = 0.55f + 0.9f * (spec.refDistanceM / (spec.refDistanceM + distanceM))

        for (i in 0 until frames) {
            var s = 0f

            // ---------- continuous bed ----------
            when (m) {
                Material.ENGINE -> {
                    phase += f0 * invSr
                    lfo1 += 0.45f * invSr
                    val am = 1f + 0.3f * sin(TWO_PI * lfo1)
                    s = (sin(TWO_PI * phase) +
                        0.55f * sin(TWO_PI * phase * 2f + 0.3f) +
                        0.34f * sin(TWO_PI * phase * 3f + 0.9f) +
                        0.18f * sin(TWO_PI * phase * 4f) +
                        0.10f * sin(TWO_PI * phase * 5f)) * 0.20f * am
                    val w = white()
                    bedLp += bedCoefEngine * (w - bedLp)
                    s += bedLp * 2.2f
                }
                Material.PERSON -> {
                    lfo1 += 0.24f * invSr
                    val breath = 0.55f + 0.45f * sin(TWO_PI * lfo1)
                    phase += f0 * invSr
                    phase2 += f0 * 1.5f * invSr
                    s = (sin(TWO_PI * phase) + 0.5f * sin(TWO_PI * phase2 + 0.7f)) * 0.35f * breath
                    val w = white()
                    bedLp += bedCoefPerson * (w - bedLp)
                    s += bedLp * 0.4f
                }
                Material.WATER -> {
                    lfo1 += 0.3f * invSr
                    val w = white()
                    bedLp += bedCoefWater * (w - bedLp)
                    s = bedLp * 0.55f * (0.7f + 0.3f * sin(TWO_PI * lfo1))
                }
                Material.ELECTRONICS -> {
                    phase += 100f * invSr
                    s = (0.4f * sin(TWO_PI * phase) + 0.15f * sin(TWO_PI * phase * 2f + 0.2f)) * 0.8f
                    val w = white()
                    bedHp += 0.55f * (w - bedHp)
                    s += (w - bedHp) * 0.05f
                }
                Material.FABRIC -> {
                    lfo1 += 2.2f * invSr
                    lfo2 += 3.7f * invSr
                    val am = 0.45f + 0.3f * sin(TWO_PI * lfo1) + 0.25f * sin(TWO_PI * lfo2)
                    val w = white()
                    bedLp += bedCoefFabric * (w - bedLp)
                    s = bedLp * 1.1f * am
                }
                Material.PLANT -> {
                    lfo1 += 1.3f * invSr
                    val am = 0.6f + 0.4f * (0.5f + 0.5f * sin(TWO_PI * lfo1))
                    val w = white()
                    bedLp += bedCoefPlant * (w - bedLp)
                    s = bedLp * 0.7f * am
                }
                Material.FOOD -> {
                    lfo1 += 0.7f * invSr
                    val w = white()
                    bedLp += bedCoefFood * (w - bedLp)
                    s = bedLp * 0.8f * (0.6f + 0.4f * sin(TWO_PI * lfo1))
                }
                Material.NEUTRAL -> {
                    phase += neutralFreq * invSr
                    lfo1 += 0.5f * invSr
                    s = sin(TWO_PI * phase) * 0.4f * (0.6f + 0.4f * sin(TWO_PI * lfo1))
                    val w = white()
                    bedLp += 0.02f * (w - bedLp)
                    s += bedLp * 0.15f
                }
                else -> {}
            }

            // ---------- events (strikes, chirps, crackles) ----------
            timeToEvent -= 1f
            if (timeToEvent <= 0f) {
                spawnEvent(m, prox)
                timeToEvent += nextInterval(m, prox)
            }
            for (v in voices) {
                if (v.active) s += voiceSample(v)
            }

            out[i] = s
        }

        // ---------- distance low-pass + dezippered gain ----------
        val fc = 12000f / (1f + distanceM * distanceM * 0.16f)
        val a = (1f - exp(-TWO_PI * fc * invSr)).coerceIn(0.001f, 1f)
        for (i in 0 until frames) {
            distLp += a * (out[i] - distLp)
            gainSmooth += 0.0045f * (gain - gainSmooth)
            out[i] = distLp * outTrim * gainSmooth
        }
    }

    private fun white(): Float = rng.nextFloat() * 2f - 1f

    private fun nextInterval(m: Material, prox: Float): Float {
        val rate = when (m) {
            Material.WOOD -> 1.4f * prox
            Material.METAL -> 0.35f * prox
            Material.GLASS -> 0.5f * prox
            Material.CERAMIC -> 0.55f * prox
            Material.PAPER -> 2.2f * prox
            Material.WATER -> 0.7f * prox
            Material.PLANT -> 0.25f * prox
            Material.FOOD -> 0.15f * prox
            Material.ELECTRONICS -> 0.12f * prox
            Material.ANIMAL -> 0.22f * prox
            Material.BIRD -> 0.3f * prox
            else -> 0f
        }
        if (rate <= 0f) return 1e9f
        val u = 0.05f + rng.nextFloat() * 0.95f
        val intervalSec = (-ln(u) / rate).coerceIn(0.12f, 20f)
        return intervalSec * sampleRate
    }

    private fun spawnEvent(m: Material, @Suppress("UNUSED_PARAMETER") prox: Float) {
        when (m) {
            Material.WOOD ->
                strike(
                    (150f + rng.nextFloat() * 90f) * fMul,
                    RATIOS_WOOD, AMPS_3, DECAYS_WOOD, 0.35f, 140f
                )
            Material.METAL -> {
                val f = (380f + rng.nextFloat() * 340f) * fMul
                strike(f, RATIOS_METAL, AMPS_4, DECAYS_METAL, 0.12f, 90f)
                if (rng.nextFloat() < 0.35f) {
                    strike(f, RATIOS_METAL, AMPS_4, DECAYS_METAL, 0.12f, 90f, ageOffset = -0.16f)
                }
            }
            Material.GLASS ->
                strike(
                    (1250f + rng.nextFloat() * 1150f) * fMul,
                    RATIOS_GLASS, AMPS_3, DECAYS_GLASS, 0.2f, 300f
                )
            Material.CERAMIC ->
                strike(
                    (680f + rng.nextFloat() * 340f) * fMul,
                    RATIOS_CERAMIC, AMPS_3, DECAYS_CERAMIC, 0.25f, 250f
                )
            Material.PAPER -> noiseBurst(0.5f, 380f, 0.02f, 0.5f)
            Material.WATER ->
                chirpVoice(
                    300f + rng.nextFloat() * 180f,
                    850f + rng.nextFloat() * 250f,
                    0.045f + rng.nextFloat() * 0.02f, 0.4f
                )
            Material.PLANT -> noiseBurst(0.45f, 22f, 0.16f, 0.25f)
            Material.FOOD -> chirpVoice(190f, 55f, 0.09f, 0.5f)
            Material.ELECTRONICS -> chirpVoice(1150f, 1150f, 0.04f, 0.25f)
            Material.ANIMAL -> {
                val f0a = 740f + rng.nextFloat() * 60f
                val f1a = 980f + rng.nextFloat() * 80f
                chirpVoice(f0a, f1a, 0.09f, 0.5f)
                chirpVoice(f0a, f1a, 0.09f, 0.45f, ageOffset = -0.22f)
            }
            Material.BIRD -> {
                val n = 3 + (rng.nextFloat() * 3f).toInt()
                var age = 0f
                for (k in 0 until n) {
                    val f = 2200f + rng.nextFloat() * 1600f
                    chirpVoice(f, f * 1.25f, 0.06f, 0.4f, ageOffset = age)
                    age -= 0.07f + rng.nextFloat() * 0.05f
                }
            }
            else -> {}
        }
    }

    private fun freeVoice(): Voice {
        for (v in voices) if (!v.active) return v
        voiceCursor = (voiceCursor + 1) % voices.size
        return voices[voiceCursor]
    }

    /** A struck object: a few inharmonic exponentially-decaying partials + transient noise. */
    private fun strike(
        baseFreq: Float,
        ratios: FloatArray,
        amps: FloatArray,
        decays: FloatArray,
        noiseAmp: Float,
        noiseDecay: Float,
        ageOffset: Float = 0f
    ): Voice {
        val v = freeVoice()
        v.active = true
        v.age = ageOffset
        v.partials = ratios.size.coerceAtMost(5)
        var deepest = decays[0]
        for (p in 0 until v.partials) {
            val f = baseFreq * ratios[p] * (1f + (rng.nextFloat() - 0.5f) * 0.02f)
            v.freqs[p] = f
            v.amps[p] = amps[p]
            v.envs[p] = 1f
            v.envCoefs[p] = exp(-decays[p] * invSr)
            v.phases[p] = rng.nextFloat()
            if (decays[p] < deepest) deepest = decays[p]
        }
        v.noiseAmp = noiseAmp
        v.noiseEnv = 1f
        v.noiseEnvCoef = exp(-noiseDecay * invSr)
        v.noiseState = 0f
        v.noiseLpCoef = 0.4f
        v.chirp = false
        v.maxAge = (5.5f / deepest).coerceAtMost(3.5f)
        return v
    }

    /** A short pitched sweep (water bubble, animal call, electronics blip). */
    private fun chirpVoice(f0: Float, f1: Float, dur: Float, amp: Float, ageOffset: Float = 0f): Voice {
        val v = freeVoice()
        v.active = true
        v.age = ageOffset
        v.partials = 0
        v.amps[0] = amp
        v.noiseAmp = 0f
        v.chirp = true
        v.chirpF0 = f0
        v.chirpF1 = f1
        v.chirpDur = dur
        v.chirpPhase = 0f
        v.maxAge = dur + 0.05f
        return v
    }

    /** A short filtered noise burst (paper crackle, plant rustle). */
    private fun noiseBurst(amp: Float, decay: Float, dur: Float, lpCoef: Float): Voice {
        val v = freeVoice()
        v.active = true
        v.age = 0f
        v.partials = 0
        v.noiseAmp = amp
        v.noiseEnv = 1f
        v.noiseEnvCoef = exp(-decay * invSr)
        v.noiseState = 0f
        v.noiseLpCoef = lpCoef
        v.chirp = false
        v.maxAge = dur
        return v
    }

    private fun voiceSample(v: Voice): Float {
        v.age += invSr
        if (v.age < 0f) return 0f // scheduled in the future
        var s = 0f
        var p = 0
        while (p < v.partials) {
            v.envs[p] *= v.envCoefs[p]
            v.phases[p] += v.freqs[p] * invSr
            if (v.phases[p] >= 1f) v.phases[p] -= 1f
            s += v.amps[p] * v.envs[p] * sin(TWO_PI * v.phases[p])
            p++
        }
        if (v.noiseAmp > 0f) {
            val w = white()
            v.noiseState += v.noiseLpCoef * (w - v.noiseState)
            v.noiseEnv *= v.noiseEnvCoef
            s += v.noiseAmp * v.noiseEnv * (w * 0.6f + v.noiseState * 0.6f)
        }
        if (v.chirp) {
            val u = v.age / v.chirpDur
            if (u >= 1f) {
                v.active = false
            } else {
                val f = v.chirpF0 + (v.chirpF1 - v.chirpF0) * u
                v.chirpPhase += f * invSr
                s += v.amps[0] * sin(TWO_PI * v.chirpPhase) * sin(PI_F * u)
            }
        }
        if (v.age > v.maxAge) v.active = false
        return s
    }

    private companion object {
        const val TWO_PI = 6.2831855f
        const val PI_F = 3.1415927f

        val RATIOS_WOOD = floatArrayOf(1f, 2.83f, 5.4f)
        val RATIOS_GLASS = floatArrayOf(1f, 1.71f, 2.98f)
        val RATIOS_CERAMIC = floatArrayOf(1f, 2.1f, 3.9f)
        val RATIOS_METAL = floatArrayOf(1f, 2.71f, 4.95f, 7.2f)

        val AMPS_3 = floatArrayOf(1f, 0.45f, 0.22f)
        val AMPS_4 = floatArrayOf(1f, 0.5f, 0.32f, 0.18f)

        val DECAYS_WOOD = floatArrayOf(16f, 26f, 36f)
        val DECAYS_GLASS = floatArrayOf(9f, 12f, 16f)
        val DECAYS_CERAMIC = floatArrayOf(6f, 9f, 13f)
        val DECAYS_METAL = floatArrayOf(2.5f, 3.2f, 4.6f, 6.5f)
    }
}
