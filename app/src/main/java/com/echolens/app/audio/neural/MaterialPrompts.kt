package com.echolens.app.audio.neural

/**
 * Fixed prompt bank for on-device neural texture generation
 * (Stable Audio Open Small, LiteRT).
 *
 * Prompts are pre-tokenized with the t5-base SentencePiece model
 * (EOS id 1 appended), so the phone needs no tokenizer at all.
 * Regenerate with tools/make_prompt_tokens.py.
 *
 * Generated file — do not edit by hand.
 */
object MaterialPrompts {

    class Prompt(
        val text: String,
        val ids: IntArray
    )

    /** [material][variant] */
    private val bank: Map<String, Array<Prompt>> = mapOf(
        "WOOD" to arrayOf(
            Prompt("dry wooden knocks and creaks, furniture wood, close range, soft room reverb, seamless loop", intArrayOf(2192, 5726, 7673, 7, 11, 11983, 157, 7, 6, 1991, 1679, 6, 885, 620, 6, 1835, 562, 3, 60, 11868, 6, 15870, 6494, 1)),
            Prompt("gentle knocking on a wooden table, warm resonance, sparse rhythm, seamless loop", intArrayOf(7569, 7673, 53, 30, 3, 9, 5726, 953, 6, 1978, 28914, 6, 14144, 7, 15, 10638, 6, 15870, 6494, 1)),
        ),
        "METAL" to arrayOf(
            Prompt("metallic rattles and clangs, distant street metal, sparse hits, seamless loop", intArrayOf(18813, 31010, 7, 11, 18297, 122, 7, 6, 10382, 2815, 1946, 6, 14144, 7, 15, 8046, 6, 15870, 6494, 1)),
            Prompt("hollow metal tube resonating with occasional bright strikes, seamless loop", intArrayOf(23471, 1946, 8017, 3, 60, 739, 1014, 28, 14521, 2756, 17613, 6, 15870, 6494, 1)),
        ),
        "GLASS" to arrayOf(
            Prompt("glass clinks and delicate crystal ringing, sparse, close, seamless loop", intArrayOf(1905, 3, 75, 4907, 7, 11, 10396, 6884, 3, 1007, 53, 6, 14144, 7, 15, 6, 885, 6, 15870, 6494, 1)),
            Prompt("wine glass humming softly with occasional touch, shimmering high tones, seamless loop", intArrayOf(2013, 1905, 3, 107, 23434, 1835, 120, 28, 14521, 1586, 6, 27536, 53, 306, 12, 1496, 6, 15870, 6494, 1)),
        ),
        "CERAMIC" to arrayOf(
            Prompt("ceramic taps and porcelain knocks on a table, dry, seamless loop", intArrayOf(8949, 5011, 7, 11, 21806, 7673, 7, 30, 3, 9, 953, 6, 2192, 6, 15870, 6494, 1)),
            Prompt("porcelain bowl touches, soft mid-range resonance, sparse, seamless loop", intArrayOf(21806, 3047, 13945, 6, 1835, 2076, 18, 5517, 28914, 6, 14144, 7, 15, 6, 15870, 6494, 1)),
        ),
        "ENGINE" to arrayOf(
            Prompt("car engine idling steadily, warm low rumble, seamless loop", intArrayOf(443, 1948, 3, 23, 26, 697, 19983, 120, 6, 1978, 731, 3, 52, 13514, 6, 15870, 6494, 1)),
            Prompt("smooth motor hum with subtle vibrations, constant, seamless loop", intArrayOf(3050, 2340, 3, 4884, 28, 9240, 14570, 7, 6, 3917, 6, 15870, 6494, 1)),
        ),
        "WATER" to arrayOf(
            Prompt("water trickling and small bubbles, stream flowing gently, seamless loop", intArrayOf(387, 7873, 697, 11, 422, 11144, 7, 6, 6093, 15401, 10772, 6, 15870, 6494, 1)),
            Prompt("soft water splashes and droplets in a quiet basin, seamless loop", intArrayOf(1835, 387, 16500, 15, 7, 11, 2328, 7677, 16, 3, 9, 4354, 21510, 6, 15870, 6494, 1)),
        ),
        "ELECTRONICS" to arrayOf(
            Prompt("electronic device hum with tiny clicks and high-frequency chirps, seamless loop", intArrayOf(3031, 1407, 3, 4884, 28, 5103, 1214, 7, 11, 306, 18, 30989, 3, 11727, 102, 7, 6, 15870, 6494, 1)),
            Prompt("quiet gadget buzz and short digital blips, sparse, seamless loop", intArrayOf(4354, 13195, 15886, 11, 710, 1125, 3, 7437, 102, 7, 6, 14144, 7, 15, 6, 15870, 6494, 1)),
        ),
        "FABRIC" to arrayOf(
            Prompt("fabric rustling softly, clothes movement, close and airy, seamless loop", intArrayOf(3002, 3, 9277, 697, 1835, 120, 6, 4954, 2426, 6, 885, 11, 799, 63, 6, 15870, 6494, 1)),
            Prompt("soft cloth flapping and brushing sounds, gentle, seamless loop", intArrayOf(1835, 10366, 5731, 5341, 11, 6453, 53, 2993, 6, 7569, 6, 15870, 6494, 1)),
        ),
        "PLANT" to arrayOf(
            Prompt("leaves rustling in a light breeze, dry foliage, seamless loop", intArrayOf(3231, 3, 9277, 697, 16, 3, 9, 659, 15825, 6, 2192, 25840, 6, 15870, 6494, 1)),
            Prompt("plant leaves brushing against each other, organic crackle, seamless loop", intArrayOf(1475, 3231, 6453, 53, 581, 284, 119, 6, 3648, 5261, 109, 6, 15870, 6494, 1)),
        ),
        "PAPER" to arrayOf(
            Prompt("paper crumpling and pages turning, close, dry, seamless loop", intArrayOf(1040, 8396, 1167, 697, 11, 1688, 5074, 6, 885, 6, 2192, 6, 15870, 6494, 1)),
            Prompt("light paper crackle and sheet movements, sparse, seamless loop", intArrayOf(659, 1040, 5261, 109, 11, 4228, 9780, 6, 14144, 7, 15, 6, 15870, 6494, 1)),
        ),
        "PERSON" to arrayOf(
            Prompt("person breathing calmly and shifting nearby, quiet presence, seamless loop", intArrayOf(568, 10882, 4447, 120, 11, 16739, 4676, 6, 4354, 3053, 6, 15870, 6494, 1)),
            Prompt("soft human rustle and steps close by, seamless loop", intArrayOf(1835, 936, 3, 9277, 109, 11, 2245, 885, 57, 6, 15870, 6494, 1)),
        ),
        "ANIMAL" to arrayOf(
            Prompt("dog barking softly in the distance with movement, seamless loop", intArrayOf(1782, 21696, 53, 1835, 120, 16, 8, 2357, 28, 2426, 6, 15870, 6494, 1)),
            Prompt("cat purring gently, warm animal sound, seamless loop", intArrayOf(1712, 3990, 1007, 10772, 6, 1978, 2586, 1345, 6, 15870, 6494, 1)),
        ),
        "BIRD" to arrayOf(
            Prompt("birds chirping in a tree, morning garden, varied calls, seamless loop", intArrayOf(6331, 3, 11727, 2462, 16, 3, 9, 2195, 6, 1379, 2004, 6, 10535, 3088, 6, 15870, 6494, 1)),
            Prompt("small bird song phrases with pauses, natural, seamless loop", intArrayOf(422, 5963, 2324, 15101, 28, 3, 9033, 7, 6, 793, 6, 15870, 6494, 1)),
        ),
        "FOOD" to arrayOf(
            Prompt("food being unwrapped and bitten, crunchy textures, close, seamless loop", intArrayOf(542, 271, 73, 210, 5846, 3138, 11, 3, 26040, 6, 29334, 17951, 6, 885, 6, 15870, 6494, 1)),
            Prompt("soft food crumble and packaging rustle, seamless loop", intArrayOf(1835, 542, 22854, 15, 11, 6745, 3, 9277, 109, 6, 15870, 6494, 1)),
        ),
        "NEUTRAL" to arrayOf(
            Prompt("soft neutral object hum with gentle taps, ambiguous material, seamless loop", intArrayOf(1835, 7163, 3735, 3, 4884, 28, 7569, 5011, 7, 6, 3, 24621, 1162, 1037, 6, 15870, 6494, 1)),
            Prompt("muted taps on an unknown surface, calm, seamless loop", intArrayOf(4035, 1054, 5011, 7, 30, 46, 7752, 1774, 6, 4447, 6, 15870, 6494, 1)),
        ),
    )

    fun prompt(material: String, variant: Int): Prompt {
        val arr = bank[material] ?: bank.getValue("NEUTRAL")
        return arr[variant.coerceIn(0, arr.size - 1)]
    }

    fun variantCount(material: String): Int =
        (bank[material] ?: bank.getValue("NEUTRAL")).size
}
