#!/usr/bin/env python3
# Generate app/src/main/java/com/echolens/app/audio/neural/MaterialPrompts.kt
# with pre-tokenized T5 ids for the fixed material prompt bank.
#
# The tokenizer is t5-base (SentencePiece). The Stable Audio Open Small
# "conditioners_float32.tflite" expects int64 token ids + attention mask,
# so we pre-compute ids on the PC and ship them in code — no tokenizer
# is needed on the phone.
#
# Usage:  python3 tools/make_prompt_tokens.py [path/to/spiece.model]
# Get spiece.model: https://huggingface.co/google-t5/t5-base/resolve/main/spiece.model

import sys
import os

try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")

# 15 materials x 2 variants. English, loop-friendly descriptions.
PROMPTS = {
    "WOOD": [
        "dry wooden knocks and creaks, furniture wood, close range, soft room reverb, seamless loop",
        "gentle knocking on a wooden table, warm resonance, sparse rhythm, seamless loop",
    ],
    "METAL": [
        "metallic rattles and clangs, distant street metal, sparse hits, seamless loop",
        "hollow metal tube resonating with occasional bright strikes, seamless loop",
    ],
    "GLASS": [
        "glass clinks and delicate crystal ringing, sparse, close, seamless loop",
        "wine glass humming softly with occasional touch, shimmering high tones, seamless loop",
    ],
    "CERAMIC": [
        "ceramic taps and porcelain knocks on a table, dry, seamless loop",
        "porcelain bowl touches, soft mid-range resonance, sparse, seamless loop",
    ],
    "ENGINE": [
        "car engine idling steadily, warm low rumble, seamless loop",
        "smooth motor hum with subtle vibrations, constant, seamless loop",
    ],
    "WATER": [
        "water trickling and small bubbles, stream flowing gently, seamless loop",
        "soft water splashes and droplets in a quiet basin, seamless loop",
    ],
    "ELECTRONICS": [
        "electronic device hum with tiny clicks and high-frequency chirps, seamless loop",
        "quiet gadget buzz and short digital blips, sparse, seamless loop",
    ],
    "FABRIC": [
        "fabric rustling softly, clothes movement, close and airy, seamless loop",
        "soft cloth flapping and brushing sounds, gentle, seamless loop",
    ],
    "PLANT": [
        "leaves rustling in a light breeze, dry foliage, seamless loop",
        "plant leaves brushing against each other, organic crackle, seamless loop",
    ],
    "PAPER": [
        "paper crumpling and pages turning, close, dry, seamless loop",
        "light paper crackle and sheet movements, sparse, seamless loop",
    ],
    "PERSON": [
        "person breathing calmly and shifting nearby, quiet presence, seamless loop",
        "soft human rustle and steps close by, seamless loop",
    ],
    "ANIMAL": [
        "dog barking softly in the distance with movement, seamless loop",
        "cat purring gently, warm animal sound, seamless loop",
    ],
    "BIRD": [
        "birds chirping in a tree, morning garden, varied calls, seamless loop",
        "small bird song phrases with pauses, natural, seamless loop",
    ],
    "FOOD": [
        "food being unwrapped and bitten, crunchy textures, close, seamless loop",
        "soft food crumble and packaging rustle, seamless loop",
    ],
    "NEUTRAL": [
        "soft neutral object hum with gentle taps, ambiguous material, seamless loop",
        "muted taps on an unknown surface, calm, seamless loop",
    ],
}

MATERIALS = ["WOOD", "METAL", "GLASS", "CERAMIC", "ENGINE", "WATER", "ELECTRONICS",
             "FABRIC", "PLANT", "PAPER", "PERSON", "ANIMAL", "BIRD", "FOOD", "NEUTRAL"]


def main():
    sp_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/spiece.model"
    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)

    lines = []
    lines.append("package com.echolens.app.audio.neural")
    lines.append("")
    lines.append("/**")
    lines.append(" * Fixed prompt bank for on-device neural texture generation")
    lines.append(" * (Stable Audio Open Small, LiteRT).")
    lines.append(" *")
    lines.append(" * Prompts are pre-tokenized with the t5-base SentencePiece model")
    lines.append(" * (EOS id 1 appended), so the phone needs no tokenizer at all.")
    lines.append(" * Regenerate with tools/make_prompt_tokens.py.")
    lines.append(" *")
    lines.append(" * Generated file — do not edit by hand.")
    lines.append(" */")
    lines.append("object MaterialPrompts {")
    lines.append("")
    lines.append("    class Prompt(")
    lines.append("        val text: String,")
    lines.append("        val ids: IntArray")
    lines.append("    )")
    lines.append("")
    lines.append("    /** [material][variant] */")
    lines.append("    private val bank: Map<String, Array<Prompt>> = mapOf(")
    for m in MATERIALS:
        lines.append(f"        \"{m}\" to arrayOf(")
        for p in PROMPTS[m]:
            ids = list(sp.encode(p)) + [1]
            arr = ", ".join(str(i) for i in ids)
            lines.append(f"            Prompt(\"{p}\", intArrayOf({arr})),")
        lines.append("        ),")
    lines.append("    )")
    lines.append("")
    lines.append("    fun prompt(material: String, variant: Int): Prompt {")
    lines.append("        val arr = bank[material] ?: bank.getValue(\"NEUTRAL\")")
    lines.append("        return arr[variant.coerceIn(0, arr.size - 1)]")
    lines.append("    }")
    lines.append("")
    lines.append("    fun variantCount(material: String): Int =")
    lines.append("        (bank[material] ?: bank.getValue(\"NEUTRAL\")).size")
    lines.append("}")

    out = os.path.join(os.path.dirname(__file__), "..",
                       "app/src/main/java/com/echolens/app/audio/neural/MaterialPrompts.kt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written {os.path.abspath(out)}")
    for m in MATERIALS[:2]:
        ids = list(sp.encode(PROMPTS[m][0])) + [1]
        print(m, "ids:", ids[:8], "...", len(ids), "tokens")


if __name__ == "__main__":
    main()
