# On-device neural texture models (Stable Audio Open Small → LiteRT)

EchoLens can generate its material sound textures **on the phone itself** —
no audio bank is shipped inside the APK. The pipeline runs Stable Audio Open
Small (341M, Stability AI Community License) through LiteRT, following the
official Arm reference implementation (Apache-2.0):
https://github.com/ARM-software/ML-examples/tree/main/kleidiai-examples/audiogen

App side: `NeuralTextureEngine.kt` binds model inputs **by shape** — the
LiteRT export reorders the DiT arguments relative to the torch signature, so
any conversion with the geometry below works. Requires **app v1.2.1+**
(v1.2.0 assumed the torch argument order and silently fell back to
procedural synthesis with the real models).

## Verified tensor signatures (probed from the real .tflite files)

| Model | Inputs | Outputs |
|---|---|---|
| `conditioners_float32.tflite` | `input_ids` int64 (1,128) · `attention_mask` int64 (1,128) · `seconds_total` f32 **(1)** | `[0]` cross embeds f32 (1,65,768) · `[1]` cross mask (1,65) · `[2]` global f32 (1,768) |
| `dit_model.tflite` | cross (1,65,768) · global (1,768) · x (1,64,256) · sigma f32 **(1)** | denoised latent (1,64,256) |
| `autoencoder_model.tflite` | latent (1,64,256) | planar stereo f32 (2,524288) — 11.9 s @ 44.1 kHz |

Reference golden run (x86, 4 threads): conditioners 0.2 s + DiT 8 steps 2.6 s
+ VAE 14.6 s → 11.6 s of audio. On a Snapdragon flagship expect ~15–40 s for
the whole texture.

## 1. Get the models (one-time)

### Option 0 — already bundled in the APK (v1.3.0+)

Nothing to do. The APK ships all three `.tflite` files as assets
(weight-only int8 conditioner 186 MB + int8 DiT 344 MB + fp16 VAE 313 MB
≈ 844 MB; the quantized conditioner verified against fp32: output cosine
similarity ≥ 0.9998, spectral profile within 8%). On first use the app
unpacks them into `/sdcard/Android/data/com.echolens.app/files/models/`.

Use the options below only if you want the *original fp32* conditioner,
custom conversions, or RAVE/MINC extras.

### Option A — download the prebuilt Arm-converted package (HF CLI)

```bash
pip install -U "huggingface_hub[cli]"
hf download jegly/audio conditioners_float32.tflite dit_model.tflite \
    autoencoder_model.tflite spiece.model --local-dir models
```

This is the byte-exact Arm pipeline package (identical files as the gated
mirror `innermost47/stable-audio-open-small-tflite`, which requires accepting
the license on https://huggingface.co/stabilityai/stable-audio-open-small
first). The weights are covered by the Stability AI Community License —
commercial use is allowed for organizations under $1M revenue
(see `THIRD_PARTY_LICENSES.md`).

### Option B — convert yourself from the gated checkpoint (PC with ~16 GB RAM)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install "stable_audio_tools==0.0.19" "litert-torch==0.9.0" "numpy==1.26.4"

# Download the gated checkpoint (accept the license on HF first):
huggingface-cli download stabilityai/stable-audio-open-small \
    model_config.json model.ckpt --local-dir workspace

# Export the three LiteRT modules (script from Arm ML-examples, Apache-2.0):
wget https://raw.githubusercontent.com/ARM-software/ML-examples/main/kleidiai-examples/audiogen/scripts/export_sao.py
wget https://raw.githubusercontent.com/ARM-software/ML-examples/main/kleidiai-examples/audiogen/scripts/model.py
python3 export_sao.py --model_config workspace/model_config.json \
    --ckpt_path workspace/model.ckpt --output_path out/
```

This produces:

| File | Role | Precision |
|---|---|---|
| `conditioners_float32.tflite` | T5 text + number conditioner | fp32 |
| `dit_model.tflite` | diffusion transformer | dynamic int8 |
| `autoencoder_model.tflite` | VAE decoder | fp16 |

## 2. Push to the phone

```bash
adb shell mkdir -p /sdcard/Android/data/com.echolens.app/files/models
adb push conditioners_float32.tflite dit_model.tflite \
         autoencoder_model.tflite \
         /sdcard/Android/data/com.echolens.app/files/models/
```

(Or copy with any file manager into the same folder. `spiece.model` is not
needed — EchoLens ships pretokenized prompts in `MaterialPrompts.kt`.)

## 3. In the app

Settings → «Нейро-текстуры материалов» → ON.
The status line under the switch shows whether the models were found and
whether the device has enough RAM (≥ 6 GB total recommended; ~3 GB peak is
used during generation).

First time a material appears (стул → WOOD, машина → ENGINE, …) the app:
1. plays the procedural texture immediately (zero latency),
2. generates the neural loop in the background (~15–40 s, 8 diffusion steps),
3. crossfades the stream to the neural loop and caches it forever
   (`files/neural_textures/*.pcm`). Next sessions reuse the cache instantly.

## Optional: per-material RAVE backends (faster, real-time)

RAVE (IRCAM, MIT) models trained per material can be exported to ONNX
(nn~ compatible) and dropped next to the LiteRT models:

```
rave_WOOD.onnx    rave_METAL.onnx    rave_WATER.onnx  ...
rave_WOOD.json    # optional: {"sample_rate": 44100}
```

`rave_<material>.onnx` takes priority over Stable Audio for that material and
generates textures in seconds on the CPU.

## Optional: visual material classifier (MINC-23)

`minc23_mobilenet_v3_int8.tflite` (see `tools/train_minc_classifier.py`) in the
models dir enables pixel-level material refinement: the detector says "bottle"
(glass by default), the classifier looks at the crop and may correct it to
"plastic" — the texture changes accordingly.

## Device notes

- The DiT runs with dynamic int8 weights (XNNPACK), the VAE decodes in fp16.
- Generation uses 4 CPU threads at background priority; the audio thread
  (48 kHz, 20 ms blocks) is never blocked — worst case the neural loop simply
  arrives later.
- On a mid-range phone without the model files nothing changes: the app works
  exactly as before with procedural synthesis.
