# Third-party licenses and components

## Neural network models (bundled in APK assets)

### yolo26s.onnx — YOLO26-S object detector (COCO-80, default since v1.4.0)
- Source: exported from Ultralytics YOLO26-S official weights (imgsz=640, end-to-end)
- License: **AGPL-3.0** (Ultralytics); same terms as the Nano below.
- https://docs.ultralytics.com/models/yolo26/

### yoloe26s.onnx — YOLOE-26s open-vocabulary detector (default since v1.5.0)
- Source: Ultralytics YOLOE-26s (detection-only head), exported to ONNX @640 with
  the EchoLens 426-class vocabulary baked in as text-prompt embeddings
  (mobileclip2_b text encoder, Apache-2.0, Apple MobileCLIP weights).
- License: **AGPL-3.0** (Ultralytics), same terms as the other Ultralytics models
  below; the EchoLens application code remains MIT.

### yolo26n.onnx — YOLO26-Nano object detector (COCO-80, fast fallback)
- Source: exported from Ultralytics YOLO26-N official weights
- License: **AGPL-3.0** (Ultralytics). The model file is distributed under its
  own license; EchoLens application code remains MIT. If AGPL does not suit
  your use case, swap the asset for an Apache-2.0 detector (e.g. RF-DETR-N)
  keeping the same [1,300,6] or [1,84,N] decode contract.
- https://docs.ultralytics.com/models/yolo26/

### depth_v2_small.onnx — Depth Anything V2 Small (relative monocular depth)
- Source: https://huggingface.co/onnx-community/depth-anything-v2-small
  (onnx/model_quantized.onnx, INT8 dynamic)
- Base model: https://huggingface.co/depth-anything/Depth-Anything-V2-Small
- License: **Apache-2.0**

## Sound library (bundled in APK assets, `app/src/main/assets/sounds/`)

Since v1.6.0 the per-class sound packs, material loops and one-shot accents are
curated REAL recordings (not synthesized) sourced from the **FSD50K** dataset:

- Dataset: FSD50K — an open dataset of human-labeled sound events
  (Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra).
- Source: https://zenodo.org/records/4060432 / https://github.com/DCASE-REPO/FSD50K
- Paper: Fonseca et al., "FSD50K: An Open Dataset of Human-Labeled Sound Events", IEEE/ACM TASLP 2022.
- Licensing: every included clip is **CC0 1.0 or CC-BY 4.0/3.0** (no NC, no
  Sampling+). Per-file attribution (FSD50K id, title, uploader, license) is in
  `app/src/main/assets/sounds/ATTRIBUTIONS.md`. CC-BY clips remain the property
  of their respective Freesound.org uploaders, credited there.
- Processing: clips were quality-filtered (SNR/clipping gates), loopified with
  tail-head crossfade, loudness-normalized and re-encoded to mono 48 kHz OGG.

### History (v1.4.0, superseded)

The original v1.4.0 library (924 OGG samples, MIT, synthesized by this project
with physical-model DSP) was fully replaced by the FSD50K recordings in v1.6.0.
The synthesis scripts are no longer part of the repository.

## Runtime libraries

### ONNX Runtime for Android
- Maven: com.microsoft.onnxruntime:onnxruntime-android
- License: **MIT** — https://github.com/microsoft/onnxruntime

### AndroidX / Jetpack (CameraX, Lifecycle, AppCompat, Material)
- License: **Apache-2.0**

### TensorFlow Lite / LiteRT (v1.0 detector; v1.2+ neural textures)
- Maven: org.tensorflow:tensorflow-lite
- License: **Apache-2.0**

## On-device generative audio (v1.2+, optional, provisioned outside the APK)

### Stable Audio Open Small (neural texture generation)
- Weights: https://huggingface.co/stabilityai/stable-audio-open-small
- License: **Stability AI Community License** — free for commercial and
  non-commercial use for organizations under $1M annual revenue; generated
  outputs belong to the user. The model files are NOT bundled: the user
  provisions them into the app models dir (see tools/litert_models.md).
- Reference LiteRT pipeline: Arm ML-examples kleidiai-examples/audiogen
  (export_sao.py, audiogen.cpp) — **Apache-2.0**, Copyright Arm Limited.
  The Kotlin port in app/.../audio/neural/ follows that implementation.

### RAVE (optional per-material backends, rave_*.onnx)
- Source: https://github.com/acids-ircam/RAVE (IRCAM / ACIDS)
- License: **MIT** (code). Trained model checkpoints carry their own terms —
  check the IRCAM Forum model card before shipping a specific checkpoint.

### MINC-2500 dataset (optional material classifier)
- Source: http://opensurfaces.cs.cornell.edu/publications/minc/
- License: research/academic terms of the MINC dataset; the classifier model
  file is user-provisioned, not bundled.

## v1.0 detector (history)
### ssd_mobilenet_v1 quantized (COCO)
- Source: https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
- License: **Apache-2.0** (TensorFlow)
