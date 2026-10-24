# Third-party licenses and components

## Neural network models (bundled in APK assets)

### yolo26n.onnx — YOLO26-Nano object detector (COCO-80)
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
