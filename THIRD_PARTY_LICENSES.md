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

### TensorFlow Lite (used in v1.0, removed in v1.1)
- License: **Apache-2.0**

## v1.0 detector (history)
### ssd_mobilenet_v1 quantized (COCO)
- Source: https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
- License: **Apache-2.0** (TensorFlow)
