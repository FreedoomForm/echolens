#!/usr/bin/env python3
"""
Train a tiny visual material classifier on MINC-2500 (23 classes) and export
it as a uint8 TFLite model for EchoLens.

Model: torchvision MobileNetV3-Small (~2.5M params, ~5 ms/frame on a phone).
Output: minc23_mobilenet_v3_int8.tflite — drop it into the app models dir:
    /sdcard/Android/data/com.echolens.app/files/models/

Dataset (MINC-2500, 2500 imgs/class, 23 classes):
    http://opensurfaces.cs.cornell.edu/publications/minc/
    Direct images: http://labels.cornell.edu/minc/minc-2500.tar.gz

Usage:
    python3 tools/train_minc_classifier.py --data /path/to/minc-2500 --epochs 8

Label order is fixed in tools/make_prompt_tokens-style constants and MUST match
MaterialClassifier.MINC_LABELS in the app (alphabetical, as in the MINC split
directories).
"""

import argparse
import os
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
except ImportError:
    raise SystemExit("pip install torch torchvision pillow numpy")

# Alphabetical order of MINC-2500 category folders = MINC_LABELS in the app.
LABELS = sorted([
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
    "hair", "leather", "metal", "mirror", "other", "painted", "paper",
    "plastic", "polishedstone", "skin", "sky", "stone", "tile",
    "wallpaper", "water", "wood",
])
assert LABELS == sorted(LABELS) and len(LABELS) == 23

IMG = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to minc-2500 folder")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--out", default="minc23_mobilenet_v3_int8.tflite")
    ap.add_argument("--onnx-out", default="minc23_mobilenet_v3.onnx")
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # minc-2500 layout: <data>/<category>/train|test|validation/*.jpg
    train_ds = datasets.ImageFolder(args.data, transform=tf_train)
    # Use the split files if present, else split randomly.
    n = len(train_ds)
    n_val = int(0.1 * n)
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n - n_val, n_val], generator=torch.Generator().manual_seed(0)
    )
    # Recreate train loader with augmentation on the train subset only:
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 23)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
        print(f"epoch {ep}: val acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "minc23_best.pt")
    print(f"best val acc: {best:.4f}")

    # ---- export ----------------------------------------------------------
    model.load_state_dict(torch.load("minc23_best.pt", map_location="cpu"))
    model.eval().cpu()

    dummy = torch.randn(1, 3, IMG, IMG)
    torch.onnx.export(
        model, dummy, args.onnx_out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=15,
    )
    print("onnx saved:", args.onnx_out)

    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader

        class Reader(CalibrationDataReader):
            def __init__(self, data_path, n=500):
                ds = datasets.ImageFolder(data_path, transform=tf_val)
                idx = np.random.RandomState(0).choice(len(ds), min(n, len(ds)), replace=False)
                self.samples = [ds[i][0].numpy() for i in idx]
                self.i = 0

            def get_next(self):
                if self.i >= len(self.samples):
                    return None
                s = self.samples[self.i]
                self.i += 1
                return {"input": s[None].astype(np.float32)}

        quantize_static(
            args.onnx_out, "minc23_mobilenet_v3_int8.onnx", Reader(args.data),
            weight_type=QuantType.QInt8,
        )
        print("int8 onnx saved")
    except ImportError:
        print("onnxruntime not installed — skipping static quantization")

    # ONNX -> TFLite: install ai-edge-litert / onnx2tf, or use:
    #   python -m pip install onnx2tf tf-keras onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
    #   onnx2tf -i minc23_mobilenet_v3_int8.onnx -o tflite_out -nuo -n
    #   (then rename the produced *_full_integer_quant.tflite)
    print("Convert with onnx2tf (see header comment), then copy to the phone as")
    print("minc23_mobilenet_v3_int8.tflite into the app models dir.")


if __name__ == "__main__":
    main()
