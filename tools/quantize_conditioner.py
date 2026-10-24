#!/usr/bin/env python3
"""Quantize the Stable Audio Open Small T5 conditioner for EchoLens bundling.

Weight-only int8 (channelwise min-max) for FULLY_CONNECTED + BATCH_MATMUL:
weights become int8, compute stays float with explicit dequant ops -> near
-lossless (verified on x86 vs fp32: output cosine similarity >= 0.9998 for
WOOD/METAL prompts, spectral profile of the full pipeline within 8%).

The embedding table (GATHER) is not weight-only-quantizable in
ai-edge-quantizer 0.10.0 and stays fp32, hence 440 -> ~186 MB.

Deterministic (min-max scales over fixed weights) -> CI produces the same
model as local verification. Run with Python 3.10+:
    pip install ai-edge-quantizer
    python3 tools/quantize_conditioner.py --src IN.tflite --dst OUT.tflite
"""
import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="fp32 conditioners tflite")
    ap.add_argument("--dst", required=True, help="output path for quantized model")
    args = ap.parse_args()

    # ai-edge-quantizer needs its own import; keep the failure readable.
    try:
        from ai_edge_quantizer import quantizer, qtyping
    except ImportError as e:
        raise SystemExit(f"pip install ai-edge-quantizer first ({e})")

    t0 = time.time()
    print(f"[1/3] loading {args.src} ...", flush=True)
    q = quantizer.Quantizer(args.src)

    print("[2/3] adding weight-only int8 configs (FC + BATCH_MATMUL) ...", flush=True)
    for op in (qtyping.TFLOperationName.FULLY_CONNECTED,
               qtyping.TFLOperationName.BATCH_MATMUL):
        q.add_weight_only_config(regex=".*", operation_name=op, num_bits=8,
                                 granularity=qtyping.QuantGranularity.CHANNELWISE)
        print(f"      + {op.value}", flush=True)

    print("[3/3] quantize + serialize ...", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    q.quantize(serialize_to_path=args.dst)

    s_in = os.path.getsize(args.src) / 1e6
    s_out = os.path.getsize(args.dst) / 1e6
    print(f"DONE  {s_in:.0f} MB -> {s_out:.0f} MB ({s_out / s_in * 100:.0f}%) "
          f"in {time.time() - t0:.0f}s -> {args.dst}")


if __name__ == "__main__":
    main()
