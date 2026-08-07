#!/usr/bin/env python3
"""Benchmark single-forward-pass inference throughput of the YOLO model.

Reproduces the FPS table in the README (imgsz 416/512/640). This measures the
raw model throughput only — the live feed adds frame encoding and the
frame-skip overlay, so the end-to-end detection-update rate is lower.

Usage:
    python scripts/benchmark.py                      # uses PlantDiseaseAPP/plant_disease/best.pt
    python scripts/benchmark.py path/to/weights.pt

Requires: ultralytics, torch, opencv-python.
"""

import sys
import time
from pathlib import Path

import numpy as np

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "PlantDiseaseAPP" / "plant_disease" / "best.pt"
IMGSZS = (416, 512, 640)
WARMUP = 3
ITERS = 30


def bench(model, imgsz):
    frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype="uint8")
    for _ in range(WARMUP):
        model.predict(frame, imgsz=imgsz, verbose=False)
    start = time.perf_counter()
    for _ in range(ITERS):
        model.predict(frame, imgsz=imgsz, verbose=False)
    latency = (time.perf_counter() - start) / ITERS * 1000  # ms
    return latency


def main():
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}\n"
                 f"Download them with: python scripts/download_weights.py")
    model = YOLO(weights)
    print(f"Model: {model.task} ({weights.name}), classes: {len(model.names)}")
    print(f"Device: {'CUDA' if model.device.type != 'cpu' else 'CPU'}")
    print(f"\n{'imgsz':>5} {'latency (ms)':>12} {'FPS':>8}")
    print("-" * 27)
    for imgsz in IMGSZS:
        ms = bench(model, imgsz)
        print(f"{imgsz:>5} {ms:>12.1f} {1000 / ms:>8.1f}")
    print("\nNote: single-forward-pass throughput only; the live feed's "
          "frame-skip overlay updates detections at a lower rate.")


if __name__ == "__main__":
    main()
