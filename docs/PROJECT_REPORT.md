# Plant Disease Detection — Project Report

A deep-learning system for plant disease detection and segmentation with a live
Django web dashboard and DJI Tello drone integration.

## 1. Overview

The system detects plant diseases in live video using a YOLO-family detector and
serves the annotated feed, per-frame disease analysis, capture history, and ZIP
export through a Django web app. It supports two video sources: a standard
webcam and a DJI Tello drone (via `djitellopy`), so fields can be inspected
aerially.

Components:

- `plant_disease` Django app — streaming views, inference, capture/export.
- `a3_part_C.ipynb` — model training notebook (YOLOv8l / YOLOv8x / Faster R-CNN).
- `scripts/drone/` — standalone drone control and webcam demo scripts.
- `PlantDiseaseAPP/plant_disease/best.pt` — trained model weights (place here).

## 2. Dataset

Trained on **PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease
Segmentation** (Wei et al., 2024):

- 11,400+ in-the-wild images, **115 plant disease classes**, instance
  segmentation annotations (COCO format).
- Paper: https://arxiv.org/abs/2409.04038 · Download: https://zenodo.org/records/14935094
- Project: https://github.com/tqwei05/PlantSeg

> Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P., Huang, Z. "PlantSeg: A
> Large-Scale In-the-wild Dataset for Plant Disease Segmentation." arXiv
> preprint arXiv:2409.04038, 2024.

The COCO annotations were converted to YOLO label format (`labels/<split>/*.txt`)
for training. Out-of-bounds boxes (which the raw annotations contain in
quantity) were clamped to image bounds during conversion.

## 3. Model training

The production model is a **YOLOv8s trained at `imgsz=640`** on the CETUS HPC
(RTX PRO 6000 Blackwell), early-stopped at epoch 95 (best at 75). Training
labels were cleaned during COCO→YOLO conversion (out-of-bounds boxes clamped),
which the original runs lacked.

**Current model (used by the web app):**

| Metric | Value |
|---|---|
| **mAP50** | **0.319** |
| **mAP50-95** | **0.197** |
| Precision / Recall | 0.378 / 0.350 |
| Val set | 1,247 images / 8,926 instances |
| Speed (CPU @416) | 38 ms/img — **26 FPS** |
| Speed (CPU @640) | 84 ms/img — **12 FPS** |

**Previous training runs** (AWS SageMaker, Tesla T4, `imgsz=416`):

| Model | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---|---|---|---|---|
| YOLOv8l (final) | 0.419 | 0.371 | **0.352** | **0.223** | best of the old runs |
| YOLOv8x | ~0.39 | ~0.33 | ~0.335 | ~0.211 | bigger model, no gain |
| Faster R-CNN R50-FPN | — | — | — | — | training only; no final metrics |

Observations:

- 115 fine-grained, in-the-wild classes is a hard task; the dataset's own
  segmentation baselines (mIoU 17.2–44.5) show this.
- YOLOv8x did not beat YOLOv8l → the bottleneck is data quality / resolution /
  class balance, not model capacity.
- Severe class imbalance: rare classes (a handful of val images) score
  mAP50 ≈ 0.02–0.17 and drag the mean down.
- Label noise was present (many annotations extend beyond image bounds); these
  were clamped/dropped in the YOLO conversion.

## 4. Live inference

The web app runs detection on the live webcam/drone stream. Measured throughput
of the trained YOLOv8s on a mid-range laptop CPU (no GPU):

| imgsz | Latency | FPS |
|---|---|---|
| 416 | 38 ms | **26 FPS** |
| 512 | 52 ms | **19 FPS** |
| 640 | 84 ms | **12 FPS** |

With the default feed settings (`imgsz=416`, `MODEL_FRAME_SKIP=3`) classification
never bottlenecks a Tello's ~10–30 fps stream.

A simulated drone flyover of PlantSeg crops (live detections with the HUD) is
shown in [`docs/demo_drone_scan.gif`](docs/demo_drone_scan.gif).

Live-feed settings (env vars, see README):

- `MODEL_SIZE` — fallback model size `n/s/m/l/x` when `best.pt` is absent.
- `MODEL_IMGSZ` — inference resolution (default 416; 512–640 for accuracy).
- `MODEL_FRAME_SKIP` — run inference every Nth frame and re-render the overlay in
  between, keeping the stream responsive on slower hardware.

For a live drone feed, a **YOLOv8s** model inferring at ~416–512 is the
recommended balance: real-time latency matters more than the few mAP points a
larger model adds, and the moving feed already degrades small-lesion accuracy.

## 5. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # app
pip install -r requirements-training.txt   # notebook (torch, pycocotools, ...)
cd PlantDiseaseAPP
python manage.py migrate && python manage.py runserver
```

- Place trained weights at `PlantDiseaseAPP/plant_disease/best.pt`.
- Standalone drone control: `python scripts/drone/drone_control.py`.
- Training notebook: `a3_part_C.ipynb` (needs the PlantSeg dataset staged as
  `PlantDiseaseAPP/plantsegv3/`).

## 6. Limitations & future work

- **Detection vs segmentation:** YOLO outputs boxes, not lesion masks. For true
  segmentation output, a YOLO-seg or mask head would be needed.
- **imgsz=416 limits small lesions:** retraining at 640 and/or higher-resolution
  inference improves recall on small lesions.
- **Class imbalance:** class-weighted loss or oversampling for rare classes
  would lift mean mAP.
- **Label noise:** the raw annotations contain out-of-bounds boxes; cleaning
  them (now done in the conversion step) helps training stability.
- **Live drone latency:** frame-skip keeps the feed usable on CPU; a GPU
  backend removes the need for it.
