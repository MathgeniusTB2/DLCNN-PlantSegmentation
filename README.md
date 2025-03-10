# DLCNN-PlantSegmentation

Deep-learning CNN system for plant disease detection and segmentation, with live
video inference from a webcam or a DJI Tello drone and a Django web dashboard.

See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for the full project
write-up (dataset, training results, live-feed tuning, limitations).

## Repository structure

```
PlantDiseaseAPP/
├── manage.py                    # Django entry point
├── plant_disease_detection/     # Django project config (settings, urls)
├── plant_disease/               # Django app: views, routes, templates, static
│   ├── views.py                 # YOLO inference, video/analysis streaming, captures
│   ├── templates/plant_disease/ # Web dashboard
│   └── static/captures/         # Captured/analysed images (runtime)
└── scripts/
    └── drone/                   # Standalone drone + webcam scripts
a3_part_C.ipynb                  # Training notebook (YOLOv8 / Faster R-CNN)
requirements.txt                 # Web app dependencies
requirements-training.txt        # Notebook/training dependencies
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd PlantDiseaseAPP
python manage.py migrate
python manage.py runserver
```

Open http://localhost:8000 — the dashboard shows the live camera feed, detection
overlay, per-frame disease analysis, capture history, and ZIP export.

### Endpoints

| URL              | Description                                  |
|------------------|----------------------------------------------|
| `/`              | Dashboard                                    |
| `/video_feed/`   | MJPEG stream (`?source=webcam` or `drone`, `?overlay=true`) |
| `/analysis_feed/`| SSE disease analysis stream                  |
| `/capture/`      | Capture + analyse the current frame          |
| `/history/`      | JSON capture history                         |
| `/export/`       | Download all captures as a ZIP               |
| `/admin/`        | Django admin                                 |

## Model weights

The app loads `PlantDiseaseAPP/plant_disease/best.pt` for detection. If the file
is missing it automatically falls back to a pretrained `yolov8n.pt` model so
the app runs out of the box. Place your trained model at that path for the real
detection classes.

### Live-feed tuning

The streaming endpoints (`/video_feed/`, `/analysis_feed/`) can be tuned via
environment variables so they stay responsive on a live drone/webcam feed:

| Env var          | Default | Meaning                                        |
|------------------|---------|------------------------------------------------|
| `MODEL_SIZE`     | `n`     | Fallback model size (`n/s/m/l/x`) when `best.pt` is absent |
| `MODEL_IMGSZ`    | `416`   | Inference input resolution                     |
| `MODEL_FRAME_SKIP` | `3`   | Run inference only on every Nth frame; the overlay persists in between |

Example:

```bash
MODEL_SIZE=s MODEL_IMGSZ=512 MODEL_FRAME_SKIP=2 python manage.py runserver
```

## Drone (DJI Tello)

The dashboard can stream from a Tello (`?source=drone`). Standalone scripts live
in `PlantDiseaseAPP/scripts/drone/`:

```bash
python scripts/drone/drone_control.py    # keyboard-controlled flight + live feed
python scripts/drone/webcam_demo.py      # webcam inference demo using best.pt
```

Controls (drone_control.py): `W/S/A/D` move, space up, `X` down, Enter takeoff,
`O/P` rotate, `Q` quit.

## Training the model

The notebook `a3_part_C.ipynb` trains YOLOv8l, YOLOv8x and Faster R-CNN
detectors on the `plantsegv3` dataset (COCO-style annotations). The dataset is
**not** committed to this repo; place it at `PlantDiseaseAPP/plantsegv3/` with:

```
plantsegv3/
├── images/
│   ├── train/
│   └── test/
└── masks/
    └── annotation_*.json
```

Install training dependencies with `pip install -r requirements-training.txt`,
then run the notebook top to bottom. The trained weights are written to
`runs/detect/*/weights/best.pt` — copy one to
`PlantDiseaseAPP/plant_disease/best.pt` for the web app.

### Dataset

The models are trained on **PlantSeg: A Large-Scale In-the-wild Dataset for
Plant Disease Segmentation** — 11,400+ images of 115 plant diseases with
instance-level annotations.

- Paper: https://arxiv.org/abs/2409.04038
- Download (Zenodo): https://zenodo.org/records/14935094
- Project: https://github.com/tqwei05/PlantSeg

> Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P., Huang, Z. "PlantSeg: A
> Large-Scale In-the-wild Dataset for Plant Disease Segmentation." arXiv
> preprint arXiv:2409.04038, 2024.
