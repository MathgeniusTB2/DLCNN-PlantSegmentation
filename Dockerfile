# syntax=docker/dockerfile:1
# Container image for the PlantSeg disease-detection dashboard.
#
#   docker build -t plantseg-dashboard .
#   docker run -p 8000:8000 plantseg-dashboard
#   -> open http://localhost:8000 (demo mode, streams bundled sample images)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBUG=0 \
    DJANGO_ALLOWED_HOSTS=* \
    DEMO_IMAGES_DIR=/app/docs/demo_images \
    MODEL_IMGSZ=416 \
    MODEL_FRAME_SKIP=3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Best-effort: fetch the trained 115-class weights if a GitHub Release is
# available; otherwise the app falls back to a pretrained COCO model.
RUN python scripts/download_weights.py \
    || echo "release weights unavailable - the app will fall back to pretrained weights"

# Apply database migrations (the SQLite file is created on first run).
RUN cd PlantDiseaseAPP && python manage.py migrate --noinput

EXPOSE 8000

WORKDIR /app/PlantDiseaseAPP
CMD ["gunicorn", "plant_disease_detection.wsgi:application", \
     "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120"]
