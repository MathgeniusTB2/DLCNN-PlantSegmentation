# Used by platforms that run a Procfile directly (Render uses render.yaml's
# startCommand; this is a drop-in equivalent for e.g. Heroku).
web: cd PlantDiseaseAPP && gunicorn plant_disease_detection.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120
