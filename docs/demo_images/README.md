# Demo images

Six sample frames from the **PlantSeg** test split, used by the web app's
demo mode (`DEMO_IMAGES_DIR`). They let you run the full dashboard — real
detection overlay + analysis — with no webcam or drone:

```bash
pip install -r requirements.txt && python scripts/download_weights.py
cd PlantDiseaseAPP && python manage.py runserver
# then open http://localhost:8000 (demo source is the default when these images exist)
```

These images are redistributed under the dataset's
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license with
attribution to the authors:

> Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P., Huang, Z. "PlantSeg: A
> Large-Scale In-the-wild Dataset for Plant Disease Segmentation." arXiv
> preprint arXiv:2409.04038, 2024.
>
> Dataset: https://zenodo.org/records/14935094 · Paper: https://arxiv.org/abs/2409.04038

Individual image credits (watermarked in the files where applicable) belong to
their original photographers, as sourced by the dataset.
