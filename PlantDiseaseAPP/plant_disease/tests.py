import json
from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase
from django.urls import reverse

from plant_disease import views


class FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Minimal stand-in for an Ultralytics YOLO model (no torch needed)."""

    def __init__(self, names):
        self.names = names

    def predict(self, frame, imgsz=416, verbose=False):
        h, w = frame.shape[:2]
        return [FakeResult([FakeBox(0, 0.95, [w * 0.1, h * 0.1, w * 0.5, h * 0.5])])]


class DashboardTests(TestCase):
    """Smoke tests for the web dashboard (no camera or GPU required)."""

    def setUp(self):
        self._orig = {
            "capture_history": views.capture_history,
            "webcam": views.webcam,
            "drone": views.drone,
            "demo_images": views.DEMO_IMAGES,
            "default_source": views.DEFAULT_SOURCE,
            "demo_index": views._demo_index,
            "get_model": views.get_model,
        }
        views.capture_history = []
        views.webcam = None
        views.drone = None
        views.DEMO_IMAGES = []
        views.DEFAULT_SOURCE = "webcam"
        views._demo_index = 0
        views.get_model = lambda: FakeModel({0: "apple scab"})

    def tearDown(self):
        views.capture_history = self._orig["capture_history"]
        views.webcam = self._orig["webcam"]
        views.drone = self._orig["drone"]
        views.DEMO_IMAGES = self._orig["demo_images"]
        views.DEFAULT_SOURCE = self._orig["default_source"]
        views._demo_index = self._orig["demo_index"]
        views.get_model = self._orig["get_model"]

    def test_index_renders(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'video-feed')

    def test_history_returns_empty_json(self):
        response = self.client.get(reverse('get_history'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content), {'history': []})

    def test_export_without_captures_returns_400(self):
        views.capture_history = []
        response = self.client.get(reverse('export_results'))
        self.assertEqual(response.status_code, 400)

    def test_capture_without_camera_returns_400(self):
        response = self.client.get(reverse('capture_image'))
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', json.loads(response.content))

    def test_demo_feed_reports_when_unconfigured(self):
        response = self.client.get(reverse('video_feed'), {'source': 'demo'})
        self.assertContains(response, 'Demo mode is not configured')

    def test_class_name_uses_model_names(self):
        views.get_model = lambda: FakeModel({0: "apple scab"})
        self.assertEqual(views.class_name(0), "apple scab")

    def test_class_name_falls_back_to_disease_classes(self):
        views.get_model = lambda: FakeModel(None)
        self.assertEqual(views.class_name(0), views.DISEASE_CLASSES[0])

    def test_capture_demo_roundtrip(self):
        tmp_dir = Path(views.CAPTURES_DIR.parent) / "test_demo_images"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        demo_img = tmp_dir / "demo.jpg"
        img = 200 * np.ones((240, 320, 3), dtype="uint8")
        cv2.imwrite(str(demo_img), img)

        views.DEMO_IMAGES = [demo_img]
        views.DEFAULT_SOURCE = "demo"
        views._demo_index = 0

        try:
            response = self.client.get(reverse('capture_image'))
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.content)
            self.assertTrue(data["success"])
            self.assertTrue(data["filename"].endswith(".jpg"))
            self.assertEqual(data["analysis"][0]["name"], "apple scab")
            self.assertTrue((views.CAPTURES_DIR / data["filename"]).exists())
            self.assertEqual(len(views.capture_history), 1)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            views.capture_history = []
