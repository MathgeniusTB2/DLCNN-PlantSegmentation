import json

from django.test import TestCase
from django.urls import reverse

from plant_disease import views


class DashboardTests(TestCase):
    """Smoke tests for the web dashboard (no camera or GPU required)."""

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
        views.webcam = None
        views.drone = None
        response = self.client.get(reverse('capture_image'))
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', json.loads(response.content))

    def test_demo_feed_reports_when_unconfigured(self):
        views.DEMO_IMAGES = []
        views.DEFAULT_SOURCE = 'webcam'
        response = self.client.get(reverse('video_feed'), {'source': 'demo'})
        self.assertContains(response, 'Demo mode is not configured')
