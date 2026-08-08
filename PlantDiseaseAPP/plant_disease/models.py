from django.db import models


class DetectionCapture(models.Model):
    """A saved detection snapshot from the dashboard, persisted in the
    project's SQLite database instead of in memory."""

    timestamp = models.CharField(max_length=32)
    filename = models.CharField(max_length=255)
    analysis = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"{self.filename} ({self.timestamp})"
