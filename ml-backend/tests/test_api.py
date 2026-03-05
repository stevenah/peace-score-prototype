"""Tests for the HTTP API endpoints."""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.api.schemas import AnalysisStatus


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create a minimal JPEG image in memory."""
    arr = np.random.randint(80, 200, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# -- Health ------------------------------------------------------------------


def test_health_returns_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("healthy", "degraded")
    assert body["version"]
    assert "models_loaded" in body


def test_health_reports_mock_mode(client):
    resp = client.get("/api/v1/health")
    body = resp.json()
    assert body["use_mock"] is True


# -- Analyze frame -----------------------------------------------------------


def test_analyze_frame_returns_score(client):
    jpeg = _make_jpeg_bytes()
    resp = client.post(
        "/api/v1/analyze/frame",
        files={"frame": ("frame.jpg", jpeg, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "peace_score" in body
    assert body["peace_score"]["score"] in (0, 1, 2, 3)
    assert 0.0 <= body["peace_score"]["confidence"] <= 1.0
    assert body["processing_time_ms"] > 0


def test_analyze_frame_with_previous_frame(client):
    frame1 = _make_jpeg_bytes()
    frame2 = _make_jpeg_bytes()
    resp = client.post(
        "/api/v1/analyze/frame",
        files={
            "frame": ("frame.jpg", frame2, "image/jpeg"),
            "previous_frame": ("prev.jpg", frame1, "image/jpeg"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["motion"] is not None
    assert body["motion"]["direction"] in ("insertion", "retraction", "stationary")


def test_analyze_frame_without_previous_has_stationary_motion(client):
    jpeg = _make_jpeg_bytes()
    resp = client.post(
        "/api/v1/analyze/frame",
        files={"frame": ("frame.jpg", jpeg, "image/jpeg")},
    )
    body = resp.json()
    if body["motion"]:
        assert body["motion"]["direction"] == "stationary"


# -- Analyze video (upload) --------------------------------------------------


def test_analyze_video_upload_accepted(client, upload_dir):
    """A valid video upload should return a queued job."""
    # Create a tiny but valid mp4-like payload (the endpoint accepts it by extension)
    fake_video = b"\x00" * 1024
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.upload_dir = upload_dir
        resp = client.post(
            "/api/v1/analyze/video",
            files={"file": ("test.mp4", fake_video, "video/mp4")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert "analysis_id" in body


def test_analyze_video_rejects_bad_type(client):
    resp = client.post(
        "/api/v1/analyze/video",
        files={"file": ("test.txt", b"not a video", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


# -- Get analysis ------------------------------------------------------------


def test_get_analysis_not_found(client):
    resp = client.get("/api/v1/analyze/nonexistent-id")
    assert resp.status_code == 404


def test_get_analysis_returns_job(client, store):
    """Create a job directly in the store, then fetch via API."""
    job_id = store.create_job("/tmp/fake.mp4")
    with patch("app.api.routes.job_store", store):
        resp = client.get(f"/api/v1/analyze/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis_id"] == job_id
    assert body["status"] == AnalysisStatus.QUEUED.value
