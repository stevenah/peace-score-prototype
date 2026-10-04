"""Tests for the HTTP API endpoints."""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
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
    assert body["motion"]["direction"] in ("insertion", "withdrawal", "stationary")


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


_VIDEO_ROUTE_REMOVED = pytest.mark.xfail(
    strict=True,
    reason="/api/v1/analyze/video was removed; uploads now go through /analyze/s3",
)


@_VIDEO_ROUTE_REMOVED
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


@_VIDEO_ROUTE_REMOVED
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


# -- Health: model and landmark load state -------------------------------------


def test_health_reports_landmarks_off_by_default(client):
    body = client.get("/api/v1/health").json()
    assert body["landmarks_loaded"] is False
    assert body["landmark_model_version"] is None
    assert body["models_loaded"] is True  # mock models


def test_health_reports_mock_landmarks(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "landmarks_enabled", True)
    body = client.get("/api/v1/health").json()
    assert body["landmarks_loaded"] is True
    assert body["landmark_model_version"] == "mock-esge10-1"


def test_health_reports_real_load_state(client, monkeypatch, tiny_bundle):
    from app.config import settings
    from app.ml.landmarks.classifier import LandmarkModelManager
    from app.ml.real_models import ModelManager

    monkeypatch.setattr(settings, "use_mock_models", False)
    monkeypatch.setattr(settings, "landmarks_enabled", True)
    monkeypatch.setattr(ModelManager, "_instance", None)
    body = client.get("/api/v1/health").json()
    assert body["models_loaded"] is False and body["landmarks_loaded"] is False

    LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")
    body = client.get("/api/v1/health").json()
    assert body["landmarks_loaded"] is True and body["landmark_model_version"] == "lm-0.0.1"


def test_lifespan_warms_up_before_starting_the_worker(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.config import settings

    order = []
    monkeypatch.setattr(settings, "warmup_models", True)
    monkeypatch.setattr(main_mod, "warm_up_models", lambda: order.append("warmup") or {})
    monkeypatch.setattr(main_mod.worker, "start", lambda: order.append("worker"))
    monkeypatch.setattr(main_mod.worker, "stop", lambda: None)
    with TestClient(main_mod.app):
        pass
    assert order == ["warmup", "worker"]
