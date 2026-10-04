"""Shared test fixtures for the ml-backend test suite."""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.services.job_store import JobStore


@pytest.fixture(autouse=True)
def _pin_settings(monkeypatch):
    """Run every test against mock models, whatever the local .env says.

    `use_mock_models` defaults to False, so without this the suite would load
    the real 70 MB checkpoint and its results would depend on the developer's
    environment. Tests that need real models opt in by overriding this.
    """
    monkeypatch.setattr(settings, "use_mock_models", True)


@pytest.fixture(autouse=True)
def _pin_landmark_settings(monkeypatch, tmp_path):
    """Landmarks off, no model warm-up, and no path to S3 unless a test opts in.

    The lifespan (entered by the ``client`` fixture) would otherwise warm up
    real models, and a committed BUNDLE.lock plus a developer's .env AWS keys
    could make a test download a bundle. The landmark singleton is reset so a
    cached load failure never leaks between tests.
    """
    from app.ml.landmarks.classifier import LandmarkModelManager

    monkeypatch.setattr(settings, "warmup_models", False)
    monkeypatch.setattr(settings, "landmarks_enabled", False)
    monkeypatch.setattr(settings, "landmarks_display", False)
    monkeypatch.setattr(settings, "landmark_bundle_lock", str(tmp_path / "no-BUNDLE.lock"))
    monkeypatch.setattr(settings, "landmark_bundle_dir", "")
    monkeypatch.setattr(settings, "landmark_cache_dir", str(tmp_path / "landmark-cache"))
    LandmarkModelManager.reset()
    yield
    LandmarkModelManager.reset()


@pytest.fixture()
def tmp_db(tmp_path):
    """Provide a path for a temporary SQLite database."""
    return str(tmp_path / "test_jobs.db")


@pytest.fixture()
def store(tmp_db):
    """Provide a fresh JobStore backed by a temp database."""
    return JobStore(db_path=tmp_db)


@pytest.fixture()
def sample_frame():
    """A 224x224 RGB frame with moderate brightness/contrast."""
    rng = np.random.RandomState(42)
    return rng.randint(80, 200, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def dark_frame():
    """A dark 224x224 frame (low brightness)."""
    rng = np.random.RandomState(0)
    return rng.randint(0, 40, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def bright_frame():
    """A bright 224x224 frame (high brightness + sufficient contrast).

    The mock classifier needs mean>=170 AND std>=40 for high scores.
    """
    rng = np.random.RandomState(1)
    return rng.randint(100, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def client():
    """FastAPI TestClient. Entering the context runs the app lifespan (worker thread)."""
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def upload_dir(tmp_path):
    """Override the upload dir to a temp directory."""
    d = str(tmp_path / "uploads")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture()
def make_tiny_bundle(tmp_path):
    """Factory for verified landmark bundles around the 2-layer test CNN.

    ``make_tiny_bundle(version="lm-0.0.1", seed=0, **meta_fields)`` writes a
    bundle under tmp_path and returns its directory. Weights are seeded
    without touching torch's global RNG state.
    """
    torch = pytest.importorskip("torch")
    from app.ml.landmarks.architectures import build
    from app.ml.landmarks.bundle import write_bundle

    def make(version: str = "lm-0.0.1", seed: int = 0, **meta_fields):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = build("tiny_test_cnn")
        fields = {"version": version, "input_size": 224, **meta_fields}
        return write_bundle(tmp_path / "bundles" / f"{version}-{seed}" / version, model, fields)

    return make


@pytest.fixture()
def tiny_bundle(make_tiny_bundle):
    return make_tiny_bundle()
