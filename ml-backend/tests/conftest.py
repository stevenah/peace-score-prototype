"""Shared test fixtures for the ml-backend test suite."""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.job_store import JobStore


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
    """FastAPI TestClient that skips the lifespan (no worker thread)."""
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def upload_dir(tmp_path):
    """Override the upload dir to a temp directory."""
    d = str(tmp_path / "uploads")
    os.makedirs(d, exist_ok=True)
    return d
