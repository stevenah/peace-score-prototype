"""The model admin API (app/api/models.py): listing, switching, serving mode, evals."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.config import settings  # noqa: E402
from app.ml.landmarks import registry  # noqa: E402
from app.ml.landmarks.bundle import MODEL_FILE  # noqa: E402
from app.ml.landmarks.classifier import LandmarkModelManager  # noqa: E402

KEY = "test-key"


@pytest.fixture()
def real_models(monkeypatch):
    """Real landmark loading on CPU (conftest pins mock models)."""
    monkeypatch.setattr(settings, "use_mock_models", False)
    monkeypatch.setattr(settings, "device", "cpu")


def _state() -> dict:
    path = Path(settings.landmark_state_file)
    return json.loads(path.read_text()) if path.exists() else {}


# -- listing ---------------------------------------------------------------------


def test_models_lists_bundles_runs_and_what_is_served(client, make_tiny_bundle):
    make_tiny_bundle("lm-0.1.0")
    make_tiny_bundle("lm-0.2.0", train={"run": "fold0", "candidate": "effb0_in1k"})
    body = client.get("/api/v1/models").json()

    assert body["admin_enabled"] is True and len(body["stations"]) == 10
    assert [m["version"] for m in body["models"]] == ["lm-0.2.0", "lm-0.1.0"]
    assert body["models"][0]["training"]["fold"] == 0 and body["models"][0]["servable"]
    assert body["experiments"] == []
    assert body["serving"] == {
        "mode": "off", "mode_saved": False, "mock": True, "selected_version": None,
        "default": {"kind": "none", "version": None}, "loaded_version": None, "device": None, "failure": None,
    }
    kinds = {k["kind"]: k for k in body["eval_kinds"]}
    assert set(kinds) == {"selftest", "latency", "accuracy", "oof_metrics", "gates_report", "replay"}
    assert kinds["selftest"]["available"] and kinds["selftest"]["target"] == "bundle"
    assert kinds["replay"]["target"] == "experiment"


def test_the_default_bundle_is_reported(client, make_tiny_bundle, monkeypatch):
    monkeypatch.setattr(settings, "landmark_bundle_dir", str(make_tiny_bundle("lm-0.3.0")))
    assert client.get("/api/v1/models").json()["serving"]["default"] == {"kind": "env", "version": "lm-0.3.0"}
    monkeypatch.setattr(settings, "landmark_bundle_dir", "")
    Path(settings.landmark_bundle_lock).write_text(json.dumps(
        {"version": "lm-1.0.0", "s3_key": "models/landmarks/x.tar.gz", "sha256": "0" * 64}))
    assert client.get("/api/v1/models").json()["serving"]["default"] == {"kind": "lock", "version": "lm-1.0.0"}


# -- switching ---------------------------------------------------------------------


def test_activate_serves_the_bundle_and_saves_the_choice(client, make_tiny_bundle, real_models):
    first, second = make_tiny_bundle("lm-0.1.0", seed=1), make_tiny_bundle("lm-0.2.0", seed=2)

    serving = client.post("/api/v1/models/active", json={"version": "lm-0.1.0"}).json()
    assert serving["loaded_version"] == "lm-0.1.0" and serving["selected_version"] == "lm-0.1.0"
    assert serving["device"] == "cpu" and serving["failure"] is None
    assert _state() == {"version": "lm-0.1.0", "path": str(first.resolve())}

    serving = client.post("/api/v1/models/active", json={"version": "lm-0.2.0"}).json()
    assert serving["loaded_version"] == "lm-0.2.0"
    assert LandmarkModelManager.get_instance().version == "lm-0.2.0"
    assert _state()["path"] == str(second.resolve())

    # A restart loads the saved choice, not the (absent) default.
    LandmarkModelManager.reset()
    assert LandmarkModelManager.get_instance().version == "lm-0.2.0"


def test_health_reports_the_activated_model(client, make_tiny_bundle, real_models):
    make_tiny_bundle("lm-0.1.0")
    client.put("/api/v1/models/mode", json={"mode": "shadow"})
    client.post("/api/v1/models/active", json={"version": "lm-0.1.0"})
    health = client.get("/api/v1/health").json()
    assert health["landmarks_loaded"] is True and health["landmark_model_version"] == "lm-0.1.0"


def test_a_bundle_that_fails_to_load_changes_nothing(client, make_tiny_bundle, real_models):
    make_tiny_bundle("lm-0.1.0")
    bad = make_tiny_bundle("lm-0.2.0")
    with open(bad / MODEL_FILE, "ab") as f:
        f.write(b"x")
    client.post("/api/v1/models/active", json={"version": "lm-0.1.0"})

    resp = client.post("/api/v1/models/active", json={"version": "lm-0.2.0"})
    assert resp.status_code == 409 and "sha256 mismatch" in resp.json()["detail"]
    serving = client.get("/api/v1/models").json()["serving"]
    assert serving["loaded_version"] == "lm-0.1.0" and serving["selected_version"] == "lm-0.1.0"
    assert serving["failure"] is None


def test_activate_refuses_unknown_versions_and_mock_mode(client, make_tiny_bundle, monkeypatch):
    make_tiny_bundle("lm-0.1.0")
    mock = client.post("/api/v1/models/active", json={"version": "lm-0.1.0"})
    assert mock.status_code == 409 and "mock" in mock.json()["detail"]
    monkeypatch.setattr(settings, "use_mock_models", False)
    assert client.post("/api/v1/models/active", json={"version": "lm-9.9.9"}).status_code == 404
    assert client.post("/api/v1/models/active", json={}).status_code == 422
    assert _state() == {}


def test_clearing_the_choice_returns_to_the_default(client, make_tiny_bundle, real_models, monkeypatch):
    default, chosen = make_tiny_bundle("lm-0.1.0"), make_tiny_bundle("lm-0.2.0")
    monkeypatch.setattr(settings, "landmark_bundle_dir", str(default))
    client.put("/api/v1/models/mode", json={"mode": "display"})
    assert client.get("/api/v1/models").json()["serving"]["loaded_version"] == "lm-0.1.0"
    client.post("/api/v1/models/active", json={"version": chosen.name})

    serving = client.delete("/api/v1/models/active").json()
    assert serving["selected_version"] is None and serving["loaded_version"] == "lm-0.1.0"
    assert _state() == {"mode": "display"}  # the mode outlives the selection


def test_clearing_with_no_default_reports_why_nothing_is_served(client, make_tiny_bundle, real_models):
    make_tiny_bundle("lm-0.1.0")
    client.put("/api/v1/models/mode", json={"mode": "shadow"})
    client.post("/api/v1/models/active", json={"version": "lm-0.1.0"})
    serving = client.delete("/api/v1/models/active").json()
    assert serving["loaded_version"] is None and "no bundle lock" in serving["failure"]


# -- serving mode ----------------------------------------------------------------------


def test_mode_switches_the_landmark_flags_and_is_saved(client):
    serving = client.put("/api/v1/models/mode", json={"mode": "display"}).json()
    assert serving["mode"] == "display" and serving["mode_saved"] is True
    assert (settings.landmarks_enabled, settings.landmarks_display) == (True, True)
    assert _state() == {"mode": "display"}

    assert client.put("/api/v1/models/mode", json={"mode": "shadow"}).json()["mode"] == "shadow"
    assert (settings.landmarks_enabled, settings.landmarks_display) == (True, False)
    assert client.put("/api/v1/models/mode", json={"mode": "loud"}).status_code == 422
    assert client.put("/api/v1/models/mode", json={"mode": "off"}).json()["mode"] == "off"
    assert not settings.landmarks_enabled and _state() == {"mode": "off"}


def test_enabling_loads_the_selected_model(client, make_tiny_bundle, real_models):
    bundle = make_tiny_bundle("lm-0.1.0")
    registry.write_state(registry.ServingState("lm-0.1.0", str(bundle)))
    assert LandmarkModelManager.peek() is None
    serving = client.put("/api/v1/models/mode", json={"mode": "shadow"}).json()
    assert serving["loaded_version"] == "lm-0.1.0" and serving["failure"] is None


def test_startup_restores_the_saved_mode(make_tiny_bundle):
    from fastapi.testclient import TestClient

    from app.main import app

    registry.write_state(registry.ServingState(mode="shadow"))
    assert registry.current_mode() == "off"
    with TestClient(app) as c:
        assert c.get("/api/v1/models").json()["serving"]["mode"] == "shadow"
    assert settings.landmarks_enabled and not settings.landmarks_display


# -- access ----------------------------------------------------------------------------


def test_everything_needs_the_api_key(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", KEY)
    for method, path in (("get", "/api/v1/models"), ("get", "/api/v1/evals"), ("delete", "/api/v1/models/active"),
                         ("post", "/api/v1/evals"), ("get", "/api/v1/evals/20260101T000000-abcdef")):
        assert getattr(client, method)(path).status_code == 401, path
    assert client.get("/api/v1/models", headers={"x-api-key": KEY}).status_code == 200


def test_changes_are_refused_when_model_admin_is_off(client, make_tiny_bundle, monkeypatch):
    make_tiny_bundle("lm-0.1.0")
    registry.write_state(registry.ServingState(mode="display"))
    monkeypatch.setattr(settings, "model_admin", False)

    assert client.post("/api/v1/models/active", json={"version": "lm-0.1.0"}).status_code == 403
    assert client.delete("/api/v1/models/active").status_code == 403
    assert client.put("/api/v1/models/mode", json={"mode": "display"}).status_code == 403
    assert client.post("/api/v1/evals", json={"kind": "selftest", "target": "lm-0.1.0"}).status_code == 403
    assert client.post("/api/v1/evals/20260101T000000-abcdef/cancel").status_code == 403

    body = client.get("/api/v1/models").json()
    assert body["admin_enabled"] is False and [m["version"] for m in body["models"]] == ["lm-0.1.0"]
    assert body["serving"]["mode"] == "off" and body["serving"]["mode_saved"] is False  # the file is ignored
    assert client.get("/api/v1/evals").json() == {"jobs": []}


# -- evals -------------------------------------------------------------------------------


def test_an_eval_runs_from_the_api(client, make_tiny_bundle):
    make_tiny_bundle("lm-0.1.0")
    created = client.post("/api/v1/evals", json={"kind": "selftest", "target": "lm-0.1.0"})
    assert created.status_code == 201
    job_id = created.json()["id"]

    deadline = time.monotonic() + 60
    job = client.get(f"/api/v1/evals/{job_id}").json()
    while job["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.1)
        job = client.get(f"/api/v1/evals/{job_id}").json()
    assert job["status"] == "succeeded", job
    assert job["result"]["ok"] is True and job["verdict"] == "pass"
    assert [j["id"] for j in client.get("/api/v1/evals").json()["jobs"]] == [job_id]
    assert Path(settings.eval_dir, job_id, "result.json").is_file()


def test_eval_requests_are_checked(client, make_tiny_bundle):
    make_tiny_bundle("lm-0.1.0")
    assert client.post("/api/v1/evals", json={"kind": "nope", "target": "lm-0.1.0"}).status_code == 404
    assert client.post("/api/v1/evals", json={"kind": "selftest", "target": "lm-9.9.9"}).status_code == 404
    bad = client.post("/api/v1/evals", json={"kind": "latency", "target": "lm-0.1.0", "params": {"frames": 1}})
    assert bad.status_code == 422 and "frames" in bad.json()["detail"]
    assert client.post("/api/v1/evals", json={"kind": "selftest"}).status_code == 422
    assert client.get("/api/v1/evals/nope").status_code == 404
    assert client.post("/api/v1/evals/nope/cancel").status_code == 404
    assert client.get("/api/v1/evals").json() == {"jobs": []}
