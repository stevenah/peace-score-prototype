"""Live websocket with the landmark feature: wire contract, isolation of
PEACE from landmark faults, and the live-socket cap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from pydantic import ConfigDict
from starlette.websockets import WebSocketDisconnect

from app.api import websocket as ws_mod
from app.api.schemas import LandmarkEvents, LandmarkResult, LiveFrameResult
from app.config import settings
from app.ml.landmarks.bench import encode_jpeg, synthetic_olympus_frame
from app.ml.landmarks.classifier import (
    BaseLandmarkClassifier,
    LandmarkFrameOutput,
    LandmarkModelManager,
    RealLandmarkClassifier,
)
from app.ml.landmarks.stations import NUM_STATIONS, STATION_ORDER
from app.ml.landmarks.tracker import StationTracker
from app.ml.mock_models import MockMotionDetector, MockPEACEClassifier, MockRegionDetector
from app.ml.pipeline import AnalysisPipeline, FrameResult

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
WS = "/api/v1/ws/live"


# Strict mirrors of the wire schema: unknown keys fail validation.
class StrictEvents(LandmarkEvents):
    model_config = ConfigDict(extra="forbid")


class StrictLandmark(LandmarkResult):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    events: StrictEvents


class StrictFrame(LiveFrameResult):
    model_config = ConfigDict(extra="forbid")
    landmark: Optional[StrictLandmark] = None


def _no_nan(text: str) -> dict:
    def refuse(c):
        raise ValueError(f"non-JSON constant {c}")

    return json.loads(text, parse_constant=refuse)


@pytest.fixture(scope="module")
def olympus_jpeg() -> bytes:
    return encode_jpeg(synthetic_olympus_frame(1920, seed=2))


@pytest.fixture(scope="module")
def small_jpeg() -> bytes:
    return encode_jpeg(np.random.RandomState(0).randint(80, 200, (224, 224, 3), dtype=np.uint8))


def _v1_keys() -> set[str]:
    return set(json.loads((CONTRACTS / "live_frame_result.v1.example.json").read_text()))


def _exchange(ws, data: bytes) -> dict:
    ws.send_bytes(data)
    msg = _no_nan(ws.receive_text())
    return msg


def _enable(monkeypatch, display: bool = False, every_n: int = 1):
    monkeypatch.setattr(settings, "landmarks_enabled", True)
    monkeypatch.setattr(settings, "landmarks_display", display)
    monkeypatch.setattr(settings, "landmark_every_n", every_n)


def _pipeline_factory(monkeypatch, make_clf, meta_version: str):
    """Serve sockets from mock PEACE + the given landmark classifier."""

    def create(with_landmarks: bool = False):
        pipe = AnalysisPipeline(MockPEACEClassifier(), MockMotionDetector(), MockRegionDetector())
        clf = make_clf()
        pipe.landmark_classifier = clf
        pipe.station_tracker = StationTracker(clf.tracker_params, clf.auto_enabled)
        pipe.landmark_meta = {"model_version": meta_version}
        return pipe

    monkeypatch.setattr(ws_mod, "create_pipeline", create)


# -- contract fixtures -------------------------------------------------------------


@pytest.mark.skipif(not CONTRACTS.exists(), reason="contracts/ lives at the monorepo root")
def test_contract_fixtures_validate_strictly():
    v1 = json.loads((CONTRACTS / "live_frame_result.v1.example.json").read_text())
    v2 = json.loads((CONTRACTS / "live_frame_result.v2.example.json").read_text())
    assert StrictFrame.model_validate(v1).landmark is None
    lm = StrictFrame.model_validate(v2).landmark
    assert lm is not None and lm.schema_ == "esge10.v1"
    assert set(v2) - set(v1) == {"landmark"}


# -- feature off -------------------------------------------------------------------


@pytest.mark.skipif(not CONTRACTS.exists(), reason="contracts/ lives at the monorepo root")
def test_flag_off_key_set_is_identical_to_v1(client, olympus_jpeg):
    with client.websocket_connect(WS) as ws:
        for _ in range(3):
            msg = _exchange(ws, olympus_jpeg)
            assert set(msg) == _v1_keys()
            StrictFrame.model_validate(msg)


# -- feature on, mock model ----------------------------------------------------------


@pytest.mark.skipif(not CONTRACTS.exists(), reason="contracts/ lives at the monorepo root")
def test_mock_stream_validates_and_observes_all_ten_stations(client, monkeypatch, olympus_jpeg):
    _enable(monkeypatch)
    observed_events: list[str] = []
    statuses = set()
    with client.websocket_connect(WS) as ws:
        for i in range(170):
            msg = _exchange(ws, olympus_jpeg)
            assert set(msg) == _v1_keys() | {"landmark"}
            frame = StrictFrame.model_validate(msg)
            lm = frame.landmark
            assert lm.model_version == "mock-esge10-1" and lm.display is False
            assert msg["frame_index"] == i
            statuses.add(lm.status)
            observed_events += lm.events.observed
        assert lm.stations == ["observed"] * NUM_STATIONS
    assert sorted(observed_events) == sorted(STATION_ORDER)  # each fires exactly once
    assert {"ok", "uncertain", "low_quality"} <= statuses


def test_display_flag_and_every_n_reach_the_wire(client, monkeypatch, olympus_jpeg):
    _enable(monkeypatch, display=True, every_n=2)
    with client.websocket_connect(WS) as ws:
        blocks = [StrictFrame.model_validate(_exchange(ws, olympus_jpeg)).landmark for _ in range(6)]
    assert all(b.display for b in blocks)
    assert [b.status == "skipped" for b in blocks] == [False, True] * 3


def test_non_processor_frames_report_unsupported_layout(client, monkeypatch, small_jpeg):
    _enable(monkeypatch)
    with client.websocket_connect(WS) as ws:
        msg = _exchange(ws, small_jpeg)
    lm = StrictFrame.model_validate(msg).landmark
    assert lm.status == "unsupported_layout" and lm.probs is None
    assert msg["peace_score"]["score"] in (0, 1, 2, 3)


# -- feature on, real model (tiny bundle) --------------------------------------------------


def test_real_tiny_bundle_over_the_socket(client, monkeypatch, tiny_bundle, olympus_jpeg):
    manager = LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")
    _pipeline_factory(monkeypatch, lambda: RealLandmarkClassifier(manager), manager.version)
    with client.websocket_connect(WS) as ws:
        for _ in range(5):
            lm = StrictFrame.model_validate(_exchange(ws, olympus_jpeg)).landmark
            assert lm.model_version == "lm-0.0.1"
            assert lm.status in ("ok", "uncertain", "low_quality")
            assert lm.layout == "olympus_1920x1080" and abs(sum(lm.probs) - 1.0) < 1e-3


# -- faults never cost PEACE or the socket ------------------------------------------------


class _Raising(BaseLandmarkClassifier):
    model_version = "lm-broken"

    def predict(self, full_frame):
        raise RuntimeError("boom")


class _NaN(BaseLandmarkClassifier):
    model_version = "lm-nan"

    def predict(self, full_frame):
        return LandmarkFrameOutput("ok", "olympus_1920x1080", "wl", (float("nan"),) * 10, 0, float("nan"), 0.5)


class _Numpy(BaseLandmarkClassifier):
    model_version = "lm-numpy"

    def predict(self, full_frame):
        p = np.full(10, 0.05, np.float32)
        p[4] = np.float32(0.55)
        return LandmarkFrameOutput("ok", "olympus_1920x1080", "nbi", tuple(p), np.int64(4), p[4],
                                   np.float64(0.7), np.float32(4.2))


@pytest.mark.parametrize("clf, status", [(_Raising, "error"), (_NaN, "error"), (_Numpy, "ok")],
                         ids=["raises", "nan", "numpy-scalars"])
def test_landmark_faults_leave_peace_and_the_socket_intact(client, monkeypatch, olympus_jpeg, clf, status):
    _pipeline_factory(monkeypatch, clf, clf.model_version)
    with client.websocket_connect(WS) as ws:
        for i in range(4):
            msg = _exchange(ws, olympus_jpeg)  # strict JSON: no NaN on the wire
            assert msg["type"] == "frame_result" and msg["frame_index"] == i
            assert msg["peace_score"]["score"] in (0, 1, 2, 3)
            lm = StrictFrame.model_validate(msg).landmark
            assert lm.status == status and lm.model_version == clf.model_version
            if status == "error":  # rejected frames never tick a station
                assert lm.stations == ["unseen"] * NUM_STATIONS and lm.current is None
                assert lm.events.observed == [] and lm.events.best_frame == []


def test_bundle_with_unservable_tracker_meta_leaves_peace_on_the_socket(
    client, monkeypatch, make_tiny_bundle, olympus_jpeg
):
    """A JSON float tracker count used to pass load and warm-up, then kill
    every socket before its first PEACE result. It is now refused at load."""
    import app.ml.pipeline as pipeline_mod

    _enable(monkeypatch)
    monkeypatch.setattr(settings, "use_mock_models", False)
    monkeypatch.setattr(settings, "device", "cpu")
    bundle = make_tiny_bundle(tracker={"evidence_needed": 3, "evidence_window": 6.0})
    monkeypatch.setattr(settings, "landmark_bundle_dir", str(bundle))
    # Real landmark attach path, mock PEACE (no 70 MB checkpoint).
    monkeypatch.setattr(pipeline_mod, "_create_peace_pipeline", lambda: AnalysisPipeline(
        MockPEACEClassifier(), MockMotionDetector(), MockRegionDetector()))
    with client.websocket_connect(WS) as ws:
        for i in range(2):
            msg = _exchange(ws, olympus_jpeg)
            assert msg["type"] == "frame_result" and msg["frame_index"] == i
            assert msg["peace_score"]["score"] in (0, 1, 2, 3) and "landmark" not in msg
    assert client.get("/api/v1/health").json()["landmarks_loaded"] is False


@pytest.mark.parametrize("bad", [{"x": object()}, {"status": float("nan")}], ids=["unserialisable", "nan"])
def test_unsafe_landmark_dict_is_dropped_not_fatal(client, monkeypatch, olympus_jpeg, bad):
    class Leaky(AnalysisPipeline):
        def analyze_frame(self, *args, **kwargs) -> FrameResult:
            result = super().analyze_frame(*args, **kwargs)
            result.landmark = dict(bad)
            return result

    monkeypatch.setattr(ws_mod, "create_pipeline", lambda with_landmarks=False: Leaky(
        MockPEACEClassifier(), MockMotionDetector(), MockRegionDetector()))
    with client.websocket_connect(WS) as ws:
        for _ in range(2):
            msg = _exchange(ws, olympus_jpeg)
            assert msg["type"] == "frame_result" and "landmark" not in msg


def test_garbage_bytes_get_an_error_and_the_socket_stays_open(client, monkeypatch, olympus_jpeg):
    _enable(monkeypatch)
    with client.websocket_connect(WS) as ws:
        err = _exchange(ws, b"definitely not a jpeg")
        assert err == {"type": "error", "frame_index": 0, "message": "Frame could not be analysed"}
        msg = _exchange(ws, olympus_jpeg)
        assert msg["type"] == "frame_result" and msg["frame_index"] == 1
        StrictFrame.model_validate(msg)


# -- admission -----------------------------------------------------------------------------


def test_live_socket_cap_is_enforced_and_released(client, monkeypatch, olympus_jpeg):
    monkeypatch.setattr(settings, "max_live_sockets", 1)
    with client.websocket_connect(WS) as first:
        assert _exchange(first, olympus_jpeg)["type"] == "frame_result"
        with client.websocket_connect(WS) as second:
            with pytest.raises(WebSocketDisconnect) as exc:
                second.receive_text()
            assert exc.value.code == ws_mod.WS_TRY_AGAIN_LATER
            assert "Too many" in exc.value.reason
        assert _exchange(first, olympus_jpeg)["type"] == "frame_result"
    # The slot is released on disconnect.
    with client.websocket_connect(WS) as again:
        assert _exchange(again, olympus_jpeg)["type"] == "frame_result"
    assert ws_mod.live_socket_count() == 0


def test_session_summary_is_logged_at_disconnect(client, monkeypatch, olympus_jpeg, caplog):
    _enable(monkeypatch)
    caplog.set_level("INFO", logger="app.api.websocket")
    with client.websocket_connect(WS) as ws:
        for _ in range(4):
            _exchange(ws, olympus_jpeg)
    lines = [r.getMessage() for r in caplog.records if "Live session summary" in r.getMessage()]
    assert lines, "no session summary logged"
    summary = json.loads(lines[-1].split("Live session summary ", 1)[1])
    assert summary["frames"] == 4 and summary["model_version"] == "mock-esge10-1"
    assert summary["landmark_status"] == {"ok": 4} and summary["landmark_ms_p95"] is not None
    assert summary["layout"] == "olympus_1920x1080"
