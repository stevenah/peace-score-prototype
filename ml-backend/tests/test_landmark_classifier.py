"""Landmark classifiers (real + mock), the session layout lock and the
JSON-safe wire payload."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import LandmarkResult
from app.config import settings
from app.ml.landmarks.bench import synthetic_olympus_frame
from app.ml.landmarks.classifier import (
    INFERENCE_LOCK,
    LandmarkFrameOutput,
    LandmarkModelManager,
    RealLandmarkClassifier,
)
from app.ml.landmarks.layouts import OLYMPUS_1920
from app.ml.landmarks.mock import (
    CYCLE,
    DWELL,
    FLOW,
    MOCK_MODEL_VERSION,
    SEGMENT,
    MockLandmarkClassifier,
)
from app.ml.landmarks.payload import build_payload, error_payload, validate_output
from app.ml.landmarks.preprocess import canonicalize, to_model_input
from app.ml.landmarks.quality import QUALITY_SIZE, QualityParams, quality_metrics, quality_score
from app.ml.landmarks.session import LayoutLock
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from app.ml.landmarks.tracker import OBSERVED, StationTracker

BACKEND = Path(__file__).resolve().parents[1]
CLASSIFIED = {"ok", "uncertain", "low_quality"}


@pytest.fixture(scope="module")
def olympus():
    return synthetic_olympus_frame(1920, seed=3)


def _lit_16x9(width: int = 1920) -> np.ndarray:
    """Right shape, but no dark corner cuts or OSD panel: fails verification."""
    return np.full((width * 9 // 16, width, 3), 120, np.uint8)


def _real(bundle_dir, **kw) -> RealLandmarkClassifier:
    return RealLandmarkClassifier(LandmarkModelManager.get_instance(bundle_dir=bundle_dir, device="cpu"), **kw)


# -- RealLandmarkClassifier --------------------------------------------------------


def test_real_classifier_outputs_a_distribution(tiny_bundle, olympus):
    out = _real(tiny_bundle).predict(olympus)
    assert out.status in CLASSIFIED
    assert out.layout == OLYMPUS_1920.name and out.mode == "wl"
    assert len(out.probs) == NUM_STATIONS and abs(sum(out.probs) - 1.0) < 1e-6
    assert out.top == int(np.argmax(out.probs)) and out.confidence == out.probs[out.top]
    assert 0.0 < out.quality <= 1.0 and out.latency_ms > 0
    assert all(type(p) is float for p in out.probs)


def test_real_classifier_reads_the_mode_badge(tiny_bundle):
    out = _real(tiny_bundle).predict(synthetic_olympus_frame(badge="nbi"))
    assert out.mode == "nbi"


def test_status_follows_tau_and_quality_gate(make_tiny_bundle, olympus):
    low_tau = make_tiny_bundle("lm-0.0.2", tau=[0.01] * NUM_STATIONS)
    assert _real(low_tau).predict(olympus).status == "ok"
    LandmarkModelManager.reset()
    high_tau = make_tiny_bundle("lm-0.0.3", tau=[1.0] * NUM_STATIONS)
    assert _real(high_tau).predict(olympus).status == "uncertain"
    LandmarkModelManager.reset()
    strict_q = make_tiny_bundle("lm-0.0.4", tau=[0.01] * NUM_STATIONS,
                                quality={"sharp_ref": 1100.0, "q_min": 1.0})
    assert _real(strict_q).predict(olympus).status == "low_quality"


def test_temperature_flattens_probabilities(make_tiny_bundle, olympus):
    sharp = _real(make_tiny_bundle("lm-0.1.0", temperature=0.05)).predict(olympus)
    LandmarkModelManager.reset()
    flat = _real(make_tiny_bundle("lm-0.1.1", temperature=20.0)).predict(olympus)
    assert sharp.top == flat.top
    assert sharp.confidence > flat.confidence


def _quality_at(frame: np.ndarray, size: int, params: QualityParams) -> float:
    img = to_model_input(canonicalize(frame, OLYMPUS_1920), OLYMPUS_1920, size)
    return quality_score(quality_metrics(img, OLYMPUS_1920, params), params)


def test_quality_is_measured_at_the_calibration_size_for_any_input_size(make_tiny_bundle, olympus):
    params = QualityParams(sharp_ref=1e6)  # unclipped: quality is proportional to sharpness
    assert QUALITY_SIZE == 224  # sharp_ref and q_min are calibrated on 224 px cache metrics
    # The bias this guards against: var(Laplacian) depends on the resolution.
    assert abs(_quality_at(olympus, 288, params) / _quality_at(olympus, QUALITY_SIZE, params) - 1.0) > 0.05
    served = {}
    for i, size in enumerate((224, 288)):
        LandmarkModelManager.reset()
        bundle = make_tiny_bundle(f"lm-0.2.{i}", input_size=size, quality={**params.to_dict(), "q_min": 0.0})
        served[size] = _real(bundle).predict(olympus).quality
    assert served[224] == served[288] == pytest.approx(_quality_at(olympus, QUALITY_SIZE, params), abs=1e-12)


def test_serving_quality_equals_the_frame_cache_quality(make_tiny_bundle, olympus):
    """A 288 px bundle serves the quality calibration saw: the frame cache's
    224 px metrics (extract_cache.py) rescored with the shipped params."""
    pd = pytest.importorskip("pandas")
    from training.landmarks.dataset import recompute_quality

    canvas = canonicalize(olympus, OLYMPUS_1920)
    m = quality_metrics(to_model_input(canvas, OLYMPUS_1920, 224), OLYMPUS_1920, QualityParams())
    cached = pd.DataFrame([{"sharpness": round(m.sharpness, 2), "dark_frac": round(m.dark_frac, 4),
                            "sat_frac": round(m.sat_frac, 4), "redout": round(m.redout, 4)}])
    shipped = QualityParams(sharp_ref=5e4)
    bundle = make_tiny_bundle(input_size=288, quality={**shipped.to_dict(), "q_min": 0.0})
    out = _real(bundle).predict(olympus)
    assert 0.0 < out.quality < 1.0
    assert out.quality == pytest.approx(float(recompute_quality(cached, shipped)[0]), abs=1e-6)


def test_blank_aperture_is_low_quality(make_tiny_bundle):
    frame = synthetic_olympus_frame()
    mask = frame.max(axis=-1) > 0
    frame[mask] = (150, 80, 60)  # flat colour: no texture, sharpness 0
    frame[:, :540] = 0
    out = _real(make_tiny_bundle(tau=[0.01] * NUM_STATIONS)).predict(frame)
    assert out.status == "low_quality" and out.quality == 0.0 and out.probs is not None


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(np.full((480, 640, 3), 120, np.uint8), id="4:3"),
        pytest.param(synthetic_olympus_frame(960), id="below-min-width"),
        pytest.param(np.zeros((1080, 1350, 3), np.uint8), id="legacy-not-served"),
        pytest.param(_lit_16x9(), id="unverified-16:9"),
    ],
)
def test_unknown_layouts_are_unsupported(tiny_bundle, frame):
    out = _real(tiny_bundle).predict(frame)
    assert out.status == "unsupported_layout"
    assert out.probs is None and out.layout is None and out.top is None


def test_forward_holds_the_process_wide_inference_lock(tiny_bundle, olympus):
    clf = _real(tiny_bundle)
    done = threading.Event()
    with INFERENCE_LOCK:
        t = threading.Thread(target=lambda: (clf.predict(olympus), done.set()))
        t.start()
        time.sleep(0.2)
        assert not done.is_set()
    t.join(5)
    assert done.is_set()


# -- layout lock hysteresis ----------------------------------------------------------


def test_layout_lock_locks_after_k_and_releases_after_k_failures(olympus):
    lock = LayoutLock(lock_after=3, release_after=5)
    bad = _lit_16x9()
    assert lock.resolve(bad).layout is None  # nothing to fall back to yet
    for i in range(3):
        d = lock.resolve(olympus)
        assert d.verified and d.layout is OLYMPUS_1920
        assert (lock.locked is OLYMPUS_1920) == (i == 2)
    for _ in range(4):  # single weird frames keep the locked layout, unverified
        d = lock.resolve(bad)
        assert d.layout is OLYMPUS_1920 and not d.verified
    d = lock.resolve(bad)  # 5th consecutive failure: released
    assert d.layout is None and lock.locked is None


def test_layout_lock_failure_count_resets_on_a_verified_frame(olympus):
    lock = LayoutLock(lock_after=3, release_after=5)
    for _ in range(3):
        lock.resolve(olympus)
    bad = _lit_16x9()
    for _ in range(3):
        for _ in range(4):
            assert lock.resolve(bad).layout is OLYMPUS_1920
        assert lock.resolve(olympus).verified
    assert lock.locked is OLYMPUS_1920


def test_layout_lock_refuses_incompatible_frames_even_when_locked(olympus):
    lock = LayoutLock()
    for _ in range(3):
        lock.resolve(olympus)
    assert lock.resolve(np.full((480, 640, 3), 90, np.uint8)).layout is None


def test_classifier_caps_unverified_frames_at_low_quality(make_tiny_bundle, olympus):
    clf = _real(make_tiny_bundle(tau=[0.01] * NUM_STATIONS))
    for _ in range(3):
        assert clf.predict(olympus).status == "ok"
    out = clf.predict(_lit_16x9())
    assert out.status == "low_quality" and out.layout == OLYMPUS_1920.name and out.probs is not None


# -- mock ------------------------------------------------------------------------------


def test_mock_is_deterministic_and_well_formed():
    a, b = MockLandmarkClassifier(), MockLandmarkClassifier()
    for i in range(CYCLE + 5):
        x, y = a.frame_at(i), b.frame_at(i)
        assert x == y
        assert x.status in CLASSIFIED and abs(sum(x.probs) - 1.0) < 1e-9
        assert x.top == int(np.argmax(x.probs)) and x.confidence == x.probs[x.top]
    assert MockLandmarkClassifier(seed=1).frame_at(0) != a.frame_at(0)


def test_mock_follows_the_clinical_flow():
    m = MockLandmarkClassifier()
    for seg, key in enumerate(FLOW):
        dwell = [m.frame_at(seg * SEGMENT + k) for k in range(DWELL)]
        assert all(o.status == "ok" and STATION_ORDER[o.top] == key for o in dwell)
        transit = [m.frame_at(seg * SEGMENT + k).status for k in range(DWELL, SEGMENT)]
        assert set(transit) == {"low_quality", "uncertain"}
        assert dwell[0].mode == ("nbi" if seg % 2 else "wl")


def test_mock_through_the_tracker_observes_all_stations(olympus):
    m, tracker = MockLandmarkClassifier(), StationTracker()
    first_seen = {}
    for i in range(CYCLE):
        step = tracker.update(m.predict(olympus).evidence())
        for k in step.observed_events:
            first_seen.setdefault(k, i)
    assert set(first_seen) == set(STATION_ORDER)
    assert max(first_seen.values()) < CYCLE
    assert tracker.state().stations == [OBSERVED] * NUM_STATIONS


def test_mock_reports_unsupported_layouts():
    assert MockLandmarkClassifier().predict(np.full((480, 640, 3), 90, np.uint8)).status == "unsupported_layout"


def test_mock_never_touches_global_random(olympus):
    random.seed(123)
    state = random.getstate()
    m = MockLandmarkClassifier()
    for _ in range(20):
        m.predict(olympus)
    assert random.getstate() == state


def test_mock_requires_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "use_mock_models", False)
    with pytest.raises(RuntimeError):
        MockLandmarkClassifier()


def test_mock_path_never_imports_torch():
    code = (
        "import sys\n"
        "from app.ml.landmarks.mock import MockLandmarkClassifier\n"
        "import numpy as np\n"
        "m = MockLandmarkClassifier()\n"
        "m.predict(np.zeros((1080, 1920, 3), np.uint8))\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    env = {**os.environ, "PEACE_USE_MOCK_MODELS": "true"}
    r = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# -- payload ------------------------------------------------------------------------------


META = {"model_version": "lm-0.0.1"}


def _validate(block: dict) -> LandmarkResult:
    json.dumps(block, allow_nan=False)
    return LandmarkResult.model_validate(block)


def _ok(**overrides) -> LandmarkFrameOutput:
    probs = [0.01] * NUM_STATIONS
    probs[5] = 0.91
    fields = dict(status="ok", layout="olympus_1920x1080", mode="wl", probs=tuple(probs), top=5,
                  confidence=0.91, quality=0.8, latency_ms=12.3456)
    fields.update(overrides)
    return LandmarkFrameOutput(**fields)


def test_payload_classified_frame_matches_the_contract():
    tracker = StationTracker()
    out = _ok()
    for _ in range(3):
        step = tracker.update(out.evidence())
    block = build_payload(out, step, META, display=True)
    assert list(block) == ["schema", "model_version", "display", "status", "layout", "mode", "probs", "top",
                           "confidence", "quality", "current", "stations", "auto_enabled", "events",
                           "landmark_ms"]
    assert block["top"] == "antrum" and block["events"]["observed"] == ["antrum"]
    assert block["landmark_ms"] == 12.346 and block["probs"][5] == 0.91
    _validate(block)


@pytest.mark.parametrize("status", ["unsupported_layout", "skipped", "error"])
def test_payload_unclassified_statuses_omit_classification_fields(status):
    block = build_payload(LandmarkFrameOutput(status), StationTracker().state(), META, display=False)
    assert block["status"] == status
    assert not {"probs", "top", "confidence", "quality", "layout", "mode"} & set(block)
    _validate(block)


def test_payload_converts_numpy_scalars_to_plain_python():
    probs = np.full(NUM_STATIONS, 0.01, np.float32)
    probs[2] = np.float32(0.91)
    out = _ok(probs=tuple(probs), top=np.int64(2), confidence=np.float32(0.91), quality=np.float64(0.5),
              latency_ms=np.float32(3.0))
    block = build_payload(out, StationTracker().state(), META, display=np.bool_(True))
    assert block["status"] == "ok" and block["top"] == "z_line"
    for v in [*block["probs"], block["confidence"], block["quality"], block["landmark_ms"]]:
        assert type(v) is float
    assert type(block["display"]) is bool
    _validate(block)


BAD_OUTPUTS = [
    pytest.param(dict(probs=(float("nan"),) * NUM_STATIONS), id="nan-probs"),
    pytest.param(dict(confidence=float("inf")), id="inf-confidence"),
    pytest.param(dict(quality=np.float32("nan")), id="nan-quality"),
    pytest.param(dict(quality=1.2), id="quality-above-1"),
    pytest.param(dict(probs=(0.1,) * 9), id="short-probs"),
    pytest.param(dict(top=10), id="top-out-of-range"),
    pytest.param(dict(top=None), id="missing-top"),
    pytest.param(dict(mode="xray"), id="bad-mode"),
    pytest.param(dict(status="great"), id="bad-status"),
    pytest.param(dict(probs=None), id="missing-probs"),
    pytest.param(dict(latency_ms=-1.0), id="negative-latency"),
]


@pytest.mark.parametrize("bad", BAD_OUTPUTS)
def test_payload_faults_become_an_error_block(bad):
    tracker = StationTracker()
    step = tracker.update(_ok().evidence())
    block = build_payload(_ok(**bad), step, META, display=True)
    assert block["status"] == "error"
    assert block["stations"] == step.stations and block["model_version"] == "lm-0.0.1"
    _validate(block)


@pytest.mark.parametrize("bad", BAD_OUTPUTS)
def test_validate_output_rejects_what_the_payload_rejects(bad):
    with pytest.raises(ValueError):
        validate_output(_ok(**bad))


@pytest.mark.parametrize(
    "out",
    [_ok(), _ok(status="uncertain", mode="rdi"), LandmarkFrameOutput("unsupported_layout"),
     LandmarkFrameOutput("skipped"), LandmarkFrameOutput("error", latency_ms=2.0)],
    ids=["ok", "uncertain", "unsupported", "skipped", "error"],
)
def test_validate_output_accepts_what_the_payload_accepts(out):
    validate_output(out)
    assert build_payload(out, StationTracker().state(), META, display=False)["status"] == out.status


def test_error_payload_survives_a_broken_tracker_step():
    class Broken:
        current = "not-a-station"
        stations = ["?"]
        auto_enabled = []
        observed_events = []
        best_frame_events = []

    block = error_payload(Broken(), None, display=True, landmark_ms=float("nan"))
    assert block["status"] == "error" and block["stations"] == ["unseen"] * NUM_STATIONS
    assert block["landmark_ms"] == 0.0 and block["model_version"] == "unknown"
    _validate(block)


def test_station_literals_follow_station_order():
    from typing import get_args

    from app.api.schemas import StationKey

    assert get_args(StationKey) == STATION_ORDER
    assert STATION_INDEX["antrum"] == 5
    assert MOCK_MODEL_VERSION.startswith("mock-")


# -- per-session summary ------------------------------------------------------------------


def test_session_stats_summarise_and_never_raise():
    from app.ml.landmarks.session import SessionStats

    stats = SessionStats()
    tracker = StationTracker()
    stats.record_frame(40.0, build_payload(_ok(), tracker.update(_ok().evidence()), META, False))
    stats.record_frame(41.0, build_payload(LandmarkFrameOutput("skipped"), tracker.update(None), META, False))
    stats.record_frame(42.0, None)
    stats.record_frame("junk", {"landmark_ms": object()})  # type: ignore[arg-type]
    stats.record_error()
    s = stats.summary()
    assert s["frames"] == 5 and s["frame_errors"] == 1 and s["stats_errors"] == 1
    assert s["landmark_status"] == {"ok": 1, "skipped": 1}
    assert s["model_version"] == "lm-0.0.1" and s["layout"] == "olympus_1920x1080"
    assert s["landmark_ms_p50"] == 12.3  # skipped frames carry no latency
    json.dumps(s, allow_nan=False)


def test_channels_last_forward_matches_contiguous(tiny_bundle):
    import torch

    from app.ml.landmarks.bundle import load_bundle

    x = torch.randn(3, 224, 224, generator=torch.Generator().manual_seed(0))
    nchw = LandmarkModelManager(load_bundle(tiny_bundle))
    nhwc = LandmarkModelManager(load_bundle(tiny_bundle), channels_last=True)
    assert not nchw._channels_last and nhwc._channels_last  # CPU default is NCHW
    assert np.allclose(nchw.forward(x), nhwc.forward(x), rtol=1e-4, atol=1e-5)
