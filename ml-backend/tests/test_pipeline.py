"""Tests for the ML pipeline and mock models."""

from __future__ import annotations

import numpy as np

from app.ml.mock_models import (
    MockMotionDetector,
    MockPEACEClassifier,
    MockRegionDetector,
    SCORE_LABELS,
)
from app.ml.pipeline import AnalysisPipeline, FrameResult


# -- MockPEACEClassifier -----------------------------------------------------


class TestMockPEACEClassifier:
    def test_returns_valid_score(self, sample_frame):
        clf = MockPEACEClassifier()
        result = clf.predict(sample_frame)
        assert result["score"] in (0, 1, 2, 3)
        assert result["label"] in SCORE_LABELS.values()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_dark_frame_scores_low(self, dark_frame):
        clf = MockPEACEClassifier()
        scores = [clf.predict(dark_frame)["score"] for _ in range(20)]
        # Most predictions for a very dark frame should be 0 or 1
        assert sum(1 for s in scores if s <= 1) >= 15

    def test_bright_frame_scores_high(self, bright_frame):
        clf = MockPEACEClassifier()
        scores = [clf.predict(bright_frame)["score"] for _ in range(20)]
        assert sum(1 for s in scores if s >= 2) >= 15


# -- MockMotionDetector ------------------------------------------------------


class TestMockMotionDetector:
    def test_no_previous_frame_is_stationary(self, sample_frame):
        det = MockMotionDetector()
        result = det.detect(sample_frame, None)
        assert result["direction"] == "stationary"
        assert result["optical_flow_magnitude"] == 0.0

    def test_identical_frames_are_stationary(self, sample_frame):
        det = MockMotionDetector()
        result = det.detect(sample_frame, sample_frame.copy())
        assert result["direction"] == "stationary"

    def test_different_frames_have_motion(self, sample_frame, bright_frame):
        det = MockMotionDetector()
        result = det.detect(bright_frame, sample_frame)
        assert result["optical_flow_magnitude"] > 0
        assert result["direction"] in ("insertion", "withdrawal", "stationary")
        assert 0.0 <= result["confidence"] <= 1.0


# -- MockRegionDetector ------------------------------------------------------


class TestMockRegionDetector:
    def test_zero_duration_defaults_to_stomach(self, sample_frame):
        det = MockRegionDetector()
        assert det.detect(sample_frame, 0.0, 0.0) == "stomach"

    def test_early_timestamp_is_esophagus(self, sample_frame):
        det = MockRegionDetector()
        assert det.detect(sample_frame, 1.0, 60.0) == "esophagus"

    def test_mid_timestamp_is_stomach(self, sample_frame):
        det = MockRegionDetector()
        assert det.detect(sample_frame, 25.0, 60.0) == "stomach"

    def test_late_timestamp_is_duodenum(self, sample_frame):
        det = MockRegionDetector()
        assert det.detect(sample_frame, 50.0, 60.0) == "duodenum"


# -- AnalysisPipeline --------------------------------------------------------


class TestAnalysisPipeline:
    def _make_pipeline(self) -> AnalysisPipeline:
        return AnalysisPipeline(
            classifier=MockPEACEClassifier(),
            motion_detector=MockMotionDetector(),
            region_detector=MockRegionDetector(),
        )

    def test_analyze_single_frame(self, sample_frame):
        pipe = self._make_pipeline()
        result = pipe.analyze_frame(sample_frame, timestamp=5.0, total_duration=60.0)
        assert isinstance(result, FrameResult)
        assert result.peace_score["score"] in (0, 1, 2, 3)
        assert result.region in ("esophagus", "stomach", "duodenum")

    def test_analyze_frames_returns_list(self, sample_frame):
        pipe = self._make_pipeline()
        frames = [
            (0, 0.0, sample_frame),
            (1, 0.5, sample_frame),
            (2, 1.0, sample_frame),
        ]
        results = pipe.analyze_frames(frames, total_duration=60.0)
        assert len(results) == 3
        assert all(isinstance(r, FrameResult) for r in results)

    def test_aggregate_empty_results(self):
        pipe = self._make_pipeline()
        agg = pipe.aggregate_results([], total_duration=10.0)
        assert agg["timeline"] == []
        assert agg["motion_analysis"]["segments"] == []
        assert agg["peace_scores"]["overall"]["score"] == 0

    def test_aggregate_results_structure(self, sample_frame):
        pipe = self._make_pipeline()
        frames = [
            (0, 0.0, sample_frame),
            (30, 1.0, sample_frame),
            (60, 2.0, sample_frame),
        ]
        results = pipe.analyze_frames(frames, total_duration=60.0)
        agg = pipe.aggregate_results(results, total_duration=60.0)

        assert "motion_analysis" in agg
        assert "segments" in agg["motion_analysis"]
        assert "peace_scores" in agg
        assert "overall" in agg["peace_scores"]
        assert "by_region" in agg["peace_scores"]
        assert "timeline" in agg
        assert len(agg["timeline"]) == 3

    def test_aggregate_overall_score_is_mean(self, sample_frame):
        """The overall PEACE score is the mean across all frames, kept as a decimal."""
        pipe = self._make_pipeline()
        # Create frame results directly to control scores
        results = [
            FrameResult(
                frame_index=0,
                timestamp=0.0,
                peace_score={"score": 3, "label": "Excellent", "confidence": 0.9},
                motion={"direction": "stationary", "confidence": 0.8, "optical_flow_magnitude": 0.0},
                region="stomach",
            ),
            FrameResult(
                frame_index=1,
                timestamp=1.0,
                peace_score={"score": 1, "label": "Inadequate", "confidence": 0.7},
                motion={"direction": "stationary", "confidence": 0.8, "optical_flow_magnitude": 0.0},
                region="stomach",
            ),
        ]
        agg = pipe.aggregate_results(results, total_duration=10.0)
        assert agg["peace_scores"]["overall"]["score"] == 2.0

    def test_timeline_entries_have_required_fields(self, sample_frame):
        pipe = self._make_pipeline()
        results = pipe.analyze_frames(
            [(0, 0.0, sample_frame)], total_duration=60.0
        )
        agg = pipe.aggregate_results(results, total_duration=60.0)
        entry = agg["timeline"][0]
        assert "timestamp" in entry
        assert "frame_index" in entry
        assert "motion" in entry
        assert "region" in entry
        assert "peace_score" in entry
        assert "confidence" in entry


# -- Landmarks (ESGE stations) -------------------------------------------------
# The landmark step is additive: PEACE, motion and region must be identical
# whether it runs or not, and a landmark fault must never reach them.

import random  # noqa: E402

import pytest  # noqa: E402

from app.config import settings  # noqa: E402
from app.ml import pipeline as pipeline_mod  # noqa: E402
from app.ml.landmarks.bench import synthetic_olympus_frame  # noqa: E402
from app.ml.landmarks.classifier import (  # noqa: E402
    BaseLandmarkClassifier,
    LandmarkFrameOutput,
    LandmarkModelManager,
    RealLandmarkClassifier,
)
from app.ml.landmarks.mock import MockLandmarkClassifier  # noqa: E402
from app.ml.landmarks.tracker import StationTracker, TrackerParams  # noqa: E402
from app.ml.pipeline import create_pipeline, warm_up_models  # noqa: E402


def _peace_view(r: FrameResult) -> tuple:
    return (r.frame_index, r.timestamp, r.peace_score, r.motion, r.region)


def _stream(n: int = 12) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(7)
    full = synthetic_olympus_frame(1920, seed=5)
    return [(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8), full) for _ in range(n)]


def _run(pipe: AnalysisPipeline, frames, with_full: bool) -> list[FrameResult]:
    # The mock PEACE model draws from the global RNGs: seed them identically.
    random.seed(1234)
    np.random.seed(1234)
    out, prev = [], None
    for i, (small, full) in enumerate(frames):
        out.append(pipe.analyze_frame(small, prev, timestamp=i * 0.5, total_duration=10.0,
                                      frame_index=i, full_frame=full if with_full else None))
        prev = small
    return out


def _mock_pipe(**landmarks) -> AnalysisPipeline:
    return AnalysisPipeline(
        classifier=MockPEACEClassifier(),
        motion_detector=MockMotionDetector(),
        region_detector=MockRegionDetector(),
        **landmarks,
    )


class _Raising(BaseLandmarkClassifier):
    model_version = "lm-broken"

    def predict(self, full_frame):
        raise RuntimeError("boom")


class _NaN(BaseLandmarkClassifier):
    model_version = "lm-nan"

    def predict(self, full_frame):
        return LandmarkFrameOutput("ok", "olympus_1920x1080", "wl", (float("nan"),) * 10, 0, float("nan"), 0.5)


class _OutOfRange(BaseLandmarkClassifier):
    """Finite and confident, but quality 1.2: the wire rejects it."""

    model_version = "lm-range"

    def predict(self, full_frame):
        probs = [0.01] * 10
        probs[0] = 0.91
        return LandmarkFrameOutput("ok", "olympus_1920x1080", "wl", tuple(probs), 0, 0.91, 1.2)


class TestLandmarkPipeline:
    def test_peace_outputs_identical_with_mock_landmarks_on_and_off(self):
        frames = _stream()
        off = _run(_mock_pipe(), frames, with_full=True)
        on = _run(_mock_pipe(landmark_classifier=MockLandmarkClassifier(),
                             station_tracker=StationTracker()), frames, with_full=True)
        assert [_peace_view(r) for r in on] == [_peace_view(r) for r in off]
        assert all(r.landmark is None for r in off)
        assert all(r.landmark is not None for r in on)

    def test_peace_outputs_identical_with_real_landmarks_on_and_off(self, tiny_bundle):
        frames = _stream(6)
        manager = LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")
        off = _run(_mock_pipe(), frames, with_full=False)
        on = _run(_mock_pipe(landmark_classifier=RealLandmarkClassifier(manager),
                             station_tracker=StationTracker()), frames, with_full=True)
        assert [_peace_view(r) for r in on] == [_peace_view(r) for r in off]
        assert {r.landmark["status"] for r in on} <= {"ok", "uncertain", "low_quality"}

    def test_no_full_frame_means_no_landmark(self, sample_frame):
        pipe = _mock_pipe(landmark_classifier=MockLandmarkClassifier(), station_tracker=StationTracker())
        assert pipe.analyze_frame(sample_frame).landmark is None

    def test_every_n_skips_frames_without_feeding_the_tracker(self):
        tracker = StationTracker()
        pipe = _mock_pipe(landmark_classifier=MockLandmarkClassifier(), station_tracker=tracker,
                          landmark_every_n=2)
        results = _run(pipe, _stream(8), with_full=True)
        statuses = [r.landmark["status"] for r in results]
        assert statuses[1::2] == ["skipped"] * 4
        assert "skipped" not in statuses[0::2]
        assert all("probs" not in r.landmark for r in results[1::2])
        assert tracker.state_dict()["n"] == 4  # only evaluated frames count

    @pytest.mark.parametrize("clf", [_Raising(), _NaN(), _OutOfRange()], ids=["raises", "nan", "out-of-range"])
    def test_landmark_faults_become_error_payloads_and_spare_peace(self, clf):
        frames = _stream(4)
        off = _run(_mock_pipe(), frames, with_full=True)
        tracker = StationTracker()
        on = _run(_mock_pipe(landmark_classifier=clf, station_tracker=tracker,
                             landmark_meta={"model_version": clf.model_version}), frames, with_full=True)
        assert [_peace_view(r) for r in on] == [_peace_view(r) for r in off]
        for r in on:
            assert r.landmark["status"] == "error"
            assert r.landmark["model_version"] == clf.model_version
            # A frame the wire rejects is never evidence: nothing ticks inside error blocks.
            assert r.landmark["stations"] == ["unseen"] * 10 and r.landmark["current"] is None
            assert r.landmark["events"] == {"observed": [], "best_frame": []}
        assert tracker.state_dict()["n"] == 4  # errors are evaluated, no evidence
        assert tracker.state_dict()["evidence"] == [None] * 4


class TestCreatePipelineLandmarks:
    def test_off_by_default(self):
        assert create_pipeline(with_landmarks=True).landmark_classifier is None

    def test_only_the_live_socket_asks_for_landmarks(self, monkeypatch):
        monkeypatch.setattr(settings, "landmarks_enabled", True)
        assert create_pipeline().landmark_classifier is None  # batch, /analyze/frame

    def test_mock_mode_attaches_the_mock_with_a_fresh_tracker(self, monkeypatch):
        monkeypatch.setattr(settings, "landmarks_enabled", True)
        monkeypatch.setattr(settings, "landmarks_display", True)
        monkeypatch.setattr(settings, "landmark_every_n", 2)
        a, b = create_pipeline(with_landmarks=True), create_pipeline(with_landmarks=True)
        assert isinstance(a.landmark_classifier, MockLandmarkClassifier)
        assert a.station_tracker is not b.station_tracker
        assert a.landmark_classifier is not b.landmark_classifier
        assert a.landmark_display is True and a.landmark_every_n == 2
        assert a.landmark_meta == {"model_version": "mock-esge10-1"}

    def _real_mode(self, monkeypatch):
        # Real landmark path, but keep PEACE on the mock (no 70 MB checkpoint).
        monkeypatch.setattr(settings, "use_mock_models", False)
        monkeypatch.setattr(settings, "landmarks_enabled", True)
        monkeypatch.setattr(pipeline_mod, "_create_peace_pipeline", _mock_pipe)

    def test_real_mode_uses_bundle_tracker_params(self, monkeypatch, make_tiny_bundle):
        self._real_mode(monkeypatch)
        auto = [True] * 10
        auto[7] = False
        path = make_tiny_bundle(tracker=TrackerParams(evidence_needed=2).to_dict(), auto_enabled=auto)
        monkeypatch.setattr(settings, "landmark_bundle_dir", str(path))
        monkeypatch.setattr(settings, "device", "cpu")
        pipe = create_pipeline(with_landmarks=True)
        assert isinstance(pipe.landmark_classifier, RealLandmarkClassifier)
        assert pipe.station_tracker.params.evidence_needed == 2
        assert pipe.station_tracker.auto_enabled == auto
        assert pipe.landmark_meta == {"model_version": "lm-0.0.1"}

    def test_tracker_construction_failure_degrades_to_peace_only(self, monkeypatch, sample_frame):
        monkeypatch.setattr(settings, "landmarks_enabled", True)

        def broken(*args, **kwargs):
            raise TypeError("an integer is required")

        monkeypatch.setattr(pipeline_mod, "StationTracker", broken)
        pipe = create_pipeline(with_landmarks=True)
        assert pipe.landmark_classifier is None and pipe.station_tracker is None
        r = pipe.analyze_frame(sample_frame, full_frame=synthetic_olympus_frame(1920))
        assert r.landmark is None and r.peace_score["score"] in (0, 1, 2, 3)

    def test_real_mode_with_unservable_tracker_meta_degrades_to_peace_only(self, monkeypatch, make_tiny_bundle):
        # A float count loads nowhere now: refused at load, cached, PEACE only.
        self._real_mode(monkeypatch)
        path = make_tiny_bundle(tracker={"evidence_needed": 3, "evidence_window": 6.0})
        monkeypatch.setattr(settings, "landmark_bundle_dir", str(path))
        monkeypatch.setattr(settings, "device", "cpu")
        pipe = create_pipeline(with_landmarks=True)
        assert pipe.landmark_classifier is None and pipe.station_tracker is None
        assert "tracker" in LandmarkModelManager.failure()
        assert warm_up_models()["landmarks"] is False

    def test_real_mode_without_a_bundle_degrades_to_peace_only(self, monkeypatch):
        self._real_mode(monkeypatch)
        pipe = create_pipeline(with_landmarks=True)
        assert pipe.landmark_classifier is None and pipe.station_tracker is None
        assert LandmarkModelManager.failure()  # cached: the next socket does not retry


class TestWarmUp:
    def test_noop_with_mock_models(self):
        assert warm_up_models() == {"peace": False, "landmarks": False}

    def test_landmark_failure_is_logged_not_raised(self, monkeypatch):
        class FakeManager:
            def predict(self, frame):
                return np.ones(3) / 3, np.ones(4) / 4

        monkeypatch.setattr(settings, "use_mock_models", False)
        monkeypatch.setattr(settings, "landmarks_enabled", True)
        monkeypatch.setattr(pipeline_mod.ModelManager, "get_instance", classmethod(lambda cls, *a: FakeManager()))
        assert warm_up_models() == {"peace": True, "landmarks": False}

    def test_a_bundle_that_breaks_the_live_path_is_not_warm(self, monkeypatch, tiny_bundle):
        monkeypatch.setattr(settings, "use_mock_models", False)
        monkeypatch.setattr(settings, "landmarks_enabled", True)
        monkeypatch.setattr(settings, "landmark_bundle_dir", str(tiny_bundle))
        monkeypatch.setattr(settings, "device", "cpu")
        monkeypatch.setattr(pipeline_mod.ModelManager, "get_instance",
                            classmethod(lambda cls, *a: (_ for _ in ()).throw(RuntimeError("no PEACE"))))

        def broken(*args, **kwargs):
            raise TypeError("Can't parse 'ksize'")

        monkeypatch.setattr("app.ml.landmarks.classifier.quality_metrics", broken)
        assert warm_up_models() == {"peace": False, "landmarks": False}
        assert LandmarkModelManager.peek() is None  # /health: landmarks_loaded false

    def test_warms_up_a_real_bundle(self, monkeypatch, tiny_bundle):
        monkeypatch.setattr(settings, "use_mock_models", False)
        monkeypatch.setattr(settings, "landmarks_enabled", True)
        monkeypatch.setattr(settings, "landmark_bundle_dir", str(tiny_bundle))
        monkeypatch.setattr(settings, "device", "cpu")
        monkeypatch.setattr(pipeline_mod.ModelManager, "get_instance",
                            classmethod(lambda cls, *a: (_ for _ in ()).throw(RuntimeError("no PEACE"))))
        assert warm_up_models() == {"peace": False, "landmarks": True}
        assert LandmarkModelManager.peek() is not None


@pytest.mark.slow
def test_real_peace_outputs_identical_with_landmarks_on_and_off(monkeypatch, tiny_bundle):
    """Non-regression on the real checkpoint (skipped if it is not present)."""
    from pathlib import Path

    from app.ml.real_models import ModelManager, RealPEACEClassifier, RealRegionDetector, SessionState

    if not Path(settings.model_path).is_file():
        pytest.skip("PEACE checkpoint not present")
    monkeypatch.setattr(ModelManager, "_instance", None)
    manager = ModelManager.get_instance(settings.model_path, "cpu")
    lm = LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")

    def real_pipe(**landmarks) -> AnalysisPipeline:
        session = SessionState()
        return AnalysisPipeline(RealPEACEClassifier(manager, session), MockMotionDetector(),
                                RealRegionDetector(session), **landmarks)

    frames = _stream(6)
    off = _run(real_pipe(), frames, with_full=False)
    on = _run(real_pipe(landmark_classifier=RealLandmarkClassifier(lm), station_tracker=StationTracker()),
              frames, with_full=True)
    assert [_peace_view(r) for r in on] == [_peace_view(r) for r in off]
    assert "probs" in on[0].peace_score
