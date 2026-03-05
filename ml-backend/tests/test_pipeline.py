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

    def test_aggregate_overall_score_is_minimum(self, sample_frame):
        """The overall PEACE score should be the min across all frames."""
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
        assert agg["peace_scores"]["overall"]["score"] == 1

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
