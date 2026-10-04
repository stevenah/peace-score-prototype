"""Analysis pipeline orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import settings
from app.ml.landmarks.classifier import (
    INFERENCE_LOCK,
    BaseLandmarkClassifier,
    LandmarkFrameOutput,
    LandmarkModelManager,
    RealLandmarkClassifier,
)
from app.ml.landmarks.payload import build_payload, error_payload, validate_output
from app.ml.landmarks.tracker import FrameEvidence, StationTracker
from app.ml.mock_models import (
    BaseMotionDetector,
    BasePEACEClassifier,
    BaseRegionDetector,
    MockMotionDetector,
    MockPEACEClassifier,
    MockRegionDetector,
    SCORE_LABELS,
)


from app.ml.real_models import (
    ModelManager,
    RealMotionDetector,
    RealPEACEClassifier,
    RealRegionDetector,
    SessionState,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    peace_score: dict
    motion: dict | None = None
    region: str = "stomach"
    # esge10.v1 wire block; None when the landmark feature is not attached or
    # no full-resolution frame was given.
    landmark: dict | None = None


@dataclass
class AnalysisPipeline:
    classifier: BasePEACEClassifier
    motion_detector: BaseMotionDetector
    region_detector: BaseRegionDetector
    sample_rate_fps: float = 2.0
    # Optional ESGE station recognition (live websocket only in v1).
    landmark_classifier: BaseLandmarkClassifier | None = None
    station_tracker: StationTracker | None = None
    landmark_every_n: int = 1
    landmark_display: bool = False
    landmark_meta: dict = field(default_factory=dict)
    _landmark_offered: int = field(default=0, init=False, repr=False)

    def analyze_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
        timestamp: float = 0.0,
        total_duration: float = 0.0,
        frame_index: int = 0,
        full_frame: np.ndarray | None = None,
    ) -> FrameResult:
        # PEACE, motion and region: unchanged, on the 224 px frame only.
        with INFERENCE_LOCK:
            peace_score = self.classifier.predict(frame)
        motion = self.motion_detector.detect(frame, prev_frame)
        region = self.region_detector.detect(frame, timestamp, total_duration)

        landmark = None
        if self.landmark_classifier is not None and full_frame is not None:
            landmark = self._analyze_landmark(full_frame)

        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            peace_score=peace_score,
            motion=motion,
            region=region,
            landmark=landmark,
        )

    def _analyze_landmark(self, full_frame: np.ndarray) -> dict:
        """Landmark step. Never raises: any fault becomes ``status: "error"``."""
        t0 = time.perf_counter()
        tracker = self.station_tracker
        if tracker is None:
            tracker = self.station_tracker = StationTracker()
        try:
            n = self._landmark_offered
            self._landmark_offered += 1
            if n % max(1, self.landmark_every_n) != 0:
                out = LandmarkFrameOutput("skipped")
                return build_payload(out, tracker.update(None), self.landmark_meta, self.landmark_display)
            out = self.landmark_classifier.predict(full_frame)
            validate_output(out)  # before the tracker: a rejected frame is never evidence
            step = tracker.update(out.evidence())
            return build_payload(out, step, self.landmark_meta, self.landmark_display)
        except Exception:
            logger.exception("Landmark step failed; PEACE result unaffected")
            try:
                step = tracker.update(FrameEvidence("error"))
            except Exception:
                step = None
            return error_payload(step, self.landmark_meta, self.landmark_display,
                                 (time.perf_counter() - t0) * 1000.0)

    def analyze_frames(
        self,
        frames: list[tuple[int, float, np.ndarray]],
        total_duration: float,
    ) -> list[FrameResult]:
        """Analyze a list of (index, timestamp, frame) tuples."""
        results: list[FrameResult] = []
        prev_frame: np.ndarray | None = None

        for idx, timestamp, frame in frames:
            result = self.analyze_frame(
                frame=frame,
                prev_frame=prev_frame,
                timestamp=timestamp,
                total_duration=total_duration,
                frame_index=idx,
            )
            results.append(result)
            prev_frame = frame

        return results

    def aggregate_results(
        self, results: list[FrameResult], total_duration: float
    ) -> dict:
        """Aggregate frame-level results into a full analysis response."""
        if not results:
            return {
                "motion_analysis": {"segments": []},
                "peace_scores": {
                    "overall": {"score": 0, "label": "Poor", "confidence": 0.0},
                    "by_region": {},
                },
                "timeline": [],
            }

        # Build timeline
        timeline = []
        for r in results:
            timeline.append(
                {
                    "timestamp": r.timestamp,
                    "frame_index": r.frame_index,
                    "motion": r.motion["direction"] if r.motion else "stationary",
                    "region": r.region,
                    "peace_score": r.peace_score["score"],
                    "confidence": r.peace_score["confidence"],
                }
            )

        # Build motion segments
        segments = self._build_motion_segments(results)

        # Build per-region scores
        region_results: dict[str, list[FrameResult]] = {}
        for r in results:
            region_results.setdefault(r.region, []).append(r)

        by_region = {}
        for region, region_frames in region_results.items():
            scores = [f.peace_score["score"] for f in region_frames]
            # Use average score in region (rounded to nearest integer)
            avg_score = round(sum(scores) / len(scores))
            avg_confidence = sum(
                f.peace_score["confidence"] for f in region_frames
            ) / len(region_frames)

            by_region[region] = {
                "score": avg_score,
                "label": SCORE_LABELS[avg_score],
                "confidence": round(avg_confidence, 2),
                "region": region,
                "frame_scores": [
                    {
                        "frame_index": f.frame_index,
                        "timestamp": f.timestamp,
                        "score": f.peace_score["score"],
                        "confidence": f.peace_score["confidence"],
                    }
                    for f in region_frames
                ],
            }

        # Overall score = average across all frames (keep decimal)
        all_scores = [r.peace_score["score"] for r in results]
        overall_score = round(sum(all_scores) / len(all_scores), 2)
        overall_confidence = round(
            sum(r.peace_score["confidence"] for r in results) / len(results), 2
        )

        return {
            "motion_analysis": {"segments": segments},
            "peace_scores": {
                "overall": {
                    "score": overall_score,
                    "label": SCORE_LABELS[round(overall_score)],
                    "confidence": overall_confidence,
                },
                "by_region": by_region,
            },
            "timeline": timeline,
        }

    def _build_motion_segments(self, results: list[FrameResult]) -> list[dict]:
        """Group consecutive frames with same motion direction into segments."""
        if not results:
            return []

        segments: list[dict] = []
        current_dir = results[0].motion["direction"] if results[0].motion else "stationary"
        segment_start = results[0].timestamp

        for i in range(1, len(results)):
            direction = (
                results[i].motion["direction"] if results[i].motion else "stationary"
            )
            if direction != current_dir:
                segments.append(
                    {
                        "start_time": segment_start,
                        "end_time": results[i].timestamp,
                        "direction": current_dir,
                        "confidence": round(
                            sum(
                                r.motion["confidence"]
                                for r in results[
                                    max(0, i - 5) : i  # last 5 frames avg
                                ]
                                if r.motion
                            )
                            / max(1, min(5, i)),
                            2,
                        ),
                    }
                )
                current_dir = direction
                segment_start = results[i].timestamp

        # Final segment
        segments.append(
            {
                "start_time": segment_start,
                "end_time": results[-1].timestamp,
                "direction": current_dir,
                "confidence": 0.85,
            }
        )

        return segments


def create_pipeline(with_landmarks: bool = False) -> AnalysisPipeline:
    """Factory function to create the appropriate pipeline.

    ``with_landmarks`` is requested by the live websocket only; the station
    classifier is attached when ``settings.landmarks_enabled`` too, and only
    if it can be served (a missing or broken bundle leaves PEACE running).
    """
    pipeline = _create_peace_pipeline()
    if with_landmarks and settings.landmarks_enabled:
        _attach_landmarks(pipeline)
    return pipeline


def _attach_landmarks(pipeline: AnalysisPipeline) -> None:
    """Attach classifier + tracker, or nothing: any failure leaves PEACE only."""
    try:
        if settings.use_mock_models:
            from app.ml.landmarks.mock import MockLandmarkClassifier

            clf: BaseLandmarkClassifier = MockLandmarkClassifier()
        else:
            clf = RealLandmarkClassifier(LandmarkModelManager.get_instance())
        tracker = StationTracker(clf.tracker_params, clf.auto_enabled)
        every_n = max(1, int(settings.landmark_every_n))
    except Exception as exc:  # LandmarkUnavailable, or anything unforeseen
        logger.warning("Landmarks enabled but unavailable; PEACE only: %s", exc)
        return
    pipeline.landmark_classifier = clf
    pipeline.station_tracker = tracker
    pipeline.landmark_every_n = every_n
    pipeline.landmark_display = bool(settings.landmarks_display)
    pipeline.landmark_meta = {"model_version": clf.model_version}


def warm_up_models() -> dict:
    """Load the real models and run one dummy forward each (startup only).

    Returns {"peace": bool, "landmarks": bool}. Failures are logged, never
    raised: the app keeps serving what it can.
    """
    status = {"peace": False, "landmarks": False}
    if settings.use_mock_models:
        return status
    try:
        manager = ModelManager.get_instance(settings.model_path, settings.device)
        with INFERENCE_LOCK:
            manager.predict(np.zeros((224, 224, 3), np.uint8))
        status["peace"] = True
    except Exception:
        logger.exception("PEACE model warm-up failed")
    if settings.landmarks_enabled:
        try:
            LandmarkModelManager.get_instance()  # loads, then warms up the full live path
            status["landmarks"] = True
        except Exception as exc:
            logger.error("Landmark model warm-up failed; landmarks off: %s", exc)
    return status


def _create_peace_pipeline() -> AnalysisPipeline:
    if settings.use_mock_models:
        return AnalysisPipeline(
            classifier=MockPEACEClassifier(),
            motion_detector=MockMotionDetector(),
            region_detector=MockRegionDetector(),
            sample_rate_fps=settings.sample_rate_fps,
        )
    else:

        print("--------------------------------")
        print("model_path: ", settings.model_path)
        print("device: ", settings.device)
        print("--------------------------------")

        manager = ModelManager.get_instance(settings.model_path, settings.device)
        session = SessionState()
        return AnalysisPipeline(
            classifier=RealPEACEClassifier(manager, session),
            motion_detector=RealMotionDetector(),
            region_detector=RealRegionDetector(session),
            sample_rate_fps=settings.sample_rate_fps,
        )
