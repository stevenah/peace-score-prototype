"""Analysis pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import settings
from app.ml.mock_models import (
    BaseMotionDetector,
    BasePEACEClassifier,
    BaseRegionDetector,
    MockMotionDetector,
    MockPEACEClassifier,
    MockRegionDetector,
    SCORE_LABELS,
)


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    peace_score: dict
    motion: dict | None = None
    region: str = "stomach"


@dataclass
class AnalysisPipeline:
    classifier: BasePEACEClassifier
    motion_detector: BaseMotionDetector
    region_detector: BaseRegionDetector
    sample_rate_fps: float = 2.0

    def analyze_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
        timestamp: float = 0.0,
        total_duration: float = 0.0,
        frame_index: int = 0,
    ) -> FrameResult:
        peace_score = self.classifier.predict(frame)
        motion = self.motion_detector.detect(frame, prev_frame)
        region = self.region_detector.detect(frame, timestamp, total_duration)

        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            peace_score=peace_score,
            motion=motion,
            region=region,
        )

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
            # Use minimum score in region (per PEACE methodology)
            min_score = min(scores)
            avg_confidence = sum(
                f.peace_score["confidence"] for f in region_frames
            ) / len(region_frames)

            by_region[region] = {
                "score": min_score,
                "label": SCORE_LABELS[min_score],
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

        # Overall score = min across all regions
        all_scores = [r.peace_score["score"] for r in results]
        overall_score = min(all_scores)
        overall_confidence = round(
            sum(r.peace_score["confidence"] for r in results) / len(results), 2
        )

        return {
            "motion_analysis": {"segments": segments},
            "peace_scores": {
                "overall": {
                    "score": overall_score,
                    "label": SCORE_LABELS[overall_score],
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


def create_pipeline() -> AnalysisPipeline:
    """Factory function to create the appropriate pipeline."""
    if settings.use_mock_models:
        return AnalysisPipeline(
            classifier=MockPEACEClassifier(),
            motion_detector=MockMotionDetector(),
            region_detector=MockRegionDetector(),
            sample_rate_fps=settings.sample_rate_fps,
        )
    else:
        # Future: load real models here
        raise NotImplementedError("Real models not yet available")
