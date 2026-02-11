"""Mock ML models for prototype development.

These produce plausible outputs using frame statistics (brightness, contrast,
frame differencing) without requiring trained model weights.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

SCORE_LABELS = {
    0: "Poor",
    1: "Inadequate",
    2: "Adequate",
    3: "Excellent",
}

REGIONS = ["esophagus", "stomach", "duodenum"]


class BasePEACEClassifier(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> dict:
        ...


class BaseMotionDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> dict:
        ...


class BaseRegionDetector(ABC):
    @abstractmethod
    def detect(
        self, frame: np.ndarray, timestamp: float, total_duration: float
    ) -> str:
        ...


class MockPEACEClassifier(BasePEACEClassifier):
    """Simulates PEACE score classification using frame statistics."""

    def predict(self, frame: np.ndarray) -> dict:
        brightness = float(np.mean(frame))
        contrast = float(np.std(frame))

        # Map brightness/contrast to plausible scores
        if brightness < 50 or contrast < 20:
            base_score = 0
        elif brightness < 100 or contrast < 40:
            base_score = 1
        elif brightness < 170:
            base_score = 2
        else:
            base_score = 3

        # Add some randomness for realism
        score = max(0, min(3, base_score + random.choice([-1, 0, 0, 0, 1])))
        confidence = round(random.uniform(0.65, 0.95), 2)

        return {
            "score": score,
            "label": SCORE_LABELS[score],
            "confidence": confidence,
        }


class MockMotionDetector(BaseMotionDetector):
    """Simulates insertion/retraction detection using frame differencing."""

    def __init__(self) -> None:
        self._frame_count = 0

    def detect(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> dict:
        self._frame_count += 1

        if prev_frame is None:
            return {
                "direction": "stationary",
                "confidence": 0.5,
                "optical_flow_magnitude": 0.0,
            }

        # Use actual frame difference as rough motion signal
        diff = float(np.mean(np.abs(frame.astype(float) - prev_frame.astype(float))))

        if diff < 10:
            direction = "stationary"
        elif diff < 30:
            # Low motion - likely stationary or slow movement
            direction = random.choice(["insertion", "stationary", "retraction"])
        else:
            # Higher motion - pick based on a simple temporal pattern
            # Early frames tend to be insertion, later frames retraction
            direction = "insertion" if self._frame_count < 50 else "retraction"

        return {
            "direction": direction,
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "optical_flow_magnitude": round(diff, 1),
        }


class MockRegionDetector(BaseRegionDetector):
    """Simulates anatomical region detection using time-based heuristic."""

    def detect(
        self, frame: np.ndarray, timestamp: float, total_duration: float
    ) -> str:
        if total_duration <= 0:
            return "stomach"

        progress = timestamp / total_duration

        if progress < 0.2:
            return "esophagus"
        elif progress < 0.7:
            return "stomach"
        else:
            return "duodenum"
