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
    """Simulates optical-flow motion detection using frame differencing.

    Uses mean pixel difference as a proxy for optical flow magnitude and
    vertical brightness gradient shift to approximate direction.
    """

    def detect(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> dict:
        if prev_frame is None:
            return {
                "direction": "stationary",
                "confidence": 0.5,
                "optical_flow_magnitude": 0.0,
            }

        diff = float(np.mean(np.abs(frame.astype(float) - prev_frame.astype(float))))

        if diff < 10:
            direction = "stationary"
            confidence = round(random.uniform(0.8, 0.95), 2)
        elif diff < 30:
            direction = "stationary"
            confidence = round(random.uniform(0.5, 0.7), 2)
        else:
            # Use vertical brightness shift as mock direction signal:
            # compare mean of top half vs bottom half across frames
            h = frame.shape[0] // 2
            top_shift = float(np.mean(frame[:h].astype(float) - prev_frame[:h].astype(float)))
            bot_shift = float(np.mean(frame[h:].astype(float) - prev_frame[h:].astype(float)))
            if bot_shift > top_shift + 2:
                direction = "insertion"
            elif top_shift > bot_shift + 2:
                direction = "retraction"
            else:
                direction = random.choice(["insertion", "retraction"])
            confidence = round(min(0.95, 0.5 + diff / 100), 2)

        return {
            "direction": direction,
            "confidence": confidence,
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
