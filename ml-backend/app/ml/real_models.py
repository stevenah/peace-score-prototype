"""Real model implementations using the trained PEACE model.

Provides adapter classes that implement the base ABCs from mock_models.py,
backed by a shared ModelManager singleton that owns the GPU model instance.

The trained model has two heads (region + score). Motion detection uses
Farneback dense optical flow (OpenCV) to classify insertion/withdrawal.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torchvision import transforms

from app.ml.mock_models import (
    BasePEACEClassifier,
    BaseMotionDetector,
    BaseRegionDetector,
    REGIONS,
    SCORE_LABELS,
)
from app.ml.model import PEACEModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton that owns the PyTorch model and handles GPU inference.

    Stateless — no per-connection temporal state. Shared across all
    WebSocket connections and API requests.
    """

    _instance: ModelManager | None = None

    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._resolve_device(device)
        logger.info("Loading PEACE model from %s on %s", model_path, self.device)

        self.model = PEACEModel(pretrained_backbone=False)
        if model_path:
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info("PEACE model loaded successfully")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @classmethod
    def get_instance(cls, model_path: str = "", device: str = "auto") -> ModelManager:
        """Get or create the singleton model manager."""
        if cls._instance is None:
            cls._instance = cls(model_path, device)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    @torch.no_grad()
    def predict(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a single frame.

        Args:
            frame: (H, W, 3) RGB numpy array (will be resized/normalized)

        Returns:
            region_probs: (3,) numpy array
            score_probs: (4,) numpy array
        """
        tensor = self.transform(frame).unsqueeze(0).to(self.device)

        region_logits, score_logits, _ = self.model(tensor)

        region_probs = torch.softmax(region_logits, dim=-1).cpu().numpy()[0]
        score_probs = torch.softmax(score_logits, dim=-1).cpu().numpy()[0]

        return region_probs, score_probs


class SessionState:
    """Per-connection state for temporal smoothing."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.region_probs: np.ndarray | None = None
        self.score_probs: np.ndarray | None = None
        self.last_result: dict | None = None

    def smooth(
        self, region_probs: np.ndarray, score_probs: np.ndarray,
    ) -> tuple[int, float, int, float]:
        """EMA smooth and return (region_idx, region_conf, score_idx, score_conf)."""
        if self.region_probs is None:
            self.region_probs = region_probs.copy()
            self.score_probs = score_probs.copy()
        else:
            self.region_probs = (
                self.alpha * region_probs + (1 - self.alpha) * self.region_probs
            )
            self.score_probs = (
                self.alpha * score_probs + (1 - self.alpha) * self.score_probs
            )

        region_idx = int(np.argmax(self.region_probs))
        score_idx = int(np.argmax(self.score_probs))

        return (
            region_idx,
            float(self.region_probs[region_idx]),
            score_idx,
            float(self.score_probs[score_idx]),
        )


class RealPEACEClassifier(BasePEACEClassifier):
    """Adapter wrapping the model for PEACE score prediction.

    This is called first in the pipeline. It runs model inference,
    caches the result in SessionState, and returns the score portion.
    """

    def __init__(self, manager: ModelManager, session: SessionState):
        self._manager = manager
        self._session = session

    def predict(self, frame: np.ndarray) -> dict:
        region_probs, score_probs = self._manager.predict(frame)

        region_idx, region_conf, score_idx, score_conf = self._session.smooth(
            region_probs, score_probs,
        )

        # Cache for region detector to read
        self._session.last_result = {
            "region": REGIONS[region_idx],
            "region_confidence": region_conf,
            "score": score_idx,
            "score_confidence": score_conf,
        }

        return {
            "score": score_idx,
            "label": SCORE_LABELS[score_idx],
            "confidence": round(score_conf, 2),
        }


class RealMotionDetector(BaseMotionDetector):
    """Optical-flow motion detector using Farneback dense flow.

    Computes dense optical flow between consecutive frames, then uses the
    mean vertical component to classify insertion (downward) vs withdrawal
    (upward) vs stationary.
    """

    # Farneback parameters tuned for endoscopy (smooth, medium-sized window)
    _FLOW_PARAMS: dict = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    # Thresholds (in pixels of flow per frame)
    _MAGNITUDE_STATIONARY = 1.5   # below this → stationary
    _DIRECTION_THRESHOLD = 0.4    # |vy/mag| must exceed this to pick a direction

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert RGB uint8 frame to single-channel grayscale."""
        import cv2
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def detect(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> dict:
        import cv2

        gray = self._to_gray(frame)

        if prev_frame is None:
            self._prev_gray = gray
            return {
                "direction": "stationary",
                "confidence": 0.5,
                "optical_flow_magnitude": 0.0,
            }

        prev_gray = self._prev_gray if self._prev_gray is not None else self._to_gray(prev_frame)

        # Dense optical flow → (H, W, 2) with channels (dx, dy)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, **self._FLOW_PARAMS,
        )

        # Cache for next call
        self._prev_gray = gray

        # Compute per-pixel magnitude and mean vertical flow
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(np.mean(mag))
        mean_vy = float(np.mean(flow[..., 1]))  # positive = downward in image

        if mean_mag < self._MAGNITUDE_STATIONARY:
            direction = "stationary"
            confidence = min(0.95, 0.7 + (self._MAGNITUDE_STATIONARY - mean_mag) / self._MAGNITUDE_STATIONARY * 0.25)
        else:
            # Ratio of vertical component to overall magnitude
            vy_ratio = mean_vy / (mean_mag + 1e-6)
            if abs(vy_ratio) < self._DIRECTION_THRESHOLD:
                # Mostly lateral motion — treat as stationary for scope direction
                direction = "stationary"
                confidence = 0.5
            elif vy_ratio > 0:
                direction = "insertion"   # downward motion in image
                confidence = min(0.95, 0.5 + abs(vy_ratio))
            else:
                direction = "withdrawal"  # upward motion in image
                confidence = min(0.95, 0.5 + abs(vy_ratio))

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "optical_flow_magnitude": round(mean_mag, 2),
        }


class RealRegionDetector(BaseRegionDetector):
    """Adapter reading cached region prediction from SessionState.

    Must be called after RealPEACEClassifier.predict() on the same frame.
    """

    def __init__(self, session: SessionState):
        self._session = session

    def detect(
        self, frame: np.ndarray, timestamp: float, total_duration: float
    ) -> str:
        result = self._session.last_result
        if result is None:
            return "stomach"
        return result["region"]
