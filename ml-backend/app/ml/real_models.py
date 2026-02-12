"""Real model implementations using the trained multitask PEACE model.

Provides adapter classes that implement the base ABCs from mock_models.py,
backed by a shared ModelManager singleton that owns the GPU model instance.

Each WebSocket connection gets its own adapter instances with independent
temporal smoothing state, while sharing the model weights via ModelManager.
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
from app.ml.model import PEACEMultitaskModel, MOTIONS
from app.ml.temporal import TemporalSmoother

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

        self.model = PEACEMultitaskModel(pretrained_backbone=False)
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
        prev_features: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """Run inference on a single frame.

        Args:
            frame: (H, W, 3) RGB numpy array (will be resized/normalized)
            prev_features: (1, neck_dim) tensor from previous frame, or None

        Returns:
            region_probs: (3,) numpy array
            score_probs: (4,) numpy array
            motion_probs: (3,) numpy array
            features: (1, neck_dim) tensor to cache for next frame
        """
        tensor = self.transform(frame).unsqueeze(0).to(self.device)

        if prev_features is not None:
            prev_features = prev_features.to(self.device)

        region_logits, score_logits, motion_logits, features = self.model(
            tensor, prev_features
        )

        region_probs = torch.softmax(region_logits, dim=-1).cpu().numpy()[0]
        score_probs = torch.softmax(score_logits, dim=-1).cpu().numpy()[0]
        motion_probs = torch.softmax(motion_logits, dim=-1).cpu().numpy()[0]

        return region_probs, score_probs, motion_probs, features.cpu()


class SessionState:
    """Per-connection state for temporal smoothing and feature caching."""

    def __init__(self, alpha: float = 0.3):
        self.smoother = TemporalSmoother(alpha=alpha)
        self.prev_features: torch.Tensor | None = None
        self.last_result: dict | None = None


class RealPEACEClassifier(BasePEACEClassifier):
    """Adapter wrapping the multitask model for PEACE score prediction.

    This is called first in the pipeline. It runs the full model inference,
    caches the result in SessionState, and returns the score portion.
    """

    def __init__(self, manager: ModelManager, session: SessionState):
        self._manager = manager
        self._session = session

    def predict(self, frame: np.ndarray) -> dict:
        region_probs, score_probs, motion_probs, features = self._manager.predict(
            frame, self._session.prev_features
        )
        self._session.prev_features = features

        (
            region_idx, region_conf,
            score_idx, score_conf,
            motion_idx, motion_conf,
        ) = self._session.smoother.update(region_probs, score_probs, motion_probs)

        # Cache full result for motion/region detectors to read
        self._session.last_result = {
            "region": REGIONS[region_idx],
            "region_confidence": region_conf,
            "score": score_idx,
            "score_confidence": score_conf,
            "motion_direction": MOTIONS[motion_idx],
            "motion_confidence": motion_conf,
        }

        return {
            "score": score_idx,
            "label": SCORE_LABELS[score_idx],
            "confidence": round(score_conf, 2),
        }


class RealMotionDetector(BaseMotionDetector):
    """Adapter reading cached motion prediction from SessionState.

    Must be called after RealPEACEClassifier.predict() on the same frame.
    """

    def __init__(self, session: SessionState):
        self._session = session

    def detect(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> dict:
        result = self._session.last_result
        if result is None:
            return {
                "direction": "stationary",
                "confidence": 0.5,
                "optical_flow_magnitude": 0.0,
            }
        return {
            "direction": result["motion_direction"],
            "confidence": round(result["motion_confidence"], 2),
            "optical_flow_magnitude": 0.0,
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
