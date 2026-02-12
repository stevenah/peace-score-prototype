"""Temporal smoothing for frame-level predictions.

Applies exponential moving average (EMA) on output probabilities
to provide stable predictions without requiring sequence buffering.
"""

from __future__ import annotations

import numpy as np


class TemporalSmoother:
    """EMA-based temporal smoother for classification probabilities.

    At each timestep, updates smoothed probabilities:
        smoothed = alpha * new + (1 - alpha) * smoothed

    With alpha=0.3 at 2fps, this gives ~1.5-2 second effective memory.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.region_probs: np.ndarray | None = None
        self.score_probs: np.ndarray | None = None
        self.motion_probs: np.ndarray | None = None

    def update(
        self,
        region_probs: np.ndarray,
        score_probs: np.ndarray,
        motion_probs: np.ndarray,
    ) -> tuple[int, float, int, float, int, float]:
        """Update smoother with new frame probabilities.

        Returns:
            (region_idx, region_confidence,
             score_idx, score_confidence,
             motion_idx, motion_confidence)
        """
        if self.region_probs is None:
            self.region_probs = region_probs.copy()
            self.score_probs = score_probs.copy()
            self.motion_probs = motion_probs.copy()
        else:
            self.region_probs = (
                self.alpha * region_probs + (1 - self.alpha) * self.region_probs
            )
            self.score_probs = (
                self.alpha * score_probs + (1 - self.alpha) * self.score_probs
            )
            self.motion_probs = (
                self.alpha * motion_probs + (1 - self.alpha) * self.motion_probs
            )

        region_idx = int(np.argmax(self.region_probs))
        score_idx = int(np.argmax(self.score_probs))
        motion_idx = int(np.argmax(self.motion_probs))

        return (
            region_idx,
            float(self.region_probs[region_idx]),
            score_idx,
            float(self.score_probs[score_idx]),
            motion_idx,
            float(self.motion_probs[motion_idx]),
        )

    def reset(self) -> None:
        """Reset smoother state for a new video/session."""
        self.region_probs = None
        self.score_probs = None
        self.motion_probs = None
