"""Per-frame image quality for the landmark path.

Cheap, model-free measures computed on the masked model input, inside the
aperture eroded a little further (the octagon edge would otherwise dominate
the sharpness term). Used to (a) drop non-diagnostic frames from training,
(b) gate live frames to ``low_quality`` so they never count as evidence, and
(c) rank candidate best frames per station.

The sharpness reference is calibrated on training data (median of clip-centre
frames) and shipped in the bundle's meta.json; the default here is only a
placeholder for tests and the mock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from app.ml.landmarks.layouts import Layout, aperture_mask

# Side (px) of the model input quality is measured on, whatever the bundle's
# own input_size: sharp_ref, q_min and the frame cache's metrics are all
# calibrated at 224 px, and var(Laplacian) falls ~20% from 224 to 288 px.
QUALITY_SIZE = 224


@dataclass(frozen=True)
class QualityParams:
    # Calibrated per bundle: median var(Laplacian) of clip-centre frames at
    # 224 px. 1100 is the measured median over 250 clips (2026-09-28).
    sharp_ref: float = 1100.0
    dark_y: float = 20.0
    dark_max: float = 0.40
    sat_y: float = 245.0
    sat_max: float = 0.25
    redout_max: float = 0.50
    edge_erode_px: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QualityParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass(frozen=True)
class QualityMetrics:
    sharpness: float
    dark_frac: float
    sat_frac: float
    redout: float


def _inner_mask(layout: Layout, size: int, erode_px: int) -> np.ndarray:
    mask = aperture_mask(layout, size).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    return cv2.erode(mask, k).astype(bool)


def quality_metrics(img: np.ndarray, layout: Layout, params: QualityParams = QualityParams()) -> QualityMetrics:
    """Metrics for a masked uint8 RGB model input (``to_model_input`` at QUALITY_SIZE)."""
    inner = _inner_mask(layout, img.shape[0], params.edge_erode_px)
    rgb = img.astype(np.float32)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
    yin = y[inner]
    r, g, b = rgb[..., 0][inner], rgb[..., 1][inner], rgb[..., 2][inner]
    return QualityMetrics(
        sharpness=float(lap[inner].var()),
        dark_frac=float((yin < params.dark_y).mean()),
        sat_frac=float((yin > params.sat_y).mean()),
        redout=float(((r > 150) & (g < 60) & (b < 60)).mean()),
    )


def quality_score(m: QualityMetrics, params: QualityParams = QualityParams()) -> float:
    """0..1: relative sharpness, zeroed by darkness, saturation or red-out."""
    if m.dark_frac >= params.dark_max or m.sat_frac >= params.sat_max or m.redout >= params.redout_max:
        return 0.0
    return float(min(1.0, m.sharpness / max(params.sharp_ref, 1e-6)))
