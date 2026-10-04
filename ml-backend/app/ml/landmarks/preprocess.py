"""Landmark preprocessing shared by training, cache extraction and serving.

    serving:   normalize(to_model_input(canonicalize(full_frame, layout)))
    training:  normalize(to_model_input(augment(decode(cached_canvas_jpeg))))

``canonicalize`` is the only place that looks at a full processor frame. The
only intended difference between training and serving is the q95 JPEG of the
frame cache; tests/test_landmark_parity.py bounds its effect.

The PEACE model keeps its own squash-to-224 path; nothing here changes it.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.ml.landmarks.layouts import CANON_SIZE, Layout, aperture_mask, canvas_geometry

PREPROCESS_VERSION = "lmpre-1"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def canonicalize(full_rgb: np.ndarray, layout: Layout) -> np.ndarray:
    """Crop the aperture ROI into a masked uint8 (320, 320, 3) RGB canvas.

    Works for any resolution with the layout's aspect ratio (ROI coordinates
    scale with width). Pixels outside the aperture are exactly 0.
    """
    s = layout.scale(full_rgb.shape[1])
    x0, y0, x1, y1 = (int(round(v * s)) for v in layout.roi)
    roi = full_rgb[y0:y1, x0:x1]
    _, h, pad_top = canvas_geometry(layout)
    small = cv2.resize(roi, (CANON_SIZE, h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANON_SIZE, CANON_SIZE, 3), np.uint8)
    canvas[pad_top:pad_top + h] = small
    canvas[~aperture_mask(layout, CANON_SIZE)] = 0
    return canvas


def to_model_input(canvas: np.ndarray, layout: Layout, size: int = 224) -> np.ndarray:
    """Downscale a canvas to the model size and re-apply the mask at that size.

    Re-masking after the resize (and, in training, after augmentation) is what
    guarantees exact zeros outside the aperture in every model input.
    """
    if canvas.shape[0] != size:
        img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    else:
        img = canvas.copy()
    img[~aperture_mask(layout, size)] = 0
    return img


def normalize(img: np.ndarray):
    """uint8 HWC RGB -> float32 CHW torch tensor with ImageNet statistics."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std
