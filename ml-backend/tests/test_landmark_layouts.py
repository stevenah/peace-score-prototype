"""Screen layout detection, aperture mask and canonical preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.landmarks.layouts import (
    CANON_SIZE,
    LEGACY_1350,
    OLYMPUS_1920,
    aperture_mask,
    detect_layout,
    detect_mode,
)
from app.ml.landmarks.preprocess import canonicalize, normalize, to_model_input


def _olympus_frame(width: int = 1920, fill: int = 160, badge: str | None = None) -> np.ndarray:
    """Synthetic processor frame: black background, lit octagon, dim OSD text."""
    import cv2

    h = int(round(width * 1080 / 1920))
    s = width / 1920
    frame = np.zeros((h, width, 3), np.uint8)
    poly = np.array([(x * s, y * s) for x, y in OLYMPUS_1920.polygon], np.int32)
    cv2.fillPoly(frame, [poly], (fill, fill // 2, fill // 3))
    # A few rows of OSD text on the left panel.
    for i in range(6):
        cv2.putText(frame, "ID pacjenta 01/01/2000", (int(10 * s), int((40 + 30 * i) * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (220, 220, 220), 1)
    if badge:
        x0, y0, x1, y1 = OLYMPUS_1920.badge_box
        text_x1 = x1 + 18 if badge == "rdi" else x1
        cv2.rectangle(frame, (int(x0 * s), int(y0 * s)), (int(text_x1 * s), int(y1 * s)),
                      (255, 255, 255), -1)
    return frame


def test_detects_olympus_at_full_hd_and_720p():
    assert detect_layout(_olympus_frame(1920)) is OLYMPUS_1920
    assert detect_layout(_olympus_frame(1280)) is OLYMPUS_1920


def test_refuses_small_or_unknown_frames():
    assert detect_layout(_olympus_frame(960)) is None  # below MIN_SERVED_WIDTH
    assert detect_layout(np.full((480, 640, 3), 120, np.uint8)) is None  # 4:3
    lit = np.full((1080, 1920, 3), 120, np.uint8)  # 16:9 but no dark corners/panel
    assert detect_layout(lit) is None
    assert detect_layout(np.zeros((1080, 1920), np.uint8)) is None  # not RGB


def test_legacy_layout_is_detected_but_not_served():
    layout = detect_layout(np.zeros((1080, 1350, 3), np.uint8))
    assert layout is LEGACY_1350
    assert not layout.served


def test_detect_mode_reads_badge():
    assert detect_mode(_olympus_frame(), OLYMPUS_1920) == "wl"
    assert detect_mode(_olympus_frame(badge="nbi"), OLYMPUS_1920) == "nbi"
    assert detect_mode(_olympus_frame(badge="rdi"), OLYMPUS_1920) == "rdi"


@pytest.mark.parametrize("size", [CANON_SIZE, 224, 288])
def test_mask_is_read_only_and_blanks_badge_and_corners(size):
    mask = aperture_mask(OLYMPUS_1920, size)
    assert mask.shape == (size, size) and mask.dtype == bool
    assert not mask.flags.writeable
    # Corners of the canvas and the pad bands are outside the aperture.
    assert not mask[0, 0] and not mask[-1, -1] and not mask[0, size // 2]
    assert mask[size // 2, size // 2]
    # Aperture covers a sensible share of the canvas (octagon in a 5:4 band).
    assert 0.55 < mask.mean() < 0.80


def test_canonicalize_zeroes_everything_outside_the_aperture():
    frame = _olympus_frame(badge="nbi")
    frame[:] = np.maximum(frame, 90)  # light up panel, corners and badge area too
    canvas = canonicalize(frame, OLYMPUS_1920)
    assert canvas.shape == (CANON_SIZE, CANON_SIZE, 3) and canvas.dtype == np.uint8
    mask = aperture_mask(OLYMPUS_1920, CANON_SIZE)
    assert canvas[~mask].max() == 0
    assert canvas[mask].min() > 0
    model_in = to_model_input(canvas, OLYMPUS_1920, 224)
    assert model_in[~aperture_mask(OLYMPUS_1920, 224)].max() == 0


def test_canonicalize_is_resolution_invariant():
    rng = np.random.default_rng(0)
    big = _olympus_frame(1920)
    texture = rng.integers(0, 40, size=big.shape, dtype=np.uint8)
    big = np.where(big > 0, np.clip(big.astype(int) + texture, 0, 255), 0).astype(np.uint8)
    import cv2

    small = cv2.resize(big, (1280, 720), interpolation=cv2.INTER_AREA)
    a = to_model_input(canonicalize(big, OLYMPUS_1920), OLYMPUS_1920).astype(float)
    b = to_model_input(canonicalize(small, OLYMPUS_1920), OLYMPUS_1920).astype(float)
    assert np.abs(a - b).mean() < 3.0


def test_canonicalize_is_deterministic():
    frame = _olympus_frame()
    assert np.array_equal(canonicalize(frame, OLYMPUS_1920), canonicalize(frame, OLYMPUS_1920))


def test_legacy_mask_never_exceeds_canonical_aperture():
    for size in (CANON_SIZE, 224):
        legacy, canon = aperture_mask(LEGACY_1350, size), aperture_mask(OLYMPUS_1920, size)
        assert not np.any(legacy & ~canon)


def test_normalize_produces_chw_float_tensor():
    torch = pytest.importorskip("torch")
    t = normalize(np.zeros((224, 224, 3), np.uint8))
    assert t.shape == (3, 224, 224) and t.dtype == torch.float32
