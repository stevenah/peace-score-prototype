"""Frame quality falls monotonically as frames get blurrier, darker or red."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ml.landmarks.bench import synthetic_olympus_frame
from app.ml.landmarks.layouts import OLYMPUS_1920
from app.ml.landmarks.preprocess import canonicalize, to_model_input
from app.ml.landmarks.quality import QualityParams, quality_metrics, quality_score


def _q(frame: np.ndarray, params: QualityParams = QualityParams()) -> float:
    img = to_model_input(canonicalize(frame, OLYMPUS_1920), OLYMPUS_1920, 224)
    return quality_score(quality_metrics(img, OLYMPUS_1920, params), params)


def _aperture(frame: np.ndarray) -> np.ndarray:
    m = np.zeros(frame.shape[:2], np.uint8)
    poly = np.array(OLYMPUS_1920.polygon, np.int32)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


@pytest.fixture(scope="module")
def sharp():
    return synthetic_olympus_frame(seed=11)


def _blur(frame, sigma):
    out = cv2.GaussianBlur(frame, (0, 0), sigma)
    out[~_aperture(frame)] = frame[~_aperture(frame)]
    return out


def test_synthetic_frame_has_usable_quality(sharp):
    assert 0.3 < _q(sharp) <= 1.0


def test_quality_falls_with_blur(sharp):
    scores = [_q(sharp)] + [_q(_blur(sharp, s)) for s in (2, 5, 12)]
    assert all(a > b for a, b in zip(scores, scores[1:])), scores
    assert scores[-1] < 0.1


def test_quality_falls_with_darkness(sharp):
    scores = [_q(synthetic_olympus_frame(seed=11, brightness=b)) for b in (1.0, 0.5, 0.12)]
    assert scores[0] > scores[1] > scores[2], scores
    assert scores[2] == 0.0  # mostly below the dark threshold -> gated out


@pytest.mark.parametrize("fill", [(210, 25, 20), (255, 255, 255)], ids=["red-out", "saturated"])
def test_red_out_and_saturation_are_gated(sharp, fill):
    frame = sharp.copy()
    ap = _aperture(frame)
    frame[ap] = (0.95 * np.asarray(fill) + 0.05 * frame[ap]).astype(np.uint8)
    assert _q(frame) == 0.0 < _q(sharp)


def test_quality_ignores_everything_outside_the_aperture(sharp):
    noisy = sharp.copy()
    rng = np.random.default_rng(0)
    outside = ~_aperture(noisy)
    noisy[outside] = rng.integers(0, 255, size=(int(outside.sum()), 3), dtype=np.uint8)
    assert _q(noisy) == pytest.approx(_q(sharp), abs=1e-6)


def test_quality_reference_scales_the_score(sharp):
    loose, strict = QualityParams(sharp_ref=100.0), QualityParams(sharp_ref=100000.0)
    assert _q(sharp, loose) == 1.0 and _q(sharp, strict) < 0.05
