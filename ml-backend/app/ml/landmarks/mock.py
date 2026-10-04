"""Deterministic mock landmark classifier for UI development and tests.

Plays a scripted upper-GI procedure, one step per ``predict`` call:

    proximal esophagus -> bulb -> D2 -> antrum -> corpus -> cardia/fundus
    -> lesser curvature -> incisura -> antrum -> Z-line -> distal esophagus
    -> proximal esophagus (then repeats)

Each station is DWELL "ok" frames followed by TRANSIT frames alternating
low_quality and uncertain (leaning to the next station), and the imaging mode
alternates wl/nbi per station. Run through the real StationTracker at 2 Hz it
observes all ten stations in about 71 s.

It still runs the real layout lock, so a non-processor source reports
``unsupported_layout`` just as the real model would. It never imports torch
and never draws from the global ``random`` (the mock PEACE model does, and
landmark draws would shift its outputs). It is only constructible in mock
mode, so simulated stations can never appear next to real PEACE scores.
"""

from __future__ import annotations

import random
import time

import numpy as np

from app.config import settings
from app.ml.landmarks.classifier import BaseLandmarkClassifier, LandmarkFrameOutput
from app.ml.landmarks.session import LayoutLock
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX

MOCK_MODEL_VERSION = "mock-esge10-1"

FLOW: tuple[str, ...] = (
    "esophagus_proximal",
    "duodenal_bulb",
    "duodenum_descending",
    "antrum",
    "corpus_greater_curvature",
    "cardia_fundus_retroflex",
    "lesser_curvature_retroflex",
    "incisura",
    "antrum",
    "z_line",
    "esophagus_distal",
    "esophagus_proximal",
)
DWELL = 8
TRANSIT = 6
SEGMENT = DWELL + TRANSIT
CYCLE = SEGMENT * len(FLOW)


def _probs(top: int, confidence: float, second: int | None, rng: random.Random) -> tuple[float, ...]:
    """A 10-way distribution whose argmax is ``top`` with p = ``confidence``."""
    probs = [0.0] * NUM_STATIONS
    probs[top] = confidence
    rest = 1.0 - confidence
    if second is not None and second != top:
        probs[second] = min(0.6 * rest, 0.8 * confidence)
        rest -= probs[second]
    others = [k for k in range(NUM_STATIONS) if k != top and k != second]
    weights = [0.2 + rng.random() for _ in others]
    total = sum(weights)
    for k, w in zip(others, weights):
        probs[k] = rest * w / total
    return tuple(probs)


class MockLandmarkClassifier(BaseLandmarkClassifier):
    model_version = MOCK_MODEL_VERSION

    def __init__(self, seed: int = 0, layout_lock: LayoutLock | None = None):
        if not settings.use_mock_models:
            raise RuntimeError("MockLandmarkClassifier is only allowed with use_mock_models=True")
        self._seed = seed
        self._layouts = layout_lock or LayoutLock()
        self._i = 0

    def predict(self, full_frame: np.ndarray) -> LandmarkFrameOutput:
        t0 = time.perf_counter()
        i, self._i = self._i, self._i + 1
        decision = self._layouts.resolve(full_frame)
        if decision.layout is None:
            return LandmarkFrameOutput("unsupported_layout", latency_ms=(time.perf_counter() - t0) * 1000)
        out = self.frame_at(i, decision.layout.name)
        if not decision.verified and out.status != "low_quality":
            out = LandmarkFrameOutput(**{**out.__dict__, "status": "low_quality"})
        return LandmarkFrameOutput(**{**out.__dict__, "latency_ms": (time.perf_counter() - t0) * 1000})

    def frame_at(self, i: int, layout: str = "olympus_1920x1080") -> LandmarkFrameOutput:
        """The scripted output for step ``i`` (pure function of seed and i)."""
        seg, pos = divmod(i % CYCLE, SEGMENT)
        here = STATION_INDEX[FLOW[seg]]
        after = STATION_INDEX[FLOW[(seg + 1) % len(FLOW)]]
        mode = "nbi" if seg % 2 else "wl"
        rng = random.Random(self._seed * 1_000_003 + i)
        if pos < DWELL:
            status, top, second = "ok", here, None
            confidence = 0.80 + 0.15 * rng.random()
            quality = 0.60 + 0.35 * rng.random()
        elif (pos - DWELL) % 2 == 0:
            status, top, second = "low_quality", here, after
            confidence = 0.45 + 0.25 * rng.random()
            quality = 0.02 + 0.12 * rng.random()
        else:
            status, top, second = "uncertain", after, here
            confidence = 0.35 + 0.10 * rng.random()
            quality = 0.45 + 0.30 * rng.random()
        return LandmarkFrameOutput(
            status=status,
            layout=layout,
            mode=mode,
            probs=_probs(top, confidence, second, rng),
            top=top,
            confidence=confidence,
            quality=quality,
        )
