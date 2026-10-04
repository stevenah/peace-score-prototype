"""Landmark classifiers for the live path.

- ``LandmarkFrameOutput``: what one full frame yields.
- ``BaseLandmarkClassifier``: the interface (the mock in ``mock.py`` and
  ``RealLandmarkClassifier`` implement it), one instance per live session.
- ``LandmarkModelManager``: the process-wide owner of the verified bundle.
  It is independent of the PEACE ``ModelManager`` (its own singleton, lock and
  cached failure) so a landmark problem can never touch PEACE.

torch is imported lazily: importing this module (as the mock path does)
does not load it.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from app.config import settings
from app.ml.landmarks.bundle import LandmarkBundle, LandmarkUnavailable, fetch_bundle, load_bundle
from app.ml.landmarks.layouts import LAYOUTS_BY_NAME, Layout, detect_mode
from app.ml.landmarks.preprocess import canonicalize, normalize, to_model_input
from app.ml.landmarks.quality import QUALITY_SIZE, quality_metrics, quality_score
from app.ml.landmarks.session import LayoutLock
from app.ml.landmarks.tracker import FrameEvidence, StationTracker, TrackerParams

logger = logging.getLogger(__name__)

__all__ = [
    "INFERENCE_LOCK",
    "BaseLandmarkClassifier",
    "LandmarkFrameOutput",
    "LandmarkModelManager",
    "LandmarkUnavailable",
    "RealLandmarkClassifier",
]

# Process-wide lock around every model forward (PEACE and landmark; see
# pipeline.py). Live sockets, /analyze/frame and the batch worker each run in
# their own thread with torch's intra-op threads; serialising the forwards
# keeps a 2-vCPU machine from oversubscribing. Reentrant so a nested acquire
# can never deadlock.
INFERENCE_LOCK = threading.RLock()


@dataclass(frozen=True)
class LandmarkFrameOutput:
    status: str  # ok | uncertain | low_quality | unsupported_layout | skipped | error
    layout: str | None = None
    mode: str | None = None  # wl | nbi | rdi (metadata only, never a model input)
    probs: tuple[float, ...] | None = None  # calibrated, ESGE order
    top: int | None = None  # station index of the argmax
    confidence: float | None = None  # probs[top]
    quality: float | None = None  # 0..1, quality.quality_score
    latency_ms: float = 0.0

    def evidence(self) -> FrameEvidence:
        return FrameEvidence(
            status=self.status,
            top=self.top,
            confidence=float(self.confidence or 0.0),
            quality=float(self.quality or 0.0),
        )


class BaseLandmarkClassifier(ABC):
    model_version: str = "unknown"
    tracker_params: TrackerParams = TrackerParams()
    auto_enabled: tuple[bool, ...] | None = None  # None = all stations

    @abstractmethod
    def predict(self, full_frame: np.ndarray) -> LandmarkFrameOutput:
        """Classify one full-resolution RGB processor frame."""


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _bundle_path_from_settings() -> Path:
    if settings.landmark_bundle_dir:
        logger.warning("Landmark bundle from PEACE_LANDMARK_BUNDLE_DIR=%s (BUNDLE.lock bypassed)",
                       settings.landmark_bundle_dir)
        return Path(settings.landmark_bundle_dir)
    return fetch_bundle(settings.landmark_bundle_lock, settings.landmark_cache_dir)


def _warmup_frame(layout: Layout) -> np.ndarray:
    """Deterministic textured full frame at the layout's reference size.

    Mid-range noise: not dark, saturated or red, and sharp, so the quality
    score runs through the sharp_ref division rather than an early zero.
    """
    rng = np.random.default_rng(0)
    return rng.integers(60, 200, (layout.ref_height, layout.ref_width, 3), dtype=np.uint8)


class LandmarkModelManager:
    """Singleton owning the verified landmark bundle.

    ``get_instance`` publishes a manager only after ``warmup()`` has run the
    full live path, so a loaded manager (and /health ``landmarks_loaded``)
    means frames can actually be served. A failed load or warm-up is cached:
    every later ``get_instance`` raises the same LandmarkUnavailable at once
    instead of re-hashing and re-loading per connection. ``reset()`` is the
    only way to retry (tests, explicit reload).
    """

    _instance: ClassVar[LandmarkModelManager | None] = None
    _failure: ClassVar[str | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, bundle: LandmarkBundle, channels_last: bool | None = None):
        """``channels_last=None`` picks NHWC for convnets on CUDA only: on the
        M1 CPU it made EfficientNet-B0 2-3x slower; bench.py --channels-last
        measures it on the serving CPU before anyone turns it on there."""
        import torch

        from app.ml.landmarks.architectures import CONV_ARCHS

        self.bundle = bundle
        self.version = bundle.version
        self.device = next(bundle.model.parameters()).device
        if channels_last is None:
            channels_last = self.device.type == "cuda"
        self._channels_last = bool(channels_last) and bundle.arch in CONV_ARCHS
        if self._channels_last:
            bundle.model.to(memory_format=torch.channels_last)

    @classmethod
    def get_instance(cls, bundle_dir: str | Path | None = None, device: str | None = None) -> LandmarkModelManager:
        inst = cls._instance
        if inst is not None:
            return inst
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            if cls._failure is not None:
                raise LandmarkUnavailable(cls._failure)
            try:
                from app.ml.real_models import ModelManager

                path = Path(bundle_dir) if bundle_dir else _bundle_path_from_settings()
                dev = ModelManager._resolve_device(device or settings.device)
                inst = cls(load_bundle(path, device=str(dev)))
                inst.warmup()
                cls._instance = inst
            except Exception as exc:
                cls._failure = str(exc) if isinstance(exc, LandmarkUnavailable) else (
                    f"landmark model failed to load: {type(exc).__name__}: {exc}"
                )
                logger.error("Landmark model unavailable: %s", cls._failure)
                raise LandmarkUnavailable(cls._failure) from exc
            return cls._instance

    @classmethod
    def peek(cls) -> LandmarkModelManager | None:
        """The loaded instance, without triggering a load."""
        return cls._instance

    @classmethod
    def failure(cls) -> str | None:
        return cls._failure

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._failure = None

    def forward(self, x) -> np.ndarray:
        """(3, S, S) normalised tensor -> float64 logits (NUM_STATIONS,)."""
        import torch

        batch = x.unsqueeze(0).to(self.device)
        if self._channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        with INFERENCE_LOCK, torch.inference_mode():
            logits, _ = self.bundle.model(batch)
        return logits[0].float().cpu().numpy().astype(np.float64)

    def warmup(self) -> None:
        """Classify a synthetic frame per served layout through the live path.

        Preprocessing, the quality gate with the bundle's parameters, the
        forward, calibration, the wire checks and the bundle's tracker all
        run once, so parameters that load but break serving raise here.
        """
        from app.ml.landmarks.payload import validate_output

        b = self.bundle
        clf = RealLandmarkClassifier(self)
        tracker = StationTracker(b.tracker, b.auto_enabled)
        for name in b.serve_layouts:
            layout = LAYOUTS_BY_NAME[name]
            out = clf.classify(_warmup_frame(layout), layout, verified=True)
            validate_output(out)
            tracker.update(out.evidence())


class RealLandmarkClassifier(BaseLandmarkClassifier):
    """Per-session adapter: layout lock + shared preprocessing + the bundle.

    status: low_quality if the quality gate fails (or the frame's layout was
    only inherited from the session lock), else uncertain if p_top < tau[top],
    else ok. Quality is measured at QUALITY_SIZE, the resolution sharp_ref and
    q_min were calibrated at, whatever the bundle's input_size.
    """

    def __init__(self, manager: LandmarkModelManager, layout_lock: LayoutLock | None = None):
        b = manager.bundle
        self._manager = manager
        self._layouts = layout_lock or LayoutLock()
        self._serve = frozenset(b.serve_layouts)
        self.model_version = b.version
        self.tracker_params = b.tracker
        self.auto_enabled = b.auto_enabled

    def predict(self, full_frame: np.ndarray) -> LandmarkFrameOutput:
        t0 = time.perf_counter()
        decision = self._layouts.resolve(full_frame)
        layout = decision.layout
        if layout is None or layout.name not in self._serve:
            return LandmarkFrameOutput("unsupported_layout", latency_ms=_ms(t0))
        return self.classify(full_frame, layout, decision.verified, t0)

    def classify(
        self, full_frame: np.ndarray, layout: Layout, verified: bool = True, t0: float | None = None
    ) -> LandmarkFrameOutput:
        """Classify a frame whose (served) layout is already resolved."""
        t0 = time.perf_counter() if t0 is None else t0
        b = self._manager.bundle
        mode = detect_mode(full_frame, layout)
        canvas = canonicalize(full_frame, layout)
        img = to_model_input(canvas, layout, b.input_size)
        qimg = img if b.input_size == QUALITY_SIZE else to_model_input(canvas, layout, QUALITY_SIZE)
        quality = quality_score(quality_metrics(qimg, layout, b.quality), b.quality)
        if not np.isfinite(quality):  # e.g. an empty inner aperture: treat as unusable
            quality = 0.0
        logits = self._manager.forward(normalize(img))
        probs = _softmax(logits / b.temperature)
        if not np.all(np.isfinite(probs)):
            raise FloatingPointError("non-finite landmark probabilities")
        top = int(np.argmax(probs))
        confidence = float(probs[top])
        if quality < b.q_min or not verified:
            status = "low_quality"
        elif confidence < b.tau[top]:
            status = "uncertain"
        else:
            status = "ok"
        return LandmarkFrameOutput(
            status=status,
            layout=layout.name,
            mode=mode,
            probs=tuple(float(p) for p in probs),
            top=top,
            confidence=confidence,
            quality=float(quality),
            latency_ms=_ms(t0),
        )
