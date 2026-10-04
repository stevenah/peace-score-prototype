"""Per-session helpers for the live landmark path.

``LayoutLock`` keeps landmark availability stable within one live session:
single frames that fail layout verification (menus, overlays, the scope
outside the patient) must not flip the checklist on and off. ``SessionStats``
collects the per-session observability summary logged at disconnect.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.ml.landmarks.layouts import MIN_SERVED_WIDTH, Layout, detect_layout


@dataclass(frozen=True)
class LayoutDecision:
    layout: Layout | None  # layout to preprocess with; None -> unsupported_layout
    verified: bool  # this frame itself passed detect_layout


class LayoutLock:
    """Layout lock with hysteresis.

    - A frame that passes ``detect_layout`` for a served layout is always
      used with that layout. After ``lock_after`` consecutive such frames of
      the same layout the session is LOCKED to it.
    - While locked, a frame that fails verification but has a compatible
      shape is still processed with the locked layout, flagged unverified
      (the classifier caps it at ``low_quality``, so it is never evidence).
    - Only ``release_after`` consecutive failures release the lock and
      report ``unsupported_layout``. Before the first lock there is nothing to
      fall back to, so failing frames are unsupported straight away.
    """

    def __init__(self, lock_after: int = 3, release_after: int = 5):
        self.lock_after = lock_after
        self.release_after = release_after
        self.reset()

    def reset(self) -> None:
        self.locked: Layout | None = None
        self._cand: Layout | None = None
        self._cand_n = 0
        self._fails = 0

    def resolve(self, frame: np.ndarray) -> LayoutDecision:
        det = detect_layout(frame)
        if det is not None and det.served:
            self._fails = 0
            self._cand_n = self._cand_n + 1 if det is self._cand else 1
            self._cand = det
            if self._cand_n >= self.lock_after:
                self.locked = det
            return LayoutDecision(det, True)

        self._fails += 1
        self._cand, self._cand_n = None, 0
        if self.locked is not None and self._fails >= self.release_after:
            self.locked = None
        if self.locked is not None and _compatible(frame, self.locked):
            return LayoutDecision(self.locked, False)
        return LayoutDecision(None, False)


def _compatible(frame: np.ndarray, layout: Layout) -> bool:
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    h, w = frame.shape[:2]
    return w >= MIN_SERVED_WIDTH and abs(w / h - layout.ref_width / layout.ref_height) < 0.02


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


class SessionStats:
    """Structured per-session summary for the disconnect log line."""

    def __init__(self) -> None:
        self.frames = 0
        self.frame_errors = 0
        self.landmark_frames = 0
        self.statuses: Counter[str] = Counter()
        self.layouts: Counter[str] = Counter()
        self.modes: Counter[str] = Counter()
        self.landmark_ms: list[float] = []
        self.processing_ms: list[float] = []
        self.model_version: str | None = None
        self.display: bool | None = None
        self.observed = 0
        self.stats_errors = 0

    def record_frame(self, processing_ms: float, landmark: dict | None) -> None:
        """Never raises: bookkeeping must not be able to end a live session."""
        self.frames += 1
        try:
            self._record(float(processing_ms), landmark)
        except Exception:
            self.stats_errors += 1

    def _record(self, processing_ms: float, landmark: dict | None) -> None:
        self.processing_ms.append(processing_ms)
        if landmark is None:
            return
        self.landmark_frames += 1
        status = str(landmark.get("status"))
        self.statuses[status] += 1
        if "model_version" in landmark:
            self.model_version = str(landmark["model_version"])
        if "display" in landmark:
            self.display = bool(landmark["display"])
        if landmark.get("layout"):
            self.layouts[str(landmark["layout"])] += 1
        if landmark.get("mode"):
            self.modes[str(landmark["mode"])] += 1
        if status != "skipped" and landmark.get("landmark_ms") is not None:
            self.landmark_ms.append(float(landmark["landmark_ms"]))
        stations = landmark.get("stations") or []
        self.observed = max(self.observed, sum(1 for s in stations if s == "observed"))

    def record_error(self) -> None:
        self.frames += 1
        self.frame_errors += 1

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "frame_errors": self.frame_errors,
            "processing_ms_p50": _pct(self.processing_ms, 50),
            "processing_ms_p95": _pct(self.processing_ms, 95),
            "landmark_frames": self.landmark_frames,
            "landmark_status": dict(sorted(self.statuses.items())),
            "landmark_errors": self.statuses.get("error", 0),
            "landmark_ms_p50": _pct(self.landmark_ms, 50),
            "landmark_ms_p95": _pct(self.landmark_ms, 95),
            "model_version": self.model_version,
            "display": self.display,
            "layout": self.layouts.most_common(1)[0][0] if self.layouts else None,
            "modes": dict(sorted(self.modes.items())),
            "stations_observed": self.observed,
            "stats_errors": self.stats_errors,
        }
