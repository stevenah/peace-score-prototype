"""The ``landmark`` block of a live frame result (wire schema esge10.v1).

See contracts/live_frame_result.v2.example.json. Field presence:

- always: schema, model_version, display, status, current, stations,
  auto_enabled, events, landmark_ms
- probs, top, confidence, quality: iff status is ok | uncertain | low_quality
- layout, mode: iff the screen layout was recognised

Every value is a plain Python str/int/float/bool (numpy scalars leak through
``json.dumps`` as TypeErrors and NaN as invalid JSON, which the browser then
drops together with the PEACE fields). The block is checked with
``json.dumps(allow_nan=False)``; if anything is off, an ``error`` block with
the tracker state is returned instead, so a landmark fault never costs PEACE.
``validate_output`` runs the same per-frame checks without a tracker step; the
pipeline calls it first, so a frame the wire rejects is never evidence.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Mapping

from app.ml.landmarks.stations import NUM_STATIONS, SCHEMA, STATION_INDEX, STATION_ORDER
from app.ml.landmarks.tracker import CANDIDATE, OBSERVED, UNSEEN, TrackerStep

logger = logging.getLogger(__name__)

STATUSES = ("ok", "uncertain", "low_quality", "unsupported_layout", "skipped", "error")
CLASSIFIED = frozenset({"ok", "uncertain", "low_quality"})
MODES = frozenset({"wl", "nbi", "rdi"})
_STATION_STATES = frozenset({UNSEEN, CANDIDATE, OBSERVED})


def _num(x: Any, ndigits: int, lo: float | None = None, hi: float | None = None) -> float:
    v = round(float(x), ndigits)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {x!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f"value {v} outside [{lo}, {hi}]")
    return v


def _key(k: Any) -> str:
    k = str(k)
    if k not in STATION_INDEX:
        raise ValueError(f"unknown station {k!r}")
    return k


def _version(meta: Mapping[str, Any] | None) -> str:
    try:
        return str((meta or {}).get("model_version") or "unknown")
    except Exception:
        return "unknown"


def _tracker_fields(step: TrackerStep) -> dict:
    stations = [str(s) for s in step.stations]
    auto = [bool(a) for a in step.auto_enabled]
    if len(stations) != NUM_STATIONS or len(auto) != NUM_STATIONS:
        raise ValueError("tracker arrays must have one entry per station")
    if any(s not in _STATION_STATES for s in stations):
        raise ValueError(f"bad station state in {stations}")
    return {
        "current": None if step.current is None else _key(step.current),
        "stations": stations,
        "auto_enabled": auto,
        "events": {
            "observed": [_key(k) for k in step.observed_events],
            "best_frame": [_key(k) for k in step.best_frame_events],
        },
    }


def _blank_tracker() -> dict:
    return {
        "current": None,
        "stations": [UNSEEN] * NUM_STATIONS,
        "auto_enabled": [False] * NUM_STATIONS,
        "events": {"observed": [], "best_frame": []},
    }


def _frame_fields(out) -> dict:
    """status, layout/mode and the classification fields of one frame output."""
    status = str(out.status)
    if status not in STATUSES:
        raise ValueError(f"unknown landmark status {status!r}")
    block: dict[str, Any] = {"status": status}
    if out.layout is not None:
        mode = str(out.mode)
        if mode not in MODES:
            raise ValueError(f"unknown mode {out.mode!r}")
        block["layout"] = str(out.layout)
        block["mode"] = mode
    if status in CLASSIFIED:
        probs = [_num(p, 4, 0.0, 1.0) for p in out.probs]
        if len(probs) != NUM_STATIONS:
            raise ValueError(f"expected {NUM_STATIONS} probs, got {len(probs)}")
        top = int(out.top)
        if not 0 <= top < NUM_STATIONS:
            raise ValueError(f"top index {top} out of range")
        block["probs"] = probs
        block["top"] = STATION_ORDER[top]
        block["confidence"] = _num(out.confidence, 3, 0.0, 1.0)
        block["quality"] = _num(out.quality, 3, 0.0, 1.0)
    return block


def validate_output(out) -> None:
    """Raise ValueError if ``build_payload`` would turn ``out`` into an error block.

    The pipeline calls this BEFORE feeding the tracker, so a frame the wire
    rejects is never counted as evidence (a NaN or out-of-range output must
    not tick a station inside a ``status: "error"`` block).
    """
    try:
        json.dumps(_frame_fields(out), allow_nan=False)
        _num(out.latency_ms, 3, 0.0)
    except Exception as exc:
        raise ValueError(f"invalid landmark output: {exc}") from exc


def _build(out, step: TrackerStep, meta: Mapping[str, Any] | None, display: bool) -> dict:
    block: dict[str, Any] = {
        "schema": SCHEMA,
        "model_version": _version(meta),
        "display": bool(display),
        **_frame_fields(out),
    }
    block.update(_tracker_fields(step))
    block["landmark_ms"] = _num(out.latency_ms, 3, 0.0)
    return block


def build_payload(out, step: TrackerStep, meta: Mapping[str, Any] | None, display: bool) -> dict:
    """Wire block for one frame; never raises (falls back to ``error``)."""
    try:
        block = _build(out, step, meta, display)
        json.dumps(block, allow_nan=False)
        return block
    except Exception:
        logger.exception("Landmark payload rejected; sending status=error")
        return error_payload(step, meta, display, getattr(out, "latency_ms", 0.0))


def error_payload(
    step: TrackerStep | None,
    meta: Mapping[str, Any] | None,
    display: bool,
    landmark_ms: Any = 0.0,
) -> dict:
    """``status: "error"`` block carrying whatever tracker state is intact."""
    try:
        tracker = _tracker_fields(step) if step is not None else _blank_tracker()
    except Exception:
        tracker = _blank_tracker()
    try:
        ms = _num(landmark_ms, 3, 0.0)
    except Exception:
        ms = 0.0
    try:
        shown = bool(display)
    except Exception:
        shown = False
    return {
        "schema": SCHEMA,
        "model_version": _version(meta),
        "display": shown,
        "status": "error",
        **tracker,
        "landmark_ms": ms,
    }
