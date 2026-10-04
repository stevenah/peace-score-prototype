"""Live station tracker: per-frame landmark evidence -> ESGE checklist state.

One instance per live connection (and per offline replay stream). The same
class runs in serving and in training/landmarks/stream_eval.py, so the logic
that is evaluated offline is exactly the logic that is deployed.

Rules (all counts are over EVALUATED frames; frames skipped by the every-N
sampler do not enter any window, so 3-of-6 stays 3-of-6 at 1 Hz):

- A frame *qualifies* for station k when its status is "ok" and its top
  class is k. Uncertain, low-quality, error and unsupported frames are
  evaluated but qualify for nothing. At most one station qualifies per frame.
- OBSERVED: k qualifies in >= evidence_needed of the last evidence_window
  evaluated frames AND k is auto-enabled. Observed is sticky (monotonic) and
  emits one ``observed`` event.
- CANDIDATE: k qualified at least once in the current evidence window but is
  not observed (always the case for stations with auto_enabled=False).
- CURRENT station: k qualifies in >= current_k of the last current_window
  evaluated frames; released to None after current_release consecutive
  evaluated frames in which the current station does not qualify.
- BEST FRAME: for an observed station, a qualifying frame whose
  confidence x quality beats the station's best by best_margin (and is at
  least best_cooldown evaluated frames after the last best) emits a
  ``best_frame`` event that always refers to THIS frame.

No anatomical order prior is used: in the data 97 of 98 patients revisit
stations, so an ordering prior would suppress legitimate observations.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Sequence

from app.ml.landmarks.stations import NUM_STATIONS, STATION_ORDER

UNSEEN, CANDIDATE, OBSERVED = "unseen", "candidate", "observed"

# Frame statuses that were evaluated (enter the windows) but carry no evidence.
_EVALUATED_NO_EVIDENCE = frozenset({"uncertain", "low_quality", "error", "unsupported_layout"})


@dataclass(frozen=True)
class TrackerParams:
    evidence_needed: int = 3
    evidence_window: int = 6
    current_k: int = 2
    current_window: int = 3
    current_release: int = 4
    best_margin: float = 1.05
    best_cooldown: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackerParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass(frozen=True)
class FrameEvidence:
    """What the tracker needs from one classified frame."""

    status: str  # ok | uncertain | low_quality | unsupported_layout | skipped | error
    top: int | None = None  # station index of the argmax, if classified
    confidence: float = 0.0
    quality: float = 0.0


@dataclass
class TrackerStep:
    current: str | None
    stations: list[str]
    auto_enabled: list[bool]
    observed_events: list[str] = field(default_factory=list)
    best_frame_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current": self.current,
            "stations": list(self.stations),
            "auto_enabled": list(self.auto_enabled),
            "events": {
                "observed": list(self.observed_events),
                "best_frame": list(self.best_frame_events),
            },
        }


class StationTracker:
    def __init__(
        self,
        params: TrackerParams = TrackerParams(),
        auto_enabled: Sequence[bool] | None = None,
    ):
        if auto_enabled is None:
            auto_enabled = (True,) * NUM_STATIONS
        if len(auto_enabled) != NUM_STATIONS:
            raise ValueError(f"auto_enabled must have {NUM_STATIONS} entries")
        self.params = params
        self.auto_enabled = [bool(a) for a in auto_enabled]
        self.reset()

    def reset(self) -> None:
        p = self.params
        self._evidence: deque[int | None] = deque(maxlen=p.evidence_window)
        self._recent: deque[int | None] = deque(maxlen=p.current_window)
        self._observed = [False] * NUM_STATIONS
        self._current: int | None = None
        self._misses = 0
        self._best = [0.0] * NUM_STATIONS
        self._last_best = [-(10**9)] * NUM_STATIONS
        self._n = 0  # evaluated frames so far

    # -- public -----------------------------------------------------------

    def update(self, ev: FrameEvidence | None) -> TrackerStep:
        """Advance by one frame. ``None`` or status "skipped" = not evaluated."""
        observed_events: list[str] = []
        best_events: list[str] = []
        if ev is not None and ev.status != "skipped":
            q = self._qualifying(ev)
            self._n += 1
            self._evidence.append(q)
            self._recent.append(q)
            self._update_current(q)
            if q is not None:
                if self._try_observe(q):
                    observed_events.append(STATION_ORDER[q])
                if self._try_best(q, ev):
                    best_events.append(STATION_ORDER[q])
        return self._step(observed_events, best_events)

    def state(self) -> TrackerStep:
        return self._step([], [])

    def state_dict(self) -> dict:
        """Full internal state, for determinism tests and debugging."""
        return {
            "evidence": list(self._evidence),
            "recent": list(self._recent),
            "observed": list(self._observed),
            "current": self._current,
            "misses": self._misses,
            "best": list(self._best),
            "last_best": list(self._last_best),
            "n": self._n,
        }

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _qualifying(ev: FrameEvidence) -> int | None:
        if ev.status != "ok" or ev.top is None:
            if ev.status not in _EVALUATED_NO_EVIDENCE and ev.status != "ok":
                raise ValueError(f"unknown frame status {ev.status!r}")
            return None
        if not 0 <= ev.top < NUM_STATIONS:
            raise ValueError(f"station index out of range: {ev.top}")
        return ev.top

    def _update_current(self, q: int | None) -> None:
        p = self.params
        counts = Counter(k for k in self._recent if k is not None)
        # Switch to (or confirm) the station with the most recent support.
        leaders = [k for k, c in counts.items() if c >= p.current_k]
        if leaders:
            # Most votes wins; ties go to the most recent qualifier.
            best = max(leaders, key=lambda k: (counts[k], self._last_index(k)))
            if best != self._current:
                self._current = best
            self._misses = 0 if q == self._current else self._misses + 1
        elif self._current is not None:
            self._misses = 0 if q == self._current else self._misses + 1
        if self._current is not None and self._misses >= p.current_release:
            self._current = None
            self._misses = 0

    def _last_index(self, k: int) -> int:
        for i in range(len(self._recent) - 1, -1, -1):
            if self._recent[i] == k:
                return i
        return -1

    def _try_observe(self, q: int) -> bool:
        if self._observed[q] or not self.auto_enabled[q]:
            return False
        if sum(1 for k in self._evidence if k == q) >= self.params.evidence_needed:
            self._observed[q] = True
            return True
        return False

    def _try_best(self, q: int, ev: FrameEvidence) -> bool:
        if not self._observed[q]:
            return False
        score = float(ev.confidence) * float(ev.quality)
        if score <= 0.0:
            return False
        if score > self._best[q] * self.params.best_margin and (
            self._n - self._last_best[q] >= self.params.best_cooldown
        ):
            self._best[q] = score
            self._last_best[q] = self._n
            return True
        return False

    def _step(self, observed_events: list[str], best_events: list[str]) -> TrackerStep:
        in_window = {k for k in self._evidence if k is not None}
        stations = [
            OBSERVED if self._observed[i] else (CANDIDATE if i in in_window else UNSEEN)
            for i in range(NUM_STATIONS)
        ]
        return TrackerStep(
            current=STATION_ORDER[self._current] if self._current is not None else None,
            stations=stations,
            auto_enabled=list(self.auto_enabled),
            observed_events=observed_events,
            best_frame_events=best_events,
        )
