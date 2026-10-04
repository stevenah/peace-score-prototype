"""StationTracker: evidence rule, stickiness, current station, best frames."""

from __future__ import annotations

import pytest

from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from app.ml.landmarks.tracker import (
    CANDIDATE,
    OBSERVED,
    UNSEEN,
    FrameEvidence,
    StationTracker,
    TrackerParams,
)

ANTRUM = STATION_INDEX["antrum"]
BULB = STATION_INDEX["duodenal_bulb"]


def ok(k: int, conf: float = 0.9, quality: float = 0.8) -> FrameEvidence:
    return FrameEvidence(status="ok", top=k, confidence=conf, quality=quality)


UNCERTAIN = FrameEvidence(status="uncertain", top=ANTRUM, confidence=0.4, quality=0.9)
LOW_Q = FrameEvidence(status="low_quality")
SKIP = FrameEvidence(status="skipped")


def run(tracker: StationTracker, frames) -> list:
    return [tracker.update(f) for f in frames]


def test_three_of_six_observes_exactly_once():
    t = StationTracker()
    steps = run(t, [ok(ANTRUM), UNCERTAIN, ok(ANTRUM), LOW_Q, ok(ANTRUM), ok(ANTRUM), ok(ANTRUM)])
    observed_at = [i for i, s in enumerate(steps) if "antrum" in s.observed_events]
    assert observed_at == [4]
    assert steps[-1].stations[ANTRUM] == OBSERVED


def test_two_of_six_is_only_a_candidate():
    t = StationTracker()
    steps = run(t, [ok(ANTRUM), UNCERTAIN, LOW_Q, ok(ANTRUM), LOW_Q, LOW_Q])
    assert steps[-1].stations[ANTRUM] == CANDIDATE
    assert not any(s.observed_events for s in steps)
    # Once the evidence leaves the window the candidate reverts to unseen.
    steps = run(t, [LOW_Q] * 6)
    assert steps[-1].stations[ANTRUM] == UNSEEN


def test_uncertain_and_low_quality_never_vote():
    t = StationTracker()
    steps = run(t, [UNCERTAIN] * 10 + [LOW_Q] * 10)
    assert all(s == UNSEEN for s in steps[-1].stations)
    assert steps[-1].current is None


def test_disabled_station_is_candidate_at_most():
    auto = [True] * NUM_STATIONS
    auto[ANTRUM] = False
    t = StationTracker(auto_enabled=auto)
    steps = run(t, [ok(ANTRUM)] * 10)
    assert steps[-1].stations[ANTRUM] == CANDIDATE
    assert not any(s.observed_events or s.best_frame_events for s in steps)
    assert steps[-1].auto_enabled[ANTRUM] is False


def test_observed_is_sticky():
    t = StationTracker()
    run(t, [ok(ANTRUM)] * 3)
    steps = run(t, [ok(BULB)] * 20 + [LOW_Q] * 20)
    assert steps[-1].stations[ANTRUM] == OBSERVED
    assert steps[-1].stations[BULB] == OBSERVED


def test_current_station_switches_and_releases():
    t = StationTracker(TrackerParams(current_k=2, current_window=3, current_release=4))
    steps = run(t, [ok(ANTRUM), ok(ANTRUM)])
    assert steps[-1].current == "antrum"
    steps = run(t, [ok(BULB), ok(BULB)])
    assert steps[-1].current == "duodenal_bulb"
    steps = run(t, [LOW_Q] * 3)
    assert steps[-1].current == "duodenal_bulb"
    steps = run(t, [LOW_Q])
    assert steps[-1].current is None


def test_skipped_frames_do_not_enter_windows():
    """every_n=2 must keep 3-of-6 meaning six evaluated frames, not three."""
    a, b = StationTracker(), StationTracker()
    evaluated = [ok(ANTRUM), LOW_Q, ok(ANTRUM), LOW_Q, LOW_Q, LOW_Q, ok(ANTRUM)]
    interleaved = [x for f in evaluated for x in (f, SKIP)]
    last_a = run(a, evaluated)[-1]
    last_b = [s for s in run(b, interleaved)][-1]
    assert last_a.stations == last_b.stations
    assert a.state_dict() == b.state_dict()
    assert b.update(None).stations == last_b.stations  # None == skipped


def test_best_frame_needs_margin_and_cooldown():
    t = StationTracker(TrackerParams(best_margin=1.05, best_cooldown=2))
    steps = run(t, [ok(ANTRUM, 0.9, 0.5)] * 3)
    assert steps[2].best_frame_events == ["antrum"]  # first frame once observed
    # Better but within cooldown -> no event; after cooldown -> event.
    s = t.update(ok(ANTRUM, 0.95, 0.9))
    assert s.best_frame_events == []
    s = t.update(ok(ANTRUM, 0.96, 0.9))
    assert s.best_frame_events == ["antrum"]
    # Not better by the margin -> no event.
    t.update(LOW_Q)
    s = t.update(ok(ANTRUM, 0.97, 0.9))
    assert s.best_frame_events == []


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        StationTracker(auto_enabled=[True] * 3)
    t = StationTracker()
    with pytest.raises(ValueError):
        t.update(FrameEvidence(status="banana"))
    with pytest.raises(ValueError):
        t.update(ok(NUM_STATIONS))


def test_deterministic_and_resettable():
    frames = [ok(i % NUM_STATIONS) for i in range(40)] + [ok(ANTRUM)] * 5
    a, b = StationTracker(), StationTracker()
    assert [s.to_dict() for s in run(a, frames)] == [s.to_dict() for s in run(b, frames)]
    a.reset()
    assert a.state_dict() == StationTracker().state_dict()


def test_scripted_clinical_flow_observes_every_present_station():
    order = ["esophagus_proximal", "esophagus_distal", "z_line", "duodenal_bulb",
             "duodenum_descending", "antrum", "cardia_fundus_retroflex",
             "lesser_curvature_retroflex", "incisura", "corpus_greater_curvature"]
    frames = []
    for key in order:
        frames += [ok(STATION_INDEX[key])] * 8 + [LOW_Q] * 4 + [UNCERTAIN] * 2
    t = StationTracker()
    steps = run(t, frames)
    assert steps[-1].stations == [OBSERVED] * NUM_STATIONS
    events = [e for s in steps for e in s.observed_events]
    assert events == list(STATION_ORDER)


def test_payload_dict_shape():
    d = StationTracker().update(ok(ANTRUM)).to_dict()
    assert set(d) == {"current", "stations", "auto_enabled", "events"}
    assert set(d["events"]) == {"observed", "best_frame"}
    assert len(d["stations"]) == NUM_STATIONS == len(d["auto_enabled"])
