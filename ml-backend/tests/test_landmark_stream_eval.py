"""Procedure Replay Benchmark (stream_sim.py / stream_eval.py) on synthetic logits.

Streams are built from generated per-frame predictions; the serving
``StationTracker`` is used unchanged. No dataset is read.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest

from app.ml.landmarks.layouts import OLYMPUS_1920
from app.ml.landmarks.preprocess import canonicalize
from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from app.ml.landmarks.tracker import FrameEvidence, StationTracker, TrackerParams
from training.landmarks.calibrate import PredSet
from training.landmarks.run_utils import softmax
from training.landmarks.stream_eval import aggregate, fast_observed, replay, station_records, stream_record
from training.landmarks.stream_sim import (
    PLACEHOLDER,
    TRANSIT_KINDS,
    StreamConfig,
    build_stream,
    build_streams,
    fill_transit,
    synth_transit,
)

PARAMS = QualityParams(sharp_ref=1000.0)
TAU = np.full(NUM_STATIONS, 0.6)
S = STATION_INDEX


def clip(uid, label, rel_t, dur=5.0, soft="", excl="", folder=None, predict=None, conf=8.0):
    """A clip spec: ``predict`` = station index the frames point to (default: the label)."""
    return {"uid": uid, "label": label, "rel_t": rel_t, "dur": dur, "soft": soft, "excl": excl,
            "folder": folder or label, "predict": S[label] if predict is None else predict, "conf": conf}


def predset(patient_clips: dict[str, list[dict]], fold: int = 0) -> PredSet:
    rows, logits = [], []
    for pid, clips in patient_clips.items():
        for c in clips:
            n = int(round(c["dur"] * 10))  # 10 fps cache
            for j in range(n):
                rows.append({"clip_uid": c["uid"], "src_idx": 3 * j, "u": j / max(n - 1, 1), "t_s": j / 10,
                             "sharpness": 1500.0, "dark_frac": 0.0, "sat_frac": 0.0, "redout": 0.0,
                             "layout_ok": True, "patient_id": pid, "label": c["label"], "soft_label": c["soft"],
                             "exclude_reason": c["excl"], "rel_t_s_f": c["rel_t"], "dur_s_f": c["dur"],
                             "rel_path": f"{c['folder']}/{pid}-{c['uid']}.mp4", "folder_class": c["folder"],
                             "primary": not c["excl"] and not c["soft"], "y": S[c["label"]]})
                z = np.zeros(NUM_STATIONS)
                z[c["predict"]] = c["conf"]
                logits.append(z)
    return PredSet(fold, pd.DataFrame(rows), np.asarray(logits))


def cfg(**kw) -> StreamConfig:
    return StreamConfig(**{"drop_p": 0.0, **kw})


# -- the fast observe rule equals the tracker --------------------------------------------------------------

def _tracker_observed(q: np.ndarray, params: TrackerParams, auto) -> np.ndarray:
    tr = StationTracker(params, auto)
    out = np.full(NUM_STATIONS, -1)
    for i, k in enumerate(q):
        ev = FrameEvidence("ok", int(k), 0.9, 0.9) if k >= 0 else FrameEvidence("uncertain", 0, 0.3, 0.9)
        for key in tr.update(ev).observed_events:
            out[S[key]] = i
    return out


@pytest.mark.parametrize("n,w", [(2, 4), (3, 6), (4, 8), (3, 4), (1, 1)])
def test_fast_observed_matches_station_tracker(n, w):
    rng = np.random.default_rng(n * 10 + w)
    params = TrackerParams(evidence_needed=n, evidence_window=w)
    for _ in range(150):
        length = int(rng.integers(1, 120))
        q = np.where(rng.random(length) < 0.4, -1, rng.integers(0, 4, length))
        auto = [bool(a) for a in rng.random(NUM_STATIONS) > 0.2]
        np.testing.assert_array_equal(fast_observed(q, n, w, auto), _tracker_observed(q, params, auto))


# -- stream construction ------------------------------------------------------------------------------------

def test_order_rate_and_gaps():
    ps = predset({"P1": [clip("c", "antrum", 100.0), clip("a", "z_line", 0.0), clip("b", "incisura", 20.0)]})
    s0 = build_stream(ps, "P1", cfg(scenario="S0"), 1.0, PARAMS)
    assert [c.clip_uid for c in s0.clips] == ["a", "b", "c"]  # OSD-time order
    assert (s0.kind == "clip").all() and len(s0) == 30  # 50 cached frames -> 10 at 2 Hz per clip
    s2 = build_stream(ps, "P1", cfg(scenario="S2"), 1.0, PARAMS)
    # gaps: b starts 15 s after a ends (30 frames); c 75 s after b ends -> capped at 30 s (60 frames)
    assert int((s2.kind == "transit").sum()) == 30 + 60
    assert np.all(np.diff(s2.t) > 0)
    s1 = build_stream(ps, "P1", cfg(scenario="S0", hz=1.0), 1.0, PARAMS)
    assert len(s1) == 15  # every 10th cached frame


def test_unknown_times_are_placed_deterministically():
    ps = predset({"P1": [clip("a", "z_line", 0.0), clip("b", "antrum", 10.0), clip("x", "incisura", np.nan)]})
    orders = {tuple(c.clip_uid for c in build_stream(ps, "P1", cfg(seed=s), 1.0, PARAMS).clips) for s in range(12)}
    assert all(set(o) == {"a", "b", "x"} for o in orders)
    assert all(o.index("a") < o.index("b") for o in orders)
    assert len(orders) > 1  # the unknown clip moves with the seed ...
    a = build_stream(ps, "P1", cfg(seed=3), 1.0, PARAMS)
    b = build_stream(ps, "P1", cfg(seed=3), 1.0, PARAMS)
    np.testing.assert_array_equal(a.t, b.t)  # ... and is reproducible for a seed


def test_s5_shuffles_and_s4_removes_the_station():
    clips = [clip(u, k, 10.0 * i) for i, (u, k) in enumerate(
        [("a", "esophagus_proximal"), ("b", "z_line"), ("c", "antrum"), ("d", "incisura"), ("e", "z_line")])]
    ps = predset({"P1": clips})
    s4 = build_stream(ps, "P1", cfg(scenario="S4", remove_station="z_line"), 1.0, PARAMS)
    assert [c.clip_uid for c in s4.clips] == ["a", "c", "d"]
    assert "z_line" not in s4.allowed
    orders = {tuple(c.clip_uid for c in build_stream(ps, "P1", cfg(scenario="S5", seed=s), 1.0, PARAMS).clips)
              for s in range(8)}
    assert len(orders) > 1


# -- ambiguous segments, recall and FDR ---------------------------------------------------------------------

def test_ambiguous_segments_count_for_either_station_and_leave_recall_denominators():
    ps = predset({"P1": [
        clip("a", "antrum", 0.0),
        clip("c", "incisura", 10.0, soft="lesser_curvature_retroflex:0.5|incisura:0.5",
             predict=S["lesser_curvature_retroflex"]),
        clip("k", "duodenum_descending", 20.0, excl="combo", folder="duodenum_descending"),
    ]})
    s = build_stream(ps, "P1", cfg(scenario="S0"), 1.0, PARAMS)
    assert s.present == {"antrum"}
    assert {"lesser_curvature_retroflex", "incisura", "duodenum_descending"} <= s.allowed
    rep = replay(s, TrackerParams(), TAU, 0.1)
    observed = {k for k, i in zip(S, rep.observed_at) if i >= 0}
    assert {"antrum", "lesser_curvature_retroflex"} <= observed
    rec = stream_record(s, rep)
    assert rec["false"] == 0 and rec["missing"] == 0
    st = pd.DataFrame(station_records(s, rep.observed_at))
    agg = aggregate(st, pd.DataFrame([rec]))
    assert set(agg["recall_per_station"]) == {"antrum"}  # only non-ambiguous stations have a denominator
    assert agg["recall_per_station"]["antrum"]["recall"] == 1.0


def test_s4_fdr_catches_a_confusable_neighbour():
    def patient(de_predicts):
        return [clip("z", "z_line", 0.0), clip("d", "esophagus_distal", 10.0, predict=de_predicts),
                clip("a", "antrum", 20.0)]

    for de_predicts, want in ((S["z_line"], 1.0), (S["esophagus_distal"], 0.0)):
        ps = predset({"P1": patient(de_predicts), "P2": patient(de_predicts)})
        streams = build_streams(ps, cfg(scenario="S4"), 1.0, PARAMS)
        fill_transit(streams, 1.0, PARAMS)
        st = []
        for s in streams:
            st += station_records(s, replay(s, TrackerParams(), TAU, 0.1).observed_at)
        agg = aggregate(pd.DataFrame(st), pd.DataFrame([{"transit_source": PLACEHOLDER}]))
        assert agg["fdr_per_station"]["z_line"]["fdr"] == want
        assert agg["fdr_per_station"]["antrum"]["fdr"] == 0.0


def test_false_observation_of_an_absent_station():
    ps = predset({"P1": [clip("a", "antrum", 0.0), clip("b", "antrum", 10.0, predict=S["incisura"])]})
    s = build_stream(ps, "P1", cfg(scenario="S0"), 1.0, PARAMS)
    rep = replay(s, TrackerParams(), TAU, 0.1)
    rec = stream_record(s, rep)
    assert rec["false"] == 1  # incisura observed, never present
    agg = aggregate(pd.DataFrame(station_records(s, rep.observed_at)), pd.DataFrame([rec]))
    assert agg["false_absent_rate"] > 0
    assert agg["best_frame_purity"] < 1.0  # incisura's best frame came from an antrum clip


def test_time_to_observe_and_quality_gate():
    ps = predset({"P1": [clip("a", "antrum", 0.0)]})
    s = build_stream(ps, "P1", cfg(scenario="S0"), 1.0, PARAMS)
    rep = replay(s, TrackerParams(evidence_needed=3), TAU, 0.1)
    rows = pd.DataFrame(station_records(s, rep.observed_at))
    tto = rows.loc[rows["station"] == "antrum", "tto"].iloc[0]
    assert tto == pytest.approx(s.t[2] - s.clips[0].t0)  # the third qualifying frame
    s.quality[:] = 0.0
    assert (replay(s, TrackerParams(), TAU, 0.1).observed_at < 0).all()  # gate on: nothing counts
    assert replay(s, TrackerParams(), TAU, 0.1, gate=False).observed_at[S["antrum"]] >= 0


# -- transit frames ----------------------------------------------------------------------------------------

def test_placeholder_transit_is_labelled_and_never_qualifies():
    ps = predset({"P1": [clip("a", "antrum", 0.0), clip("b", "incisura", 20.0)]})
    streams = build_streams(ps, cfg(scenario="S2"), 1.0, PARAMS)
    fill_transit(streams, 1.0, PARAMS)
    s = streams[0]
    assert s.transit_source == PLACEHOLDER and len(s.transit_idx) == 30
    rep = replay(s, TrackerParams(), TAU, 0.1)
    assert (rep.status[s.transit_idx] != "ok").all()
    assert stream_record(s, rep)["transit_not_ok"] == 30


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    from app.ml.landmarks.bench import synthetic_olympus_frame

    root = tmp_path_factory.mktemp("cache")
    canvases = [canonicalize(synthetic_olympus_frame(1920, seed=j), OLYMPUS_1920) for j in range(5)]
    for uid in ("a", "b", "c"):
        d = root / "frames" / uid
        d.mkdir(parents=True)
        for j in range(50):
            cv2.imwrite(str(d / f"{3 * j:05d}.jpg"), cv2.cvtColor(canvases[j % 5], cv2.COLOR_RGB2BGR))
    return root


def test_model_transit_frames_are_generated_and_classified(tiny_cache):
    ps = predset({"P1": [clip("a", "antrum", 0.0), clip("b", "incisura", 8.0)]})
    streams = build_streams(ps, cfg(scenario="S2"), 1.0, PARAMS)
    seen = []

    def hook(canvases, layouts):
        seen.extend(canvases)
        z = np.zeros((len(canvases), NUM_STATIONS))
        z[:, S["corpus_greater_curvature"]] = 9.0
        return z

    fill_transit(streams, 1.0, PARAMS, hook, cache_root=tiny_cache)
    s = streams[0]
    assert s.transit_source == "model" and len(seen) == len(s.transit_idx) == 6
    assert (s.probs[s.transit_idx].argmax(1) == S["corpus_greater_curvature"]).all()
    assert all(c.shape == (320, 320, 3) and c.dtype == np.uint8 for c in seen)
    assert ((s.quality[s.transit_idx] >= 0) & (s.quality[s.transit_idx] <= 1)).all()


@pytest.mark.parametrize("kind", TRANSIT_KINDS)
def test_synthetic_transit_kinds_are_deterministic(kind):
    from app.ml.landmarks.bench import synthetic_olympus_frame

    canvas = canonicalize(synthetic_olympus_frame(1920, seed=1), OLYMPUS_1920)
    a = synth_transit(canvas, kind, np.random.default_rng(5))
    b = synth_transit(canvas, kind, np.random.default_rng(5))
    assert a.dtype == np.uint8 and a.shape == canvas.shape
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, canvas)


def test_every_scenario_builds(tiny_cache):
    ps = predset({"P1": [clip("a", "antrum", 0.0), clip("b", "incisura", 8.0)],
                  "P2": [clip("c", "z_line", 0.0)]})
    for scen in ("S0", "S2", "S4", "S5"):
        for hz in (2.0, 1.0):
            streams = build_streams(ps, cfg(scenario=scen, hz=hz, drop_p=0.05), 1.0, PARAMS)
            fill_transit(streams, 1.0, PARAMS)
            assert streams and all(replace(s.cfg).scenario == scen for s in streams)


# -- transit sources: only clips already replayed (review fix) ------------------------------------------------

def _station_of(ps: PredSet) -> dict[str, frozenset]:
    from training.landmarks.stream_sim import clip_table

    return dict(clip_table(ps)["stations"])


def test_s4_transit_is_never_made_from_the_removed_station():
    ps = predset({"P1": [
        clip("a", "esophagus_proximal", 0.0), clip("z1", "z_line", 10.0), clip("c", "antrum", 20.0),
        clip("s", "incisura", 30.0, soft="lesser_curvature_retroflex:0.5|incisura:0.5"),
        clip("z2", "z_line", 40.0), clip("d", "corpus_greater_curvature", 50.0),
    ]})
    stations = _station_of(ps)
    for removed in ("z_line", "incisura", "lesser_curvature_retroflex"):
        for seed in range(6):
            s = build_stream(ps, "P1", cfg(scenario="S4", remove_station=removed, seed=seed), 1.0, PARAMS)
            assert s.transit_src, "the gaps must produce transit frames"
            assert all(removed not in stations[uid] for uid, *_ in s.transit_src)
            assert {uid for uid, *_ in s.transit_src} <= {c.clip_uid for c in s.clips}


def test_s2_transit_never_previews_a_later_clip():
    ps = predset({"P1": [clip(u, k, 10.0 * i) for i, (u, k) in enumerate(
        [("a", "esophagus_proximal"), ("b", "z_line"), ("c", "antrum"), ("d", "duodenal_bulb"), ("e", "incisura")])]})
    for scen in ("S2", "S5"):
        for seed in range(6):
            s = build_stream(ps, "P1", cfg(scenario=scen, seed=seed), 1.0, PARAMS)
            end = {c.clip_uid: c.t1 for c in s.clips}
            for i, (uid, *_) in zip(s.transit_idx, s.transit_src):
                assert end[uid] <= s.t[i]  # the source clip was fully replayed before this transit frame


def test_observations_on_transit_frames_are_reported_separately():
    ps = predset({"P1": [clip("a", "antrum", 0.0), clip("b", "incisura", 20.0)]})
    s = build_stream(ps, "P1", cfg(scenario="S2"), 1.0, PARAMS)
    fill_transit([s], 1.0, PARAMS)
    tr = s.transit_idx
    s.probs[tr] = np.eye(NUM_STATIONS)[S["antrum"]] * 0.9 + 0.01  # transit that looks like antrum ...
    s.quality[tr] = 1.0
    s.probs[s.kind == "clip"] = np.eye(NUM_STATIONS)[S["incisura"]] * 0.9 + 0.01  # ... and clips that do not
    rep = replay(s, TrackerParams(), TAU, 0.1)
    rows = pd.DataFrame(station_records(s, rep.observed_at))
    assert rows.set_index("station").loc["antrum", "on_transit"]
    assert not rows.set_index("station").loc["incisura", "on_transit"]
    agg = aggregate(rows, pd.DataFrame([stream_record(s, rep)]))
    assert agg["observed_on_transit"] == 0.5


def test_with_temperature_recalibrates_from_the_stored_logits():
    ps = predset({"P1": [clip("a", "antrum", 0.0)]})
    s = build_stream(ps, "P1", cfg(scenario="S0"), 1.0, PARAMS)
    hot = s.with_temperature(4.0)
    np.testing.assert_allclose(hot.probs, softmax(s.logits, 4.0))
    assert hot.probs.max() < s.probs.max() and hot.logits is s.logits


# -- auto_enabled in the stream metrics ------------------------------------------------------------------------

def test_disabled_stations_count_in_no_stream_metric():
    ps = predset({"P1": [clip("a", "antrum", 0.0), clip("b", "incisura", 10.0), clip("c", "z_line", 20.0)],
                  "P2": [clip("d", "antrum", 0.0), clip("e", "incisura", 10.0), clip("f", "z_line", 20.0,
                                                                                          predict=S["incisura"])]})
    auto = [k != "incisura" for k in STATION_ORDER]
    st, sr, st4 = [], [], []
    for s in build_streams(ps, cfg(scenario="S0"), 1.0, PARAMS):
        rep = replay(s, TrackerParams(), TAU, 0.1, auto)
        assert rep.observed_at[S["incisura"]] < 0
        st += station_records(s, rep.observed_at, auto_enabled=auto,
                              observed_if_enabled=fast_observed(rep.q, 3, 6))
        sr.append(stream_record(s, rep))
    agg = aggregate(pd.DataFrame(st), pd.DataFrame(sr), auto)
    assert "incisura" not in agg["recall_per_station"] and agg["never_enabled"] == ["incisura"]
    assert agg["recall_mean"] == pytest.approx(np.mean([v["recall"] for v in agg["recall_per_station"].values()]))
    for s in build_streams(ps, cfg(scenario="S4"), 1.0, PARAMS):
        fill_transit([s], 1.0, PARAMS)
        rep = replay(s, TrackerParams(), TAU, 0.1, auto)
        st4 += station_records(s, rep.observed_at, auto_enabled=auto, observed_if_enabled=fast_observed(rep.q, 3, 6))
    a4 = aggregate(pd.DataFrame(st4), pd.DataFrame([{"transit_source": PLACEHOLDER}]), auto)
    assert "incisura" not in a4["fdr_per_station"]  # never observed live: not a live false discovery ...
    assert a4["fdr_per_station_if_enabled"]["incisura"]["fdr"] > 0  # ... but it WOULD be (G-STATION input)


# -- nested tuning: auto_enabled per outer fold (critic C4 / review fix) --------------------------------------

def rich_predset(fold: int, n_patients: int = 5, bad: str | None = None, seed: int = 0) -> PredSet:
    """A fold of patients with all 10 stations; ``bad`` is always predicted as its neighbour."""
    rng = np.random.default_rng(seed)
    clips = {}
    for p in range(n_patients):
        pid = f"HT9{fold}{p}"
        cl = []
        for i, key in enumerate(STATION_ORDER):
            pred = S[key] if key != bad else (S[key] + 1) % NUM_STATIONS
            cl.append(clip(f"f{fold}p{p}s{i}", key, 12.0 * i, predict=pred, conf=float(rng.uniform(6, 9))))
        clips[pid] = cl
    ps = predset(clips, fold=fold)
    f = ps.frames
    f["group_id"], f["mode_major"], f["nbi_frac_f"] = f["patient_id"], "wl", 0.0
    f["cap_b"], f["room_hash"], f["cohort"], f["fold_i"] = False, "r1", "new_1920", fold
    ps.logits = ps.logits + rng.normal(0, 0.3, ps.logits.shape)
    return ps


@pytest.fixture(scope="module")
def tuned():
    from training.landmarks.run_utils import GATES_PATH, read_json
    from training.landmarks.tune_tracker import tune

    gates = read_json(GATES_PATH)
    gates["G-STATION"] = {**gates["G-STATION"], "per_mode_stations": []}
    oof = {k: rich_predset(k, bad="antrum" if k != 2 else None, seed=k) for k in range(3)}
    temps = {k: 1.0 for k in oof}
    tables: dict = {}
    out = tune(oof, temps, PARAMS, seeds=(0,), log=lambda *_: None, gates=gates, n_boot=50, tables=tables)
    return out, tables


def test_tune_recomputes_auto_enabled_per_outer_fold(tuned):
    out, _ = tuned
    auto = {k: dict(zip(STATION_ORDER, v["auto_enabled"])) for k, v in out["per_fold"].items()}
    # antrum is broken in folds 0 and 1: every fold's OTHER folds include a broken one -> manual-only ...
    assert not any(a["antrum"] for a in auto.values())
    # ... while the healthy stations stay enabled, decided on the other folds only
    assert all(a["z_line"] and a["corpus_greater_curvature"] for a in auto.values())
    assert out["per_fold"]["0"]["station_gate"]["antrum"]["pass"] is False


def test_heldout_scores_apply_the_fold_auto_enabled(tuned):
    out, tables = tuned
    s2 = out["heldout_pooled"]["S2"]
    assert "antrum" not in s2["recall_per_station"] and "antrum" in s2["never_enabled"]
    st = tables["S2"]
    assert not st.loc[st["station"] == "antrum", "observed"].any()
    assert set(st.loc[st["station"] == "antrum", "auto"]) == {False}
    s4 = out["heldout_pooled"]["S4"]
    assert "antrum" not in s4["fdr_per_station"] and "antrum" in s4["fdr_per_station_if_enabled"]
    assert out["force_disabled"] == []


def test_forced_stations_and_heldout_temperatures():
    from training.landmarks.run_utils import GATES_PATH, read_json
    from training.landmarks.tune_tracker import tune

    gates = read_json(GATES_PATH)
    gates["G-STATION"] = {**gates["G-STATION"], "per_mode_stations": []}
    oof = {k: rich_predset(k, n_patients=4, seed=10 + k) for k in range(3)}
    held = {k: {j: (1.5 if j == (k - 1) % 3 else 1.0) for j in range(3)} for k in range(3)}
    out = tune(oof, {k: 1.0 for k in oof}, PARAMS, seeds=(0,), log=lambda *_: None, gates=gates, n_boot=20,
               force_disabled=["z_line"], temps_heldout=held)
    assert out["force_disabled"] == ["z_line"]
    assert all(not v["auto_enabled"][S["z_line"]] for v in out["per_fold"].values())
    assert all(v["auto_enabled"][S["antrum"]] for v in out["per_fold"].values())
    for k, v in out["per_fold"].items():  # fold k is tuned with the temperatures that never saw fold k
        assert v["temperatures"] == {str(j): t for j, t in held[int(k)].items()}


# -- the locked test replays like its dev reference (review fix) ------------------------------------------------

def _exp(tmp_path, transit_source, with_gates=True):
    from training.landmarks.run_utils import write_json

    exp = tmp_path / "exp"
    auto = [k != "incisura" for k in STATION_ORDER]
    ref = {"S2": {"recall_mean": 0.5}, "S4": {"fdr_overall": 0.05}}
    write_json(exp / "tracker.json", {"transit_source": transit_source, "heldout_pooled": ref,
                                      "shipped": {"tracker": TrackerParams().to_dict()},
                                      "shipped_calibration": {"T": 1.0, "tau": [0.6] * NUM_STATIONS, "q_min": 0.1}})
    write_json(exp / "calibration.json", {"quality_params": PARAMS.to_dict()})
    write_json(exp / "eval_oof.json", {"clip": {"macro_f1": {"value": 0.5}}})
    if with_gates:
        write_json(exp / "gates_result.json", {"auto_enabled": auto})
    return exp


@pytest.mark.parametrize("dev_source,transit,gates,why", [
    ([PLACEHOLDER], "model", True, "transit source differs"),
    (["model"], "placeholder", True, "transit source differs"),
    (["model"], "model", False, "gates_result.json"),
])
def test_locked_test_refuses_a_protocol_mismatch_before_looking(tmp_path, dev_source, transit, gates, why):
    from training.landmarks.evaluate import score_test

    exp = _exp(tmp_path, dev_source, gates)
    with pytest.raises(SystemExit, match=why):
        score_test(exp, transit=transit, hook=lambda c, lay: np.zeros((len(c), NUM_STATIONS)))
    assert not (exp / "TEST_SCORED").exists()  # the one look is not spent


def test_locked_test_classifies_transit_with_the_final_model_and_ships_auto_enabled(tmp_path, tiny_cache,
                                                                                     monkeypatch):
    from training.landmarks import evaluate as E

    exp = _exp(tmp_path, ["model"])
    ps = predset({"HT990": [clip("a", "antrum", 0.0), clip("b", "incisura", 8.0), clip("c", "z_line", 16.0)]})
    f = ps.frames
    f["group_id"], f["mode_major"], f["nbi_frac_f"], f["cap_b"] = f["patient_id"], "wl", 0.0, False
    f["room_hash"], f["cohort"] = "r1", "new_1920"
    monkeypatch.setattr(E, "load_predset", lambda path, m: ps)
    seen = []

    def hook(canvases, layouts):
        seen.append(len(canvases))
        z = np.zeros((len(canvases), NUM_STATIONS))
        z[:, S["corpus_greater_curvature"]] = 9.0
        return z

    m = pd.DataFrame({"clip_uid": ["a", "b", "c"], "layout": OLYMPUS_1920.name})
    out = E.score_test(exp, hook=hook, manifest=m, cache_root=tiny_cache)
    assert seen and out["transit_source"] == {"test": "model", "dev_reference": ["model"]}
    assert out["auto_enabled"][S["incisura"]] is False
    assert "incisura" not in out["S2"]["recall_per_station"]  # the shipped station set, like the reference
    assert (exp / "TEST_SCORED").exists() and (exp / "eval_test.json").exists()
    with pytest.raises(SystemExit, match="already scored"):
        E.score_test(exp, hook=hook, manifest=m, cache_root=tiny_cache)
