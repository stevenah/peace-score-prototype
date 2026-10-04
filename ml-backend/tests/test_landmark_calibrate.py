"""Landmark calibration (training/landmarks/calibrate.py) on synthetic logits.

No dataset is read: every PredSet here is generated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from app.ml.landmarks.quality import QualityMetrics, QualityParams, quality_score  # noqa: E402
from app.ml.landmarks.stations import NUM_STATIONS  # noqa: E402
from training.landmarks import dataset as D  # noqa: E402
from training.landmarks.calibrate import (  # noqa: E402
    PredSet,
    crossfit,
    ece,
    fit_q_min,
    fit_tau,
    fit_temperature,
    fit_thresholds,
    frame_status,
    nll,
    shipped,
)
from training.landmarks.run_utils import softmax  # noqa: E402

PARAMS = QualityParams(sharp_ref=1000.0)


def _sample_labels(logits: np.ndarray, t: float, rng) -> np.ndarray:
    p = softmax(logits, t)
    c = p.cumsum(1)
    r = rng.random((len(p), 1))
    return np.minimum((r > c).sum(1), NUM_STATIONS - 1)


def synthetic_predset(fold: int, n_clips: int = 60, frames_per_clip: int = 20, t_true: float = 2.0,
                      seed: int = 0, strength: float = 6.0) -> PredSet:
    """Frames whose labels are drawn from softmax(logits / t_true)."""
    rng = np.random.default_rng(seed)
    n = n_clips * frames_per_clip
    y_clip = np.arange(n_clips) % NUM_STATIONS
    y = np.repeat(y_clip, frames_per_clip)
    logits = rng.normal(0, 2.0, (n, NUM_STATIONS))
    logits[np.arange(n), y] += strength
    frames = pd.DataFrame({
        "clip_uid": np.repeat([f"f{fold}c{i}" for i in range(n_clips)], frames_per_clip),
        "src_idx": np.tile(np.arange(frames_per_clip) * 3, n_clips),
        "u": np.tile(np.linspace(0, 1, frames_per_clip), n_clips),
        "sharpness": rng.uniform(200, 2000, n), "dark_frac": np.zeros(n), "sat_frac": np.zeros(n),
        "redout": np.zeros(n), "layout_ok": True, "primary": True,
        "patient_id": np.repeat([f"P{fold}{i % 6}" for i in range(n_clips)], frames_per_clip),
    })
    frames["y"] = _sample_labels(logits, t_true, rng)
    return PredSet(fold, frames, logits)


# -- temperature ---------------------------------------------------------------------------------------

def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3.0, (30000, NUM_STATIONS))
    for t_true in (0.7, 1.0, 2.5):
        y = _sample_labels(logits, t_true, rng)
        assert fit_temperature(logits, y) == pytest.approx(t_true, rel=0.05)


def test_soft_targets_equal_hard_targets_when_one_hot():
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 2.0, (2000, NUM_STATIONS))
    y = rng.integers(0, NUM_STATIONS, 2000)
    assert nll(logits, np.eye(NUM_STATIONS)[y], 1.7) == pytest.approx(nll(logits, y, 1.7))


def test_temperature_scaling_improves_ece_on_overconfident_logits():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 3.0, (20000, NUM_STATIONS)) * 3  # 3x overconfident
    y = _sample_labels(logits, 3.0, rng)
    t = fit_temperature(logits, y)
    assert ece(softmax(logits, t), y) < ece(softmax(logits), y) / 2


# -- tau -------------------------------------------------------------------------------------------------

def test_tau_meets_the_closed_set_precision_target():
    ps = synthetic_predset(0, n_clips=200, seed=4)
    probs = softmax(ps.logits, 2.0)
    y = ps.frames["y"].to_numpy()
    for target in (0.85, 0.9, 0.95):
        tau, reached, accepted = fit_tau(probs, y, target)
        top, conf = probs.argmax(1), probs.max(1)
        for c in range(NUM_STATIONS):
            assert tau[c] >= 0.5
            if reached[c]:
                acc = (top == c) & (conf >= tau[c])
                assert acc.sum() == accepted[c] >= 20
                assert (y[acc] == c).mean() >= target
                # smallest such threshold: every lower candidate threshold misses the target
                for t in np.unique(conf[(top == c) & (conf < tau[c]) & (conf >= 0.5)]):
                    sel = (top == c) & (conf >= t)
                    assert (y[sel] == c).mean() < target


def test_unreachable_station_gets_tau_one():
    rng = np.random.default_rng(5)
    probs = softmax(rng.normal(0, 1, (500, NUM_STATIONS)))
    probs[:, 3] += 2.0  # always predicted 3 ...
    probs /= probs.sum(1, keepdims=True)
    y = np.zeros(500, int)  # ... and never right
    tau, reached, _ = fit_tau(probs, y, 0.9)
    assert tau[3] == 1.0 and not reached[3]


def test_tau_handles_tied_confidences():
    probs = np.full((100, NUM_STATIONS), 0.02)
    probs[:, 0] = 0.82
    y = np.r_[np.zeros(95, int), np.ones(5, int)]
    tau, reached, accepted = fit_tau(probs, y, 0.9)
    assert reached[0] and accepted[0] == 100 and tau[0] == pytest.approx(0.82)


# -- status and quality ---------------------------------------------------------------------------------

def test_frame_status_matches_the_serving_rule():
    probs = np.array([[0.9] + [0.1 / 9] * 9, [0.6] + [0.4 / 9] * 9, [0.95] + [0.05 / 9] * 9])
    tau = np.full(NUM_STATIONS, 0.7)
    st = frame_status(probs, np.array([0.5, 0.5, 0.05]), tau, q_min=0.1, layout_ok=np.array([True, True, True]))
    assert list(st) == ["ok", "uncertain", "low_quality"]
    st = frame_status(probs, np.array([0.5, 0.5, 0.5]), tau, 0.1, np.array([False, True, True]))
    assert st[0] == "low_quality"  # an unverified layout is never evidence
    assert list(frame_status(probs, np.zeros(3), tau, 0.1, gate=False)) == ["ok", "uncertain", "ok"]


def test_recomputed_quality_equals_quality_score():
    rng = np.random.default_rng(6)
    f = pd.DataFrame({"sharpness": rng.uniform(0, 3000, 200), "dark_frac": rng.uniform(0, 0.6, 200),
                      "sat_frac": rng.uniform(0, 0.3, 200), "redout": rng.uniform(0, 0.6, 200)})
    got = D.recompute_quality(f, PARAMS)
    want = [quality_score(QualityMetrics(*r), PARAMS) for r in f[["sharpness", "dark_frac", "sat_frac", "redout"]]
            .itertuples(index=False)]
    np.testing.assert_allclose(got, want)


def test_q_min_is_a_percentile_of_clip_centre_quality():
    ps = synthetic_predset(0, n_clips=100, seed=7)
    f = ps.frames
    centre = f.loc[(f["u"] - 0.5).abs().groupby(f["clip_uid"]).idxmin()]
    want = np.percentile(np.minimum(1, centre["sharpness"] / PARAMS.sharp_ref), 10)
    assert fit_q_min(f, PARAMS, 10) == pytest.approx(want)


# -- honest cross-fitting ---------------------------------------------------------------------------------

def test_crossfit_thresholds_never_use_the_scored_fold():
    oof = {k: synthetic_predset(k, seed=10 + k) for k in range(3)}
    inner = {k: synthetic_predset(k, seed=20 + k) for k in range(3)}
    cal = crossfit(oof, inner, PARAMS, 0.9, 10)
    # wreck fold 0's OOF: its own thresholds must not move
    bad = {**oof, 0: PredSet(0, oof[0].frames, -oof[0].logits)}
    cal_bad = crossfit(bad, inner, PARAMS, 0.9, 10)
    assert cal_bad[0].tau == cal[0].tau and cal_bad[0].T == cal[0].T
    assert cal_bad[1].tau != cal[1].tau  # ... while the folds that pool it do change
    # tau_0 is exactly the fit on folds {1, 2}
    temps = {k: c.T for k, c in cal.items()}
    tau, q_min, _, _ = fit_thresholds(oof, [1, 2], temps, PARAMS, 0.9, 10)
    np.testing.assert_allclose(cal[0].tau, tau)
    assert cal[0].q_min == pytest.approx(q_min)


def test_temperature_per_fold_comes_from_inner_val():
    oof = {k: synthetic_predset(k, seed=30 + k, t_true=1.0) for k in range(2)}
    inner = {k: synthetic_predset(k, n_clips=150, seed=40 + k, t_true=2.0) for k in range(2)}
    cal = crossfit(oof, inner, PARAMS)
    assert all(c.T == pytest.approx(2.0, rel=0.15) for c in cal.values())


def test_shipped_values_use_all_folds():
    oof = {k: synthetic_predset(k, seed=50 + k, t_true=1.5) for k in range(3)}
    s = shipped(oof, PARAMS)
    assert s.T == pytest.approx(1.5, rel=0.15)
    assert len(s.tau) == NUM_STATIONS and all(0.5 <= t <= 1.0 for t in s.tau)


# -- leave-station-out rejection scores (lso.py) ---------------------------------------------------------

def test_rejection_scores_and_auroc():
    from training.landmarks.lso import auroc, energy, fpr_at_tpr, knn_score, msp

    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.ones(5), np.ones(5)) == 0.5
    assert fpr_at_tpr(np.linspace(1, 2, 100), np.linspace(0, 0.9, 50)) == 0.0
    rng = np.random.default_rng(8)
    z = rng.normal(0, 1, (50, NUM_STATIONS))
    z[:25, 0] += 8  # confident (in-distribution-like) rows
    assert msp(z, 1.0)[:25].min() > msp(z, 1.0)[25:].max()
    assert energy(z, 1.0)[:25].mean() > energy(z, 1.0)[25:].mean()
    bank = rng.normal(0, 1, (200, 16))
    near = bank[:10] + 0.01 * rng.normal(0, 1, (10, 16))
    far = rng.normal(0, 1, (10, 16))
    assert knn_score(bank, near, k=1).min() > knn_score(bank, far, k=1).max()


# -- held-out temperatures (review fix: T_{k-1} was fit on fold k) --------------------------------------------

def _with_fold(ps: PredSet, fold: int) -> PredSet:
    ps.frames["fold_i"] = fold
    return ps


def test_heldout_temperatures_never_use_the_reported_folds_labels():
    from training.landmarks.calibrate import heldout_temperatures, inner_fold_of, temperature_frames

    n = 3
    oof = {k: _with_fold(synthetic_predset(k, seed=60 + k, t_true=1.0), k) for k in range(n)}
    # model j's inner-val fold is (j + 1) % n, as in dataset.fold_patients
    inner = {j: _with_fold(synthetic_predset((j + 1) % n, n_clips=150, seed=70 + j, t_true=2.5), (j + 1) % n)
             for j in range(n)}
    assert [inner_fold_of(inner[j]) for j in range(n)] == [1, 2, 0]
    held = heldout_temperatures(oof, inner, PARAMS, 0.0)
    for k in range(n):
        j = (k - 1) % n  # the model whose inner-val IS fold k
        assert held[k][j] == pytest.approx(fit_temperature(*temperature_frames(oof[j], PARAMS, 0.0)))
        assert held[k][j] == pytest.approx(1.0, rel=0.2)  # refit on its own outer fold (t_true 1.0) ...
        assert all(held[k][i] == pytest.approx(2.5, rel=0.2) for i in range(n) if i != j)  # ... others unchanged
    # wrecking fold k's labels in the inner-val predictions cannot move fold k's thresholds
    base = crossfit(oof, inner, PARAMS, 0.9, 10)
    bad = dict(inner)
    bad[2] = PredSet(2, inner[2].frames, -inner[2].logits)  # model 2's inner-val = fold 0
    assert crossfit(oof, bad, PARAMS, 0.9, 10)[0].tau == base[0].tau
    assert crossfit(oof, bad, PARAMS, 0.9, 10)[1].tau != base[1].tau  # fold 1 legitimately pools model 2


# -- G-STATION, outcome and the CI-width simulation (evaluate.py) -----------------------------------------------

def _gates():
    from training.landmarks.run_utils import GATES_PATH, read_json

    return read_json(GATES_PATH)


def _per_class(prec=1.0, lo=0.95, rec=0.9):
    from app.ml.landmarks.stations import STATION_ORDER

    return {k: {"precision": prec, "precision_lo": lo, "recall": rec} for k in STATION_ORDER}


def _cp(station: str, mode: str, per_patient: dict[str, list[bool]]) -> pd.DataFrame:
    from app.ml.landmarks.stations import STATION_INDEX

    y = STATION_INDEX[station]
    rows = [{"y": y, "pred": y if ok else (y + 1) % NUM_STATIONS, "patient_id": p, "mode_major": mode}
            for p, oks in per_patient.items() for ok in oks]
    return pd.DataFrame(rows)


def test_wl_oesophagus_gate_is_patient_level_and_recorded_per_mode():
    from training.landmarks.evaluate import mode_breakdown, station_gate

    g = _gates()["G-STATION"]
    nbi = _cp("esophagus_proximal", "nbi", {f"N{i}": [True] for i in range(20)})
    # 7 WL patients, one of them with 5 WL clips all wrong: clip-level 6/11, patient-level 6/7
    wl = _cp("esophagus_proximal", "wl", {**{f"W{i}": [True] for i in range(6)}, "W6": [False] * 5})
    bm = mode_breakdown(pd.concat([nbi, wl], ignore_index=True))
    pc = bm["wl"]["per_class"]["esophagus_proximal"]
    assert (pc["n"], pc["patients"], pc["patients_correct"]) == (11, 7, 6)
    assert pc["patient_lo"] == pytest.approx(0.487, abs=0.002)  # 6/7 -> below 0.50: needs 7/7
    auto, per = station_gate(_per_class(), bm, {}, g)
    assert not auto[0] and per["esophagus_proximal"]["by_mode"] == {"wl": False, "nbi": True}
    wl7 = _cp("esophagus_proximal", "wl", {f"W{i}": [True] for i in range(7)})
    auto, per = station_gate(_per_class(), mode_breakdown(pd.concat([nbi, wl7], ignore_index=True)), {}, g)
    assert auto[0] and per["esophagus_proximal"]["by_mode"] == {"wl": True, "nbi": True}


def test_station_gate_uses_the_s4_fdr_of_the_enabled_station():
    from training.landmarks.evaluate import mode_breakdown, station_gate

    g = {**_gates()["G-STATION"], "per_mode_stations": []}
    bm = mode_breakdown(_cp("antrum", "wl", {"A": [True]}))
    auto, per = station_gate(_per_class(), bm, {"antrum": {"fdr": 0.2}, "z_line": {"fdr": 0.05}}, g)
    assert auto == [k != "antrum" for k in per]
    assert per["antrum"]["s4_fdr"] == 0.2


def test_outcome_requires_g_stream_for_go_and_partial():
    from app.ml.landmarks.stations import NUM_STATIONS as N
    from training.landmarks.evaluate import outcome

    gates = _gates()
    assert gates["G-STREAM"]["blocking"] is True
    blocking = [k for k, v in gates.items() if isinstance(v, dict) and v.get("blocking")]
    ok = {k: "PASS" for k in blocking}
    all_on, some = [True] * N, [True] * (N - 2) + [False] * 2
    assert outcome(ok, all_on, gates) == "GO"
    assert outcome(ok, some, gates) == "PARTIAL"
    assert outcome({**ok, "G-STREAM": "FAIL"}, all_on, gates) == "NO-GO"  # the finding: was GO
    assert outcome({**ok, "G-STREAM": "FAIL"}, some, gates) == "NO-GO"
    assert outcome({**ok, "G-STREAM": "PASS (placeholder transit: optimistic)"}, all_on, gates).startswith("PENDING")
    assert outcome({k: v for k, v in ok.items() if k != "G-SERVE"}, all_on, gates) == "PENDING (G-SERVE)"
    assert outcome(ok, [False] * N, gates) == "NO-GO"


def test_ci_simulation_resamples_patients_with_replacement(tmp_path):
    from training.landmarks.evaluate import ci_simulation
    from training.landmarks.run_utils import write_json

    rng = np.random.default_rng(0)
    rows = []
    for p in range(6):
        for c in range(30):
            y = c % NUM_STATIONS
            rows.append({"clip_uid": f"p{p}c{c}", "y": y, "pred": y if rng.random() < 0.8 else (y + 1) % 10})
    cp = pd.DataFrame(rows)
    cp.to_csv(tmp_path / "clip_predictions_oof.csv.gz", index=False)
    write_json(tmp_path / "eval_oof.json", {"clip": {"macro_f1": {"value": 0.8}}})
    m = pd.DataFrame({"clip_uid": cp["clip_uid"], "patient_id": [u.split("c")[0] for u in cp["clip_uid"]]})
    out = ci_simulation(tmp_path, n_test_patients=19, n_sim=200, manifest=m)
    # 19 patients from 6 WITHOUT replacement = the same 6 every time (zero spread); with replacement: a real spread
    assert out["clip_macro_f1"]["half_width"] > 0.005
    assert "WITH replacement" in out["resampling"]
