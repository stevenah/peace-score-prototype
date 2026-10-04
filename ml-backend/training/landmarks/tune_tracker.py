"""Nested tuning of the tracker and its thresholds on the replay benchmark (M3).

    python -m training.landmarks.tune_tracker --exp runs/<exp> [--seeds 2] [--transit placeholder|model]
        [--disable station,...]

Grid (243 configurations):
    tau precision target {0.85, 0.90, 0.95} x q_min percentile {P5, P10, P25}
    x evidence N {2, 3, 4} x window W {4, 6, 8} x current_release {3, 4, 6}

Nesting (critic C4): for outer fold k, configurations are SELECTED on the
S2/S4 streams of the other four folds, where fold j's thresholds (tau, q_min)
are fit on the folds other than j and k; the selected configuration is then
SCORED on fold k with thresholds fit on the folds other than k. Every
temperature used for fold k comes from ``calibrate.heldout_temperatures``
(never fit on fold k's labels). ``auto_enabled_k`` is recomputed for each
held-out fold from the G-STATION criteria (evaluate.station_gate) on the
folds other than k: clip precision/recall/per-mode from their OOF, S4 FDR of
each station (enabled) replayed with the selected configuration. Fold k is
scored with ``auto_enabled_k`` applied, so the held-out numbers (G-STREAM,
the G-TEST reference) describe the system that is shipped, not one with all
10 stations on. Reported numbers are the held-out-fold scores. The shipped
configuration is the modal selection, with thresholds refit on all dev OOF
(``shipped_calibration``); the shipped auto_enabled comes from
``evaluate --report`` (gates_result.json).

Objective: maximise mean station recall (S2), subject to S4 FDR <= 0.05 and
median time-to-observe <= 3 s; if nothing is feasible, the smallest
violation wins. Selection counts every station except ``--disable``d ones
(auto_enabled_k depends on the selected configuration, so it cannot also
drive the selection). ``current_release`` does not change recall or FDR and
is picked afterwards (minimum flicker, real StationTracker).

``--disable`` is the pre-registered dev-only retune path when G-STREAM fails:
those stations are forced manual-only in every fold and in the shipped set
(recorded as ``force_disabled``).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from app.ml.landmarks.tracker import TrackerParams
from training.landmarks import dataset as D
from training.landmarks import evaluate as E
from training.landmarks.calibrate import (
    PredSet,
    fit_thresholds,
    frame_status,
    load_experiment,
    shipped,
)
from training.landmarks.run_utils import GATES_PATH, read_json, write_json
from training.landmarks.stream_eval import (
    ALL_TRUE,
    aggregate,
    fast_observed,
    fold_hooks,
    load_calibration,
    replay,
    station_records,
    stream_record,
)
from training.landmarks.stream_sim import Stream, StreamConfig, build_streams, fill_transit

PRECISIONS = (0.85, 0.90, 0.95)
Q_PCTS = (5, 10, 25)
EVIDENCE = (2, 3, 4)
WINDOWS = (4, 6, 8)
RELEASES = (3, 4, 6)
FDR_MAX = 0.05
TTO_MAX = 3.0
GATE_N_BOOT = 500  # patient bootstrap of the nested G-STATION precision CI


@dataclass
class FoldStreams:
    s2: list[Stream]
    s4: list[Stream]

    def with_temperature(self, T: float) -> "FoldStreams":
        return FoldStreams([s.with_temperature(T) for s in self.s2], [s.with_temperature(T) for s in self.s4])


def make_streams(oof: dict[int, PredSet], temps: dict[int, float], params: QualityParams, seeds: Sequence[int],
                 hooks: dict | None = None, hz: float = 2.0) -> dict[int, FoldStreams]:
    out = {}
    for k, ps in oof.items():
        s2, s4 = [], []
        for seed in seeds:
            s2 += build_streams(ps, StreamConfig("S2", hz=hz, seed=seed), temps[k], params)
            s4 += build_streams(ps, StreamConfig("S4", hz=hz, seed=seed), temps[k], params)
        fill_transit(s2 + s4, temps[k], params, hooks.get(k) if hooks else None)
        out[k] = FoldStreams(s2, s4)
    return out


def _qualifying(s: Stream, tau, q_min: float) -> np.ndarray:
    st = frame_status(s.probs, s.quality, tau, q_min, s.layout_ok)
    return np.where(st == "ok", s.probs.argmax(1), -1)


def score(fs: Sequence[FoldStreams], thresholds: Sequence[tuple[np.ndarray, float]], n: int, w: int,
          auto_enabled: Sequence[bool] = ALL_TRUE) -> dict:
    """Mean recall (S2), S4 FDR and median TTO for one (N, W) over folds with their own thresholds,
    counting auto-enabled stations only."""
    obs_n = np.zeros(NUM_STATIONS)
    pres_n = np.zeros(NUM_STATIONS)
    ttos, fdr = [], []
    for f, (tau, q_min) in zip(fs, thresholds):
        for s in f.s2:
            at = fast_observed(_qualifying(s, tau, q_min), n, w, auto_enabled)
            for k, key in enumerate(STATION_ORDER):
                if key in s.present and auto_enabled[k]:
                    pres_n[k] += 1
                    if at[k] >= 0:
                        obs_n[k] += 1
                        seg = next((c for c in s.clips if not c.ambiguous and key in c.stations), None)
                        if seg is not None:
                            ttos.append(s.t[at[k]] - seg.t0)
        for s in f.s4:
            k = STATION_INDEX[s.cfg.remove_station]
            if auto_enabled[k]:
                fdr.append(fast_observed(_qualifying(s, tau, q_min), n, w, auto_enabled)[k] >= 0)
    with np.errstate(invalid="ignore"):
        rec = obs_n / pres_n
    return {"recall": float(np.nanmean(rec)) if (pres_n > 0).any() else 0.0,
            "fdr": float(np.mean(fdr)) if fdr else 0.0,
            "tto": float(np.median(ttos)) if ttos else float("inf")}


def station_fdr(fs: Sequence[FoldStreams], thresholds: Sequence[tuple[np.ndarray, float]], n: int,
                w: int) -> dict[str, dict]:
    """{station: {"fdr", "n"}}: S4 false-observation rate of each station WITH the station enabled."""
    hits, count = np.zeros(NUM_STATIONS), np.zeros(NUM_STATIONS)
    for f, (tau, q_min) in zip(fs, thresholds):
        for s in f.s4:
            k = STATION_INDEX[s.cfg.remove_station]
            count[k] += 1
            hits[k] += fast_observed(_qualifying(s, tau, q_min), n, w)[k] >= 0
    return {key: {"fdr": float(hits[k] / count[k]), "n": int(count[k])}
            for k, key in enumerate(STATION_ORDER) if count[k]}


def fold_auto_enabled(oof: dict[int, PredSet], use_folds: Sequence[int], temps: dict[int, float],
                      params: QualityParams, thresholds: Sequence[tuple[np.ndarray, float]],
                      s4_fdr: dict[str, dict], gates: dict, n_boot: int = GATE_N_BOOT) -> tuple[list[bool], dict]:
    """auto_enabled from the G-STATION criteria on ``use_folds`` only (never the reported fold)."""
    cp = pd.concat([E.clip_predictions(oof[j], temps[j], params, q_min)
                    for j, (_, q_min) in zip(use_folds, thresholds)], ignore_index=True)
    per_class = E.per_class_metrics(cp, n_boot=n_boot)
    return E.station_gate(per_class, E.mode_breakdown(cp), s4_fdr, gates["G-STATION"])


def _key(res: dict) -> tuple:
    viol = max(0.0, res["fdr"] - FDR_MAX) + max(0.0, res["tto"] - TTO_MAX) / 10.0
    return (viol > 0, viol, -res["recall"], res["fdr"])


def select(fs: Sequence[FoldStreams], thr_grid: dict[tuple, list[tuple[np.ndarray, float]]],
           auto_enabled: Sequence[bool] = ALL_TRUE) -> tuple[dict, dict]:
    """Best (precision, q_pct, N, W) on the given folds; thr_grid[(prec, q)] = per-fold thresholds."""
    best, best_res = None, None
    for (prec, qp), thr in thr_grid.items():
        for n, w in itertools.product(EVIDENCE, WINDOWS):
            if n > w:
                continue
            r = score(fs, thr, n, w, auto_enabled)
            if best_res is None or _key(r) < _key(best_res):
                best, best_res = {"precision": prec, "q_pct": qp, "evidence_needed": n, "evidence_window": w}, r
    return best, best_res


def pick_release(fs: Sequence[FoldStreams], thresholds, cfg: dict, auto_enabled: Sequence[bool] = ALL_TRUE) -> int:
    best, best_flicker = RELEASES[0], float("inf")
    for rel in RELEASES:
        tp = TrackerParams(evidence_needed=cfg["evidence_needed"], evidence_window=cfg["evidence_window"],
                           current_release=rel)
        rows = []
        for f, (tau, q_min) in zip(fs, thresholds):
            for s in f.s2:
                rows.append(stream_record(s, replay(s, tp, tau, q_min, auto_enabled)))
        sr = pd.DataFrame(rows)
        flicker = float(((sr["switches"] - sr["true_switches"]) / sr["minutes"]).mean())
        if flicker < best_flicker - 1e-9:
            best, best_flicker = rel, flicker
    return best


def tune(oof: dict[int, PredSet], temps: dict[int, float], params: QualityParams, seeds: Sequence[int] = (0, 1),
         hooks: dict | None = None, log=print, temps_heldout: dict[int, dict[int, float]] | None = None,
         gates: dict | None = None, force_disabled: Sequence[str] = (), n_boot: int = GATE_N_BOOT,
         tables: dict | None = None) -> dict:
    """tracker.json content; ``tables`` (if given) receives the held-out station records per scenario."""
    gates = gates or read_json(GATES_PATH)
    folds = sorted(oof)
    temps_heldout = temps_heldout or {k: dict(temps) for k in folds}
    forced = {STATION_ORDER[STATION_INDEX[k]] for k in force_disabled}
    selectable = [key not in forced for key in STATION_ORDER]
    streams = make_streams(oof, temps, params, seeds, hooks)
    per_fold, heldout = {}, {}
    st_rows, sr_rows = [], []
    for k in folds:
        inner = [j for j in folds if j != k]
        tk = temps_heldout[k]
        fs_inner = [streams[j] if tk[j] == temps[j] else streams[j].with_temperature(tk[j]) for j in inner]
        thr_grid = {}
        for prec, qp in itertools.product(PRECISIONS, Q_PCTS):
            thr = []
            for j in inner:
                tau, q_min, _, _ = fit_thresholds(oof, [i for i in folds if i not in (j, k)], tk, params, prec, qp)
                thr.append((tau, q_min))
            thr_grid[(prec, qp)] = thr
        cfg, sel_res = select(fs_inner, thr_grid, selectable)
        thr_sel = thr_grid[(cfg["precision"], cfg["q_pct"])]
        cfg["current_release"] = pick_release(fs_inner, thr_sel, cfg, selectable)
        n, w = cfg["evidence_needed"], cfg["evidence_window"]
        auto_k, gate_k = fold_auto_enabled(oof, inner, tk, params, thr_sel, station_fdr(fs_inner, thr_sel, n, w),
                                           gates, n_boot)
        auto_k = [bool(a and ok) for a, ok in zip(auto_k, selectable)]
        tau_k, q_k, _, _ = fit_thresholds(oof, inner, tk, params, cfg["precision"], cfg["q_pct"])
        held = score([streams[k]], [(tau_k, q_k)], n, w, auto_k)
        tp = TrackerParams(evidence_needed=n, evidence_window=w, current_release=cfg["current_release"])
        for s in streams[k].s2 + streams[k].s4:
            rep = replay(s, tp, tau_k, q_k, auto_k)
            st_rows += station_records(s, rep.observed_at, auto_enabled=auto_k,
                                       observed_if_enabled=fast_observed(rep.q, n, w))
            if s.cfg.scenario == "S2":
                sr_rows.append(stream_record(s, rep))
        per_fold[k] = {**cfg, "selection": sel_res, "tau": tau_k.tolist(), "q_min": q_k, "auto_enabled": auto_k,
                       "station_gate": gate_k, "temperatures": {str(j): t for j, t in tk.items()}}
        heldout[k] = held
        log(f"fold {k}: {cfg} selection {sel_res} -> held-out {held}; "
            f"manual-only {[key for key, a in zip(STATION_ORDER, auto_k) if not a]}")

    st = pd.DataFrame(st_rows)
    sr = pd.DataFrame(sr_rows)
    s2_st, s4_st = st[st["scenario"] == "S2"], st[st["scenario"] == "S4"]
    pooled = {"S2": aggregate(s2_st, sr), "S4": aggregate(s4_st, sr.iloc[:0])}
    if tables is not None:
        tables.update({"S2": s2_st.reset_index(drop=True), "S4": s4_st.reset_index(drop=True)})
    modal = Counter(tuple(sorted((a, b) for a, b in c.items() if a in
                                 ("precision", "q_pct", "evidence_needed", "evidence_window", "current_release")))
                    for c in per_fold.values()).most_common(1)[0][0]
    ship_cfg = dict(modal)
    ship_cal = shipped(oof, params, ship_cfg["precision"], ship_cfg["q_pct"])
    tracker = TrackerParams(evidence_needed=ship_cfg["evidence_needed"], evidence_window=ship_cfg["evidence_window"],
                            current_release=ship_cfg["current_release"])
    return {
        "grid": {"precision": PRECISIONS, "q_pct": Q_PCTS, "evidence_needed": EVIDENCE,
                 "evidence_window": WINDOWS, "current_release": RELEASES},
        "objective": {"fdr_max": FDR_MAX, "tto_max_s": TTO_MAX},
        "seeds": list(seeds),
        "transit_source": sorted({s.transit_source for f in streams.values() for s in f.s2}),
        "force_disabled": sorted(forced, key=STATION_INDEX.get),
        "per_fold": {str(k): {**TrackerParams(evidence_needed=c["evidence_needed"],
                                              evidence_window=c["evidence_window"],
                                              current_release=c["current_release"]).to_dict(),
                              "precision": c["precision"], "q_pct": c["q_pct"], "selection": c["selection"],
                              "auto_enabled": c["auto_enabled"], "station_gate": c["station_gate"],
                              "temperatures": c["temperatures"], "heldout": heldout[k]}
                     for k, c in per_fold.items()},
        "heldout_pooled": pooled,
        "shipped": {"tracker": tracker.to_dict(), "precision": ship_cfg["precision"], "q_pct": ship_cfg["q_pct"]},
        "shipped_calibration": ship_cal.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--transit", choices=("placeholder", "model"), default="placeholder")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--disable", default="", help="comma-separated stations forced manual-only (G-STREAM retune)")
    a = ap.parse_args(argv)
    m = D.load_manifest()
    oof = load_experiment(a.exp, m, "oof")
    cal, params, raw = load_calibration(a.exp)
    temps = {k: c.T for k, c in cal.items()}
    held = {int(k): {int(j): t for j, t in v.items()} for k, v in (raw.get("temps_heldout") or {}).items()} or None
    if held is None:
        print("WARNING: calibration.json has no temps_heldout (old calibrate run); T_{k-1} was fit on fold k")
    hooks = None
    if a.transit == "model":
        from training.landmarks.run_utils import resolve_device

        hooks = fold_hooks(a.exp, resolve_device(a.device))
    tables: dict = {}
    out = tune(oof, temps, params, range(a.seeds), hooks, temps_heldout=held,
               force_disabled=[x for x in a.disable.split(",") if x], tables=tables)
    write_json(a.exp / "tracker.json", out)
    for scen, t in tables.items():  # CI-width simulation input (evaluate --ci-sim)
        t.to_csv(a.exp / f"tracker_heldout_{scen}_stations.csv.gz", index=False)
    print("shipped", out["shipped"], "force_disabled", out["force_disabled"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
