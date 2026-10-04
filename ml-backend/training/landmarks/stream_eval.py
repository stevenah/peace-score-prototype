"""Procedure Replay Benchmark metrics through the serving ``StationTracker`` (unchanged).

    python -m training.landmarks.stream_eval --exp runs/<exp> [--scenarios S0,S2,S4,S5,S2@1hz]
        [--seeds 5] [--tracker runs/<exp>/tracker.json] [--transit placeholder|model] [--device cuda]

Per stream (stream_sim.py) every frame gets the status serving would give it
(calibrate.frame_status with the fold's CROSS-FIT T, tau and q_min), goes
through ``StationTracker.update`` and the resulting states/events are scored:

    recall_k        station k observed / patients where k is present
    fdr_k (S4)      k observed although all of k's clips were removed
    false_absent    observed stations the patient never had (S2)
    tto             time from the start of k's first clip to "observed" (s),
                    and the share observed before that clip ended
    on_transit      share of observations fired on a synthetic transit frame
                    (reported on its own; transit is made from past clips)
    transit         share of transit frames with current=None / status!=ok
    flicker         current-station switches per minute minus true switches
    best purity     best_frame events whose frame belongs to a clip of k
    completeness    |missing| + |false| per patient; % of all-10 patients at 10/10

Stations that are not auto-enabled (G-STATION failures, per outer fold in
tune_tracker) can never be observed live, so every stream metric is computed
over the auto-enabled stations only; station records carry ``auto`` and the
all-enabled ``observed_if_enabled`` (the G-STATION S4 FDR input).

CIs are patient-cluster bootstraps. S2 results are labelled with their
transit source: ``placeholder_uniform`` (OOF-only; optimistic, transit
frames never qualify) or ``model`` (the fold model classified them).
``fast_observed`` is a vectorised equivalent of the tracker's observe rule
for tuning; tests assert it matches ``StationTracker`` exactly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from app.ml.landmarks.tracker import FrameEvidence, StationTracker, TrackerParams
from training.landmarks import dataset as D
from training.landmarks.calibrate import Calib, frame_status, load_experiment
from training.landmarks.run_utils import cluster_bootstrap, read_json, write_json
from training.landmarks.stream_sim import Stream, StreamConfig, build_streams, fill_transit, model_hook

ALL_TRUE = (True,) * NUM_STATIONS


@dataclass
class Replay:
    status: np.ndarray  # (n,) ok | uncertain | low_quality
    q: np.ndarray  # (n,) qualifying station index or -1
    current: np.ndarray  # (n,) current station index or -1
    observed_at: np.ndarray  # (10,) frame index of the observed event, -1 if never
    best_events: list[tuple[int, int]]  # (frame index, station)


def evidence(stream: Stream, tau: Sequence[float], q_min: float, gate: bool = True):
    """(status, top, confidence) arrays exactly as serving computes them."""
    status = frame_status(stream.probs, stream.quality, np.asarray(tau), q_min, stream.layout_ok, gate)
    top = stream.probs.argmax(1)
    conf = stream.probs[np.arange(len(top)), top]
    return status, top, conf


def replay(stream: Stream, params: TrackerParams, tau: Sequence[float], q_min: float,
           auto_enabled: Sequence[bool] = ALL_TRUE, gate: bool = True) -> Replay:
    status, top, conf = evidence(stream, tau, q_min, gate)
    tr = StationTracker(params, auto_enabled)
    n = len(stream)
    current = np.full(n, -1)
    observed_at = np.full(NUM_STATIONS, -1)
    best = []
    for i in range(n):
        step = tr.update(FrameEvidence(str(status[i]), int(top[i]), float(conf[i]), float(stream.quality[i])))
        current[i] = STATION_INDEX[step.current] if step.current else -1
        for k in step.observed_events:
            observed_at[STATION_INDEX[k]] = i
        for k in step.best_frame_events:
            best.append((i, STATION_INDEX[k]))
    q = np.where(status == "ok", top, -1)
    return Replay(status, q, current, observed_at, best)


def fast_observed(q: np.ndarray, n_needed: int, window: int, auto_enabled: Sequence[bool] = ALL_TRUE) -> np.ndarray:
    """First frame index at which each station becomes observed (-1 never).

    Same rule as StationTracker: station k is observed at the first evaluated
    frame whose top qualifying station is k and where k qualified in >= N of
    the last W evaluated frames (the current one included).
    """
    out = np.full(NUM_STATIONS, -1)
    for k in range(NUM_STATIONS):
        if not auto_enabled[k]:
            continue
        a = (q == k).astype(np.int32)
        if a.sum() < n_needed:
            continue
        c = np.cumsum(a)
        win = c - np.concatenate([np.zeros(window, np.int32), c[:-window]])[:len(c)]
        hit = np.flatnonzero((win >= n_needed) & (a == 1))
        if len(hit):
            out[k] = hit[0]
    return out


# -- per-stream records ---------------------------------------------------------------------------------

def _first_clip(stream: Stream, k: str):
    for c in stream.clips:
        if not c.ambiguous and k in c.stations:
            return c
    return None


def station_records(stream: Stream, observed_at: np.ndarray, extra: dict | None = None,
                    auto_enabled: Sequence[bool] = ALL_TRUE,
                    observed_if_enabled: np.ndarray | None = None) -> list[dict]:
    """One row per station: present / allowed / observed / time-to-observe.

    ``observed_at`` comes from a replay with ``auto_enabled`` (a disabled
    station is never observed); ``observed_if_enabled`` optionally gives the
    all-enabled observe index (``fast_observed`` with ALL_TRUE), which is what
    the station WOULD do if it were enabled (observation of one station does
    not depend on the others' flags).
    """
    rows = []
    for k, key in enumerate(STATION_ORDER):
        auto = bool(auto_enabled[k])
        obs = bool(observed_at[k] >= 0 and auto)
        t_obs = float(stream.t[observed_at[k]]) if obs else np.nan
        seg = _first_clip(stream, key)
        rows.append({
            "patient_id": stream.patient_id, "fold": stream.fold, "seed": stream.cfg.seed,
            "scenario": stream.cfg.scenario, "hz": stream.cfg.hz, "removed": stream.cfg.remove_station,
            "station": key, "present": key in stream.present, "allowed": key in stream.allowed,
            "auto": auto, "observed": obs,
            "observed_if_enabled": bool((observed_if_enabled if observed_if_enabled is not None
                                         else observed_at)[k] >= 0),
            "on_transit": bool(obs and stream.kind[observed_at[k]] == "transit"),
            "tto": t_obs - seg.t0 if (obs and seg is not None) else np.nan,
            "within_clip": bool(obs and seg is not None and t_obs <= seg.t1),
            **(extra or {}),
        })
    return rows


def stream_record(stream: Stream, rep: Replay, extra: dict | None = None) -> dict:
    tr = stream.transit_idx
    observed = {STATION_ORDER[k] for k in np.flatnonzero(rep.observed_at >= 0)}
    switches, last = 0, -1
    for c in rep.current:
        if c >= 0 and c != last:
            switches += int(last >= 0)
            last = c
    true_sw = sum(1 for a, b in zip(stream.clips[:-1], stream.clips[1:])
                  if not a.ambiguous and not b.ambiguous and a.stations != b.stations)
    minutes = max((stream.t[-1] - stream.t[0]) / 60.0, 1e-6) if len(stream) else 1e-6
    pure = sum(1 for i, k in rep.best_events
               if stream.kind[i] == "clip" and STATION_ORDER[k] in stream.clips[stream.clip_pos[i]].stations)
    return {
        "patient_id": stream.patient_id, "fold": stream.fold, "seed": stream.cfg.seed,
        "scenario": stream.cfg.scenario, "hz": stream.cfg.hz, "removed": stream.cfg.remove_station,
        "transit_source": stream.transit_source, "n_frames": len(stream), "minutes": minutes,
        "transit_frames": int(len(tr)),
        "transit_current_none": int((rep.current[tr] < 0).sum()) if len(tr) else 0,
        "transit_not_ok": int((rep.status[tr] != "ok").sum()) if len(tr) else 0,
        "switches": switches, "true_switches": true_sw,
        "best_events": len(rep.best_events), "best_pure": pure,
        "n_present": len(stream.present),
        "missing": len(stream.present - observed), "false": len(observed - stream.allowed),
        "all10": len(stream.present) == NUM_STATIONS, "got10": len(stream.present) == NUM_STATIONS
        and stream.present <= observed,
        **(extra or {}),
    }


# -- aggregation ---------------------------------------------------------------------------------------------

def _ratio_ci(num: np.ndarray, den: np.ndarray, groups: np.ndarray, n_boot: int = 1000):
    def stat(i):
        d = den[i].sum()
        return num[i].sum() / d if d else np.nan

    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    return cluster_bootstrap(stat, groups, n_boot=n_boot)


def _auto_rows(st: pd.DataFrame, auto_enabled: Sequence[bool]) -> pd.Series:
    """Rows of auto-enabled stations: the ``auto_enabled`` argument AND the per-row ``auto`` flag."""
    auto = {k for k, a in zip(STATION_ORDER, auto_enabled) if a}
    ok = st["station"].isin(auto)
    return ok & st["auto"].astype(bool) if "auto" in st else ok


def aggregate(st: pd.DataFrame, sr: pd.DataFrame, auto_enabled: Sequence[bool] = ALL_TRUE,
              n_boot: int = 1000) -> dict:
    """Scenario-level metrics from station records (st) and stream records (sr).

    Stream metrics count auto-enabled stations only (``auto_enabled`` and the
    records' own ``auto`` flag, which tune_tracker sets per outer fold). S4
    also reports ``fdr_per_station_if_enabled`` over every station (the
    G-STATION input: what the station would do if it were enabled).
    """
    on = _auto_rows(st, auto_enabled) if len(st) else pd.Series(dtype=bool)
    out: dict = {"n_patients": int(st["patient_id"].nunique()), "n_streams": int(len(sr)),
                 "transit_source": sorted(set(sr["transit_source"]))}
    scen = st["scenario"].iloc[0] if len(st) else ""
    if scen == "S4":
        is_rem = st["station"] == st["removed"]
        per, per_all = {}, {}
        for k in STATION_ORDER:
            for table, col, sel in ((per, "observed", is_rem & on), (per_all, "observed_if_enabled", is_rem)):
                r = st[sel & (st["station"] == k)]
                if len(r) and col in r:
                    p, lo, hi = _ratio_ci(r[col].to_numpy(float), np.ones(len(r)), r["patient_id"].to_numpy(),
                                          n_boot)
                    table[k] = {"fdr": p, "lo": lo, "hi": hi, "n": int(len(r))}
        r = st[is_rem & on]
        p, lo, hi = _ratio_ci(r["observed"].to_numpy(float), np.ones(len(r)), r["patient_id"].to_numpy(), n_boot)
        out.update({"fdr_per_station": per, "fdr_per_station_if_enabled": per_all, "fdr_overall": p,
                    "fdr_overall_lo": lo, "fdr_overall_hi": hi,
                    "on_transit": float(r.loc[r["observed"], "on_transit"].mean())
                    if "on_transit" in r and r["observed"].any() else None})
        return out

    per = {}
    st_on = st[on]
    for k in STATION_ORDER:
        r = st_on[(st_on["station"] == k) & st_on["present"]]
        if len(r):
            p, lo, hi = _ratio_ci(r["observed"].to_numpy(float), np.ones(len(r)), r["patient_id"].to_numpy(), n_boot)
            tto = r.loc[r["observed"], "tto"].dropna()
            per[k] = {"recall": p, "lo": lo, "hi": hi, "n": int(len(r)),
                      "tto_median": float(tto.median()) if len(tto) else None,
                      "within_clip": float(r["within_clip"].mean())}
    out["recall_per_station"] = per
    out["never_enabled"] = sorted(set(st.loc[st["present"], "station"]) - set(per), key=STATION_INDEX.get)
    out["recall_mean"] = float(np.mean([v["recall"] for v in per.values()])) if per else None
    rp = st_on[st_on["present"]]
    if len(rp):
        # mean-over-stations recall with a patient bootstrap
        def mean_recall(i):
            sub = rp.iloc[i]
            g = sub.groupby("station")["observed"].mean()
            return float(g.mean())
        _, lo, hi = cluster_bootstrap(mean_recall, rp["patient_id"].to_numpy(), n_boot=n_boot)
        out["recall_mean_lo"], out["recall_mean_hi"] = lo, hi
    absent = st[~st["allowed"]]
    out["false_absent_rate"] = float(absent["observed"].mean()) if len(absent) else None
    tto = rp.loc[rp["observed"], "tto"].dropna()
    out["tto_median"] = float(tto.median()) if len(tto) else None
    out["tto_p90"] = float(tto.quantile(0.9)) if len(tto) else None
    out["within_clip"] = float(rp["within_clip"].mean()) if len(rp) else None
    obs = rp[rp["observed"]]
    out["observed_on_transit"] = float(obs["on_transit"].mean()) if len(obs) and "on_transit" in obs else None
    tf = sr["transit_frames"].sum()
    out["transit_abstention_current"] = float(sr["transit_current_none"].sum() / tf) if tf else None
    out["transit_abstention_frame"] = float(sr["transit_not_ok"].sum() / tf) if tf else None
    out["flicker_per_min"] = float(((sr["switches"] - sr["true_switches"]) / sr["minutes"]).mean())
    be = sr["best_events"].sum()
    out["best_frame_purity"] = float(sr["best_pure"].sum() / be) if be else None
    out["completeness_error_mean"] = float((sr["missing"] + sr["false"]).mean())
    a10 = sr[sr["all10"]]
    out["all10_patients"] = int(a10["patient_id"].nunique())
    out["all10_got10"] = float(a10["got10"].mean()) if len(a10) else None
    return out


# -- driver --------------------------------------------------------------------------------------------------

def parse_scenario(s: str) -> StreamConfig:
    """"S2" / "S2@1hz" / "S4" ... -> StreamConfig."""
    name, _, rate = s.partition("@")
    hz = float(rate.lower().replace("hz", "")) if rate else 2.0
    return StreamConfig(scenario=name, hz=hz)


def run_scenarios(oof: dict, cal: dict[int, Calib], params: QualityParams, scenarios: Sequence[str],
                  seeds: Sequence[int], tracker: TrackerParams | dict[int, TrackerParams],
                  auto_enabled: Sequence[bool] = ALL_TRUE, hooks: dict | None = None,
                  gates: Sequence[bool] = (True,), layout_by_clip: dict | None = None) -> dict:
    """{scenario[/gate-off]: aggregate} plus the raw record tables."""
    results, tables = {}, {}
    for scen in scenarios:
        for gate in gates:
            st_rows, sr_rows = [], []
            for seed in seeds:
                cfg = replace(parse_scenario(scen), seed=seed)
                for k, ps in oof.items():
                    c = cal[k]
                    streams = build_streams(ps, cfg, c.T, params)
                    fill_transit(streams, c.T, params, hooks.get(k) if hooks else None,
                                 layout_by_clip=layout_by_clip)
                    tp = tracker[k] if isinstance(tracker, dict) else tracker
                    for s in streams:
                        rep = replay(s, tp, c.tau, c.q_min, auto_enabled, gate)
                        st_rows += station_records(s, rep.observed_at, auto_enabled=auto_enabled,
                                                   observed_if_enabled=fast_observed(
                                                       rep.q, tp.evidence_needed, tp.evidence_window))
                        sr_rows.append(stream_record(s, rep))
            st, sr = pd.DataFrame(st_rows), pd.DataFrame(sr_rows)
            key = scen if gate else f"{scen}/gate_off"
            results[key] = aggregate(st, sr, auto_enabled)
            tables[key] = (st, sr)
    return {"results": results, "tables": tables}


def load_calibration(exp_dir: Path) -> tuple[dict[int, Calib], QualityParams, dict]:
    c = read_json(Path(exp_dir) / "calibration.json")
    params = QualityParams.from_dict(c["quality_params"])
    per = {int(k): Calib(v["T"], tuple(v["tau"]), v["q_min"], tuple(v["tau_reached"]))
           for k, v in c["per_fold"].items()}
    return per, params, c


def build_run_net(run_dir: Path, weights: str):
    """``architectures.build`` for a run's candidate with ``run_dir/weights`` loaded (CPU, eval mode)."""
    import torch

    from app.ml.landmarks.architectures import build
    from training.landmarks.backbones import get_candidate

    cand = read_json(Path(run_dir) / "config.json")["candidate"]
    arch = cand.split(":", 1)[1] if cand.startswith("scratch:") else get_candidate(cand).arch
    net = build(arch)
    net.load_state_dict(torch.load(Path(run_dir) / weights, map_location="cpu", weights_only=True))
    return net.eval()


def _run_hook(run_dir: Path, weights: str, device: str):
    size = int(read_json(Path(run_dir) / "config.json")["input"]["size"])
    return model_hook(build_run_net(run_dir, weights).to(device), device, size)


def fold_hooks(exp_dir: Path, device: str) -> dict:
    """{fold: model hook} with each fold's selected weights (GPU box)."""
    from training.landmarks.run_utils import fold_dirs

    return {k: _run_hook(d, f"best_{read_json(d / 'metrics.json')['best']['variant']}.pt", device)
            for k, d in fold_dirs(exp_dir).items()}


def final_hook(exp_dir: Path, device: str):
    """Model hook of the final all-dev model (the one scored on the locked test)."""
    run = Path(exp_dir) / "all_dev"
    return _run_hook(run, f"final_{read_json(run / 'metrics.json')['best']['variant']}.pt", device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--scenarios", default="S0,S2,S4,S5,S2@1hz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tracker", type=Path, help="tracker.json from tune_tracker (default TrackerParams)")
    ap.add_argument("--transit", choices=("placeholder", "model"), default="placeholder")
    ap.add_argument("--gate-off", action="store_true", help="also report every scenario with the quality gate off")
    ap.add_argument("--auto-from", type=Path, help="JSON with auto_enabled[10] (gates_result.json); default all on")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    m = D.load_manifest()
    oof = load_experiment(a.exp, m, "oof")
    cal, params, _ = load_calibration(a.exp)
    tracker: TrackerParams | dict = TrackerParams()
    if a.tracker:
        t = read_json(a.tracker)
        tracker = {int(k): TrackerParams.from_dict(v) for k, v in t["per_fold"].items()} if "per_fold" in t \
            else TrackerParams.from_dict(t)
    hooks = None
    if a.transit == "model":
        from training.landmarks.run_utils import resolve_device

        hooks = fold_hooks(a.exp, resolve_device(a.device))
    lay = dict(zip(m["clip_uid"], m["layout"]))
    auto = tuple(bool(x) for x in read_json(a.auto_from)["auto_enabled"]) if a.auto_from else ALL_TRUE
    out = run_scenarios(oof, cal, params, a.scenarios.split(","), range(a.seeds), tracker, auto_enabled=auto,
                        hooks=hooks, gates=(True, False) if a.gate_off else (True,), layout_by_clip=lay)
    res = out["results"]
    write_json(a.exp / f"stream_eval_{a.transit}.json", res)
    for key, (st, sr) in out["tables"].items():
        st.to_csv(a.exp / f"stream_{key.replace('/', '_').replace('@', '_')}_stations.csv.gz", index=False)
    for key, r in res.items():
        if "fdr_overall" in r:
            print(f"{key:14s} S4 FDR {r['fdr_overall']:.3f} [{r['fdr_overall_lo']:.3f},{r['fdr_overall_hi']:.3f}]")
        else:
            print(f"{key:14s} recall {r['recall_mean']:.3f} tto {r['tto_median']} "
                  f"abst {r['transit_abstention_frame']} purity {r['best_frame_purity']} ({r['transit_source']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
