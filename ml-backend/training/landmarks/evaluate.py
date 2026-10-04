"""Model evaluation, gates report, locked test scoring and CI-width simulation (M3/M4).

    python -m training.landmarks.evaluate --exp runs/<exp>                 # dev OOF metrics -> eval_oof.json
    python -m training.landmarks.evaluate --exp runs/<exp> --report        # + gates_report.md, gates_result.json
    python -m training.landmarks.evaluate --exp runs/<exp> --ci-sim        # test-size CI widths from OOF
    python -m training.landmarks.evaluate --predict test --run runs/<exp>/all_dev   # final model -> test npz
    python -m training.landmarks.evaluate --exp runs/<exp> --test          # score the locked test ONCE
    python -m training.landmarks.evaluate --temporal-split                  # laptop: private period file

Metrics (patient-cluster bootstrap CIs, group bootstrap alongside):
- clip macro-F1 (primary) and balanced accuracy; per-class precision/recall/F1
  (Wilson CIs); 10x10 confusion; confusable-pair rates; top-2 accuracy;
- patient-level per-station metrics (co-primary): every patient weighs the
  same, whatever their number of clips;
- frame macro-F1 on quality-passing centre frames and on all frames; ECE
  (15 bins), NLL, Brier after the cross-fit T; risk-coverage AURC; accuracy
  by clip position u;
- breakdowns: imaging mode x class, mode-standardised macro-F1 (50/50 WL/NBI
  within every class), cap, room, legacy shift set, temporal hold-out;
- the merged hierarchy (PE+DE, LC+incisura) used by G-MODEL.

Clip label = argmax of the mean log-probability over quality-passing frames
with u in [0.2, 0.8] (all frames when none pass; coverage is reported).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER, STATION_REGION
from training.landmarks import dataset as D
from training.landmarks.calibrate import (
    Calib,
    PredSet,
    brier,
    ece,
    load_experiment,
    load_predset,
    quality_of,
)
from training.landmarks.common import PRIVATE_ROOT, read_csv
from training.landmarks.run_utils import (
    GATES_PATH,
    balanced_accuracy,
    cluster_bootstrap,
    confusion,
    fold_dirs,
    log_softmax,
    macro_f1,
    prf_from_cm,
    read_json,
    softmax,
    wilson,
    write_json,
)

CONFUSABLE = {
    "PE/DE": ("esophagus_proximal", "esophagus_distal"),
    "DE/Z": ("esophagus_distal", "z_line"),
    "CF/LC/I": ("cardia_fundus_retroflex", "lesser_curvature_retroflex", "incisura"),
    "I/A": ("incisura", "antrum"),
    "LC/GC": ("lesser_curvature_retroflex", "corpus_greater_curvature"),
    "B/D2/A": ("duodenal_bulb", "duodenum_descending", "antrum"),
}
OESOPHAGEAL = [i for i, k in enumerate(STATION_ORDER) if STATION_REGION[k] == "esophagus"]
TEMPORAL_PATH = PRIVATE_ROOT / "temporal_holdout.csv"
N_BOOT = 2000


# -- merged hierarchy ------------------------------------------------------------------------------------

def merged_map(gates: dict | None = None) -> tuple[list[str], np.ndarray]:
    """(node names, station index -> node index) for the pre-registered merges."""
    gates = gates or read_json(GATES_PATH)
    merges = gates["merged_hierarchy"]
    member = {s: name for name, ss in merges.items() for s in ss}
    nodes, idx = [], np.zeros(NUM_STATIONS, int)
    for i, k in enumerate(STATION_ORDER):
        name = member.get(k, k)
        if name not in nodes:
            nodes.append(name)
        idx[i] = nodes.index(name)
    return nodes, idx


def merge_logprobs(lp: np.ndarray, idx: np.ndarray, n_nodes: int) -> np.ndarray:
    p = np.exp(lp)
    out = np.zeros((len(p), n_nodes))
    for i, j in enumerate(idx):
        out[:, j] += p[:, i]
    return np.log(np.clip(out, 1e-12, 1))


# -- clip-level aggregation ---------------------------------------------------------------------------------

def clip_predictions(ps: PredSet, T: float, params: QualityParams, q_min: float,
                     u_range: tuple[float, float] = (0.2, 0.8)) -> pd.DataFrame:
    """One row per primary clip: truth, prediction, mean log-probs, metadata."""
    f = ps.frames
    lp = log_softmax(ps.logits, T)
    q = quality_of(f, params)
    good = (f["u"].between(*u_range) & f["layout_ok"]).to_numpy() & (q >= q_min)
    rows = []
    for uid, g in f[f["primary"]].groupby("clip_uid"):
        idx = g.index.to_numpy()  # PredSet frames carry a RangeIndex: labels == positions
        use = idx[good[idx]]
        covered = len(use) > 0
        m = lp[use if covered else idx].mean(0)
        r = f.loc[idx[0]]
        rows.append({"clip_uid": uid, "y": int(r["y"]), "pred": int(m.argmax()),
                     "top2": bool(r["y"] in np.argsort(-m)[:2]), "covered": covered,
                     "patient_id": r["patient_id"], "group_id": r["group_id"], "fold": ps.fold,
                     "mode_major": r["mode_major"], "nbi_frac": r["nbi_frac_f"], "cap": bool(r["cap_b"]),
                     "room": r["room_hash"], "cohort": r["cohort"],
                     **{f"lp{i}": float(v) for i, v in enumerate(m)}})
    return pd.DataFrame(rows)


def _boot(cp: pd.DataFrame, fn, by: str = "patient_id", n: int = N_BOOT):
    y, p = cp["y"].to_numpy(), cp["pred"].to_numpy()
    return cluster_bootstrap(lambda i: fn(y[i], p[i]), cp[by].to_numpy(), n_boot=n)


def per_class_metrics(cp: pd.DataFrame, n_classes: int = NUM_STATIONS, names=STATION_ORDER,
                      n_boot: int = N_BOOT) -> dict:
    """Per-class clip precision (patient-bootstrap CI), recall (Wilson CI) and F1."""
    y, p = cp["y"].to_numpy(), cp["pred"].to_numpy()
    cm = confusion(y, p, n_classes)
    prec, rec, f1 = prf_from_cm(cm)
    per = {}
    for c, key in enumerate(names):
        n_true, n_pred = int(cm[c].sum()), int(cm[:, c].sum())
        if not n_true and not n_pred:
            continue

        def p_stat(i, c=c):
            m = confusion(y[i], p[i], n_classes)
            return m[c, c] / m[:, c].sum() if m[:, c].sum() else np.nan

        _, plo, phi = cluster_bootstrap(p_stat, cp["patient_id"].to_numpy(), n_boot=n_boot)
        rlo, rhi = wilson(cm[c, c], n_true)
        per[key] = {"n": n_true, "precision": prec[c], "precision_lo": plo, "precision_hi": phi,
                    "recall": rec[c], "recall_lo": rlo, "recall_hi": rhi, "f1": f1[c]}
    return per


def clip_metrics(cp: pd.DataFrame, n_classes: int = NUM_STATIONS, names=STATION_ORDER) -> dict:
    y, p = cp["y"].to_numpy(), cp["pred"].to_numpy()
    cm = confusion(y, p, n_classes)
    out = {"n_clips": int(len(cp)), "n_patients": int(cp["patient_id"].nunique()),
           "coverage": float(cp["covered"].mean()) if "covered" in cp else None}
    for name, fn in (("macro_f1", lambda a, b: macro_f1(a, b, n_classes)),
                     ("bal_acc", lambda a, b: balanced_accuracy(a, b, n_classes))):
        pt, lo, hi = _boot(cp, fn)
        _, glo, ghi = _boot(cp, fn, "group_id")
        out[name] = {"value": pt, "lo": lo, "hi": hi, "group_lo": glo, "group_hi": ghi}
    out["per_class"] = per_class_metrics(cp, n_classes, names)
    out["confusion"] = cm.astype(int).tolist()
    if "top2" in cp:
        out["top2_acc"] = float(cp["top2"].mean())
    return out


def confusable_rates(cp: pd.DataFrame) -> dict:
    out = {}
    for name, keys in CONFUSABLE.items():
        ids = [STATION_INDEX[k] for k in keys]
        sel = cp["y"].isin(ids)
        n = int(sel.sum())
        wrong = cp.loc[sel, "pred"].isin(ids) & (cp.loc[sel, "pred"] != cp.loc[sel, "y"])
        out[name] = {"n": n, "rate": float(wrong.mean()) if n else None}
    return out


def patient_level(cp: pd.DataFrame) -> dict:
    """Per-station patient-weighted recall/precision/F1 and their macro average (co-primary)."""
    per = {}
    for c, key in enumerate(STATION_ORDER):
        rec = cp[cp["y"] == c].groupby("patient_id")["pred"].apply(lambda s: float((s == c).mean()))
        prec = cp[cp["pred"] == c].groupby("patient_id")["y"].apply(lambda s: float((s == c).mean()))
        if not len(rec):
            continue
        r, p = float(rec.mean()), float(prec.mean()) if len(prec) else float("nan")
        per[key] = {"patients": int(len(rec)), "recall": r, "precision": p,
                    "f1": 2 * p * r / (p + r) if p + r > 0 else 0.0}
    f1s = [v["f1"] for v in per.values() if np.isfinite(v["f1"])]
    return {"per_station": per, "macro_f1": float(np.mean(f1s)) if f1s else None}


def mode_breakdown(cp: pd.DataFrame) -> dict:
    """Per imaging mode: per-station clip recall (Wilson) and patient-level recall.

    Patient level (the per-mode gate input): one vote per patient, a success
    when at least half of that patient's clips of the station in that mode are
    right, Wilson CI over patients. A few patients with many clips cannot
    carry the scarce WL oesophagus estimate.
    """
    out = {}
    for mode in ("wl", "nbi", "mixed"):
        sub = cp[cp["mode_major"] == mode]
        per = {}
        for c, key in enumerate(STATION_ORDER):
            s = sub[sub["y"] == c]
            if len(s):
                k = int((s["pred"] == c).sum())
                lo, hi = wilson(k, len(s))
                votes = (s["pred"] == c).groupby(s["patient_id"]).mean() >= 0.5
                plo, phi = wilson(int(votes.sum()), len(votes))
                per[key] = {"n": int(len(s)), "recall": k / len(s), "lo": lo, "hi": hi,
                            "patients": int(len(votes)), "patients_correct": int(votes.sum()),
                            "patient_recall": float(votes.mean()), "patient_lo": plo, "patient_hi": phi}
        out[mode] = {"n_clips": int(len(sub)), "macro_f1": macro_f1(sub["y"], sub["pred"]) if len(sub) else None,
                     "per_class": per}
    return out


def mode_standardised_f1(cp: pd.DataFrame) -> dict:
    """Macro-F1 with clip weights that make every class 50/50 WL/NBI (where both exist)."""
    nbi = (cp["nbi_frac"].fillna(0) >= 0.5).to_numpy()
    w = np.ones(len(cp))
    unbalanceable = []
    for c in np.unique(cp["y"]):
        sel = (cp["y"] == c).to_numpy()
        n_n, n_w = int((sel & nbi).sum()), int((sel & ~nbi).sum())
        if n_n and n_w:
            w[sel & nbi] = 0.5 * sel.sum() / n_n
            w[sel & ~nbi] = 0.5 * sel.sum() / n_w
        else:
            unbalanceable.append(STATION_ORDER[c])
    y, p = cp["y"].to_numpy(), cp["pred"].to_numpy()
    pt, lo, hi = cluster_bootstrap(lambda i: macro_f1(y[i], p[i], w=w[i]), cp["patient_id"].to_numpy(),
                                   n_boot=N_BOOT)
    return {"value": pt, "lo": lo, "hi": hi, "unbalanceable_classes": unbalanceable}


def group_breakdown(cp: pd.DataFrame, col: str, names: dict | None = None) -> dict:
    out = {}
    for v, sub in cp.groupby(col):
        out[str(names.get(v, v) if names else v)] = {
            "n_clips": int(len(sub)), "n_patients": int(sub["patient_id"].nunique()),
            "macro_f1": macro_f1(sub["y"], sub["pred"]), "accuracy": float((sub["y"] == sub["pred"]).mean())}
    return out


# -- frame-level --------------------------------------------------------------------------------------------

def frame_metrics(oof: dict[int, PredSet], cal: dict[int, Calib], params: QualityParams) -> dict:
    ys, ps_all, cen, qual_ok, us = [], [], [], [], []
    for k, ps in oof.items():
        f = ps.frames
        prim = (f["primary"] & f["layout_ok"]).to_numpy()
        ys.append(f["y"].to_numpy()[prim])
        ps_all.append(softmax(ps.logits[prim], cal[k].T))
        q = quality_of(f, params)[prim] >= cal[k].q_min
        qual_ok.append(q)
        u = f["u"].to_numpy()[prim]
        us.append(u)
        cen.append(q & (u >= 0.2) & (u <= 0.8))
    y, p = np.concatenate(ys), np.concatenate(ps_all)
    cen, qok, u = np.concatenate(cen), np.concatenate(qual_ok), np.concatenate(us)
    pred = p.argmax(1)
    calm = qok & (u >= 0.1) & (u <= 0.9)
    conf = p.max(1)[qok]
    order = np.argsort(-conf)
    risk = np.cumsum(pred[qok][order] != y[qok][order]) / np.arange(1, qok.sum() + 1)
    by_u = {}
    for lo in np.arange(0, 1.0, 0.1):
        s = qok & (u >= lo) & (u < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
        by_u[f"{lo:.1f}"] = float((pred[s] == y[s]).mean()) if s.any() else None
    return {
        "centre_quality": {"n": int(cen.sum()), "macro_f1": macro_f1(y[cen], pred[cen]),
                           "accuracy": float((pred[cen] == y[cen]).mean())},
        "all_frames": {"n": int(len(y)), "macro_f1": macro_f1(y, pred), "accuracy": float((pred == y).mean())},
        "calibration": {"n": int(calm.sum()), "ece": ece(p[calm], y[calm]), "brier": brier(p[calm], y[calm]),
                        "nll": float(-np.log(np.clip(p[calm][np.arange(calm.sum()), y[calm]], 1e-12, 1)).mean())},
        "aurc": float(risk.mean()) if len(risk) else None,
        "accuracy_by_u": by_u,
    }


# -- the full dev report -------------------------------------------------------------------------------------

def load_cal(exp_dir: Path) -> tuple[dict[int, Calib], QualityParams]:
    from training.landmarks.stream_eval import load_calibration

    per, params, _ = load_calibration(exp_dir)
    return per, params


def evaluate_oof(exp_dir: Path, manifest: pd.DataFrame | None = None) -> dict:
    manifest = D.load_manifest() if manifest is None else manifest
    oof = load_experiment(exp_dir, manifest, "oof")
    cal, params = load_cal(exp_dir)
    cp = pd.concat([clip_predictions(ps, cal[k].T, params, cal[k].q_min) for k, ps in oof.items()],
                   ignore_index=True)
    nodes, idx = merged_map()
    lp = cp[[f"lp{i}" for i in range(NUM_STATIONS)]].to_numpy()
    mcp = cp.assign(y=idx[cp["y"]], pred=merge_logprobs(lp, idx, len(nodes)).argmax(1))
    rooms = {h: f"room_{i + 1}" for i, h in enumerate(cp["room"].value_counts().index)}
    out = {
        "exp": Path(exp_dir).name, "folds": sorted(oof),
        "clip": clip_metrics(cp),
        "clip_merged": clip_metrics(mcp, len(nodes), nodes),
        "patient_level": patient_level(cp),
        "confusable": confusable_rates(cp),
        "frame": frame_metrics(oof, cal, params),
        "by_mode": mode_breakdown(cp),
        "mode_standardised_macro_f1": mode_standardised_f1(cp),
        "by_cap": group_breakdown(cp, "cap"),
        "by_room": group_breakdown(cp, "room", rooms),
        "per_fold_macro_f1": {str(k): macro_f1(g["y"], g["pred"]) for k, g in cp.groupby("fold")},
    }
    holdout = Path(exp_dir) / "holdout"
    if (holdout / "oof_fold0.npz").exists():
        hp = load_predset(holdout / "oof_fold0.npz", manifest)
        hc = clip_predictions(hp, float(np.median([c.T for c in cal.values()])), params,
                              float(np.median([c.q_min for c in cal.values()])))
        out["temporal_holdout"] = clip_metrics(hc)
    shift = Path(exp_dir) / "all_dev" / "shift_preds.npz"
    if shift.exists():
        sp = load_predset(shift, manifest)
        sc = clip_predictions(sp, float(np.median([c.T for c in cal.values()])), params, 0.0)
        out["legacy_shift"] = clip_metrics(sc)
    write_json(Path(exp_dir) / "eval_oof.json", out)
    cp.drop(columns=["patient_id", "group_id", "room"]).to_csv(Path(exp_dir) / "clip_predictions_oof.csv.gz",
                                                                 index=False)
    return out


# -- gates ---------------------------------------------------------------------------------------------------

def _opt(path: Path):
    return read_json(path) if path.exists() else None


def data_gate() -> tuple[str, dict]:
    """folds --verify on the manifest + the static-overlay scan report (exact zeros: train.py asserts)."""
    from training.landmarks.common import MANIFEST_PATH, REPORTS_DIR
    from training.landmarks.folds import verify

    if not MANIFEST_PATH.exists():
        return "NOT_RUN", {"reason": "manifest not available"}
    errs = verify(read_csv(MANIFEST_PATH))
    scan = REPORTS_DIR / "static_scan.txt"
    scan_ok = scan.exists() and "ok: True" in scan.read_text()
    status = "PASS" if not errs and scan_ok else "FAIL" if errs else "NOT_RUN"
    return status, {"folds_verify_errors": len(errs), "static_scan_ok": scan_ok if scan.exists() else None}


def leak_gate() -> tuple[str, dict]:
    from training.landmarks.common import REPORTS_DIR

    rep = REPORTS_DIR / "leak_audit.md"
    if not rep.exists():
        return "NOT_RUN", {"reason": "leak_audit.md not found (run leak_audit.py on the laptop)"}
    text = rep.read_text()
    return ("PASS" if any(line.startswith("PASS") for line in text.splitlines()) else "FAIL"), {"report": rep.name}


def fallback_fold(exp_dir: Path) -> int | None:
    """The pre-registered ECE fallback: the fold model with the MEDIAN inner-val clip macro-F1."""
    scores = {k: read_json(d / "metrics.json")["best"].get("clip_macro_f1") for k, d in fold_dirs(exp_dir).items()}
    scores = {k: v for k, v in scores.items() if v is not None}
    if not scores:
        return None
    order = sorted(scores, key=lambda k: (scores[k], k))
    return int(order[(len(order) - 1) // 2])


def station_gate(per_class: dict, by_mode: dict, s4_fdr: dict, g: dict) -> tuple[list[bool], dict]:
    """G-STATION-k for every station -> (auto_enabled[10], per-station detail).

    ``per_class``: per_class_metrics; ``by_mode``: mode_breakdown;
    ``s4_fdr``: {station: {"fdr": ...}} measured with the station ENABLED
    (tune_tracker's ``fdr_per_station_if_enabled``); a missing entry is not a
    failure (no S4 stream had that station). Proximal/distal oesophagus are
    gated per mode: NBI clip recall and the WL patient-level Wilson lower
    bound. Serving has ONE flag per station, so a WL failure disables the
    station in every mode (``auto_enabled``); ``by_mode`` in the detail
    records what a mode-aware tracker could enable (WL frames / NBI frames).
    """
    auto, per = [], {}
    for key in STATION_ORDER:
        pc = per_class.get(key, {})
        base = bool(pc) and pc["precision"] >= g["min_clip_precision"] and pc["precision_lo"] >= \
            g["min_clip_precision_lower_ci"] and pc["recall"] >= g["min_clip_recall"]
        fdr = (s4_fdr.get(key) or {}).get("fdr")
        base &= fdr is None or fdr <= g["max_s4_fdr"]
        wl_ok = nbi_ok = True
        if key in g["per_mode_stations"]:
            wl = by_mode["wl"]["per_class"].get(key)
            nb = by_mode["nbi"]["per_class"].get(key)
            nbi_ok = bool(nb) and nb["recall"] >= g["per_mode"]["min_recall_nbi"]
            wl_ok = bool(wl) and wl["patient_lo"] >= g["per_mode"]["min_recall_wl_lower_ci"]
        ok = bool(base and wl_ok and nbi_ok)
        per[key] = {"pass": ok, "precision": pc.get("precision"), "precision_lo": pc.get("precision_lo"),
                    "recall": pc.get("recall"), "s4_fdr": fdr,
                    "by_mode": {"wl": bool(base and wl_ok), "nbi": bool(base and nbi_ok)}}
        auto.append(ok)
    return auto, per


def outcome(statuses: dict[str, str], auto_enabled, gates: dict) -> str:
    """The pre-registered outcome (gates.json "outcomes") from gate statuses.

    GO: every blocking gate passes (G-STREAM included) and every station
    passes G-STATION. PARTIAL: every blocking gate passes with G-STREAM
    measured on the shipped station set, >= 1 station auto-enabled. NO-GO: a
    blocking gate fails or no station is auto-enabled. PENDING: a blocking
    gate has not run (or only on placeholder transit) yet.
    """
    blocking = [k for k, v in gates.items() if isinstance(v, dict) and v.get("blocking")]
    st = {k: statuses.get(k, "NOT_RUN") for k in blocking}
    if any(v.startswith("FAIL") for v in st.values()) or not any(auto_enabled):
        return "NO-GO"
    if any(v != "PASS" for v in st.values()):
        return "PENDING (" + ", ".join(sorted(k for k, v in st.items() if v != "PASS")) + ")"
    return "GO" if all(auto_enabled) else "PARTIAL"


def gate_results(exp_dir: Path, ev: dict, gates: dict | None = None) -> dict:
    """PASS / FAIL / NOT_RUN per gate, per-station auto_enabled and the outcome.

    G-STATION's S4 FDR input and G-STREAM both come from tune_tracker's nested
    held-out replay (tracker.json): tuned tracker and thresholds per outer
    fold, auto_enabled recomputed per outer fold. Stations disabled on dev
    with ``tune_tracker --disable`` stay disabled.
    """
    gates = gates or read_json(GATES_PATH)
    exp_dir = Path(exp_dir)
    sanity = _opt(exp_dir / "sanity.json") or {}
    stream = _opt(exp_dir / "stream_eval_model.json") or _opt(exp_dir / "stream_eval_placeholder.json")
    tracker = _opt(exp_dir / "tracker.json")
    res: dict = {"G-DATA": data_gate(), "G-LEAK": leak_gate()}

    g = gates["G-PIPELINE"]
    perm = sanity.get("perm")
    res["G-PIPELINE"] = ("NOT_RUN", None) if perm is None else (
        "PASS" if perm["clip_macro_f1"] <= g["max_clip_macro_f1"] else "FAIL", perm["clip_macro_f1"])

    g = gates["G-MODEL"]
    cm = ev["clip_merged"]
    node_rec = {k: v["recall"] for k, v in cm["per_class"].items()}
    e = ev["frame"]["calibration"]["ece"]
    ok = (cm["macro_f1"]["value"] >= g["min_clip_macro_f1"] and cm["macro_f1"]["lo"] >= g["min_clip_macro_f1_lower_ci"]
          and min(node_rec.values()) >= g["min_node_recall"] and e <= g["max_ece"])
    res["G-MODEL"] = ("PASS" if ok else "FAIL", {"macro_f1": cm["macro_f1"], "min_node_recall": min(node_rec.values()),
                                                 "ece": e})

    g = gates["G-SHORTCUT"]
    if not sanity.get("nbi"):
        res["G-SHORTCUT"] = ("NOT_RUN", None)
    else:
        eff = {"nbi": sanity["nbi"]["effect"], "mode_switch": (sanity.get("mode_switch") or {}).get("effect"),
               "cap": (sanity.get("cap") or {}).get("effect")}
        ok = (abs(eff["nbi"]) <= g["max_nbi_effect"]
              and (eff["mode_switch"] is None or abs(eff["mode_switch"]) <= g["max_mode_switch_effect"])
              and (eff["cap"] is None or eff["cap"] <= g["max_cap_effect"]))
        pc = sanity.get("positive_control")
        if pc is None:
            status = "INCOMPLETE" if ok else "FAIL"
        else:
            status = ("PASS" if ok and pc.get("tripped") else "FAIL")
        res["G-SHORTCUT"] = (status, {**eff, "positive_control_tripped": pc.get("tripped") if pc else None})

    if tracker and "fdr_per_station_if_enabled" in tracker["heldout_pooled"]["S4"]:
        s4, s4_source = tracker["heldout_pooled"]["S4"]["fdr_per_station_if_enabled"], "tracker.json (nested, tuned)"
    else:
        s4, s4_source = (stream or {}).get("S4", {}).get("fdr_per_station", {}), "stream_eval (default tracker)"
    auto, per = station_gate(ev["clip"]["per_class"], ev["by_mode"], s4, gates["G-STATION"])
    forced = set((tracker or {}).get("force_disabled", []))
    for i, key in enumerate(STATION_ORDER):
        if key in forced:
            auto[i] = False
            per[key]["pass"] = False
            per[key]["force_disabled"] = True
    res["G-STATION"] = ("PASS" if all(auto) else "PARTIAL", per)

    g = gates["G-STREAM"]
    if not tracker:
        res["G-STREAM"] = ("NOT_RUN", None)
    else:
        s2 = tracker["heldout_pooled"]["S2"]  # auto-enabled stations only (per outer fold)
        s4p = tracker["heldout_pooled"]["S4"]
        rec_each = [v["recall"] for v in s2["recall_per_station"].values()]
        checks = {
            "recall_mean": s2["recall_mean"] >= g["min_s2_recall_mean"],
            "recall_each": min(rec_each) >= g["min_s2_recall_each"] if rec_each else False,
            "s4_fdr": (s4p.get("fdr_overall") or 0) <= g["max_s4_fdr_overall"],
            "tto": (s2["tto_median"] or 1e9) <= g["max_tto_median_s"],
            "within_clip": (s2["within_clip"] or 0) >= g["min_within_clip"],
            "transit_abstention": (s2["transit_abstention_frame"] or 0) >= g["min_transit_abstention"],
            "best_frame_purity": (s2["best_frame_purity"] or 0) >= g["min_best_frame_purity"],
        }
        if stream and "S5" in stream and "S2" in stream:
            checks["s5_drop"] = stream["S2"]["recall_mean"] - stream["S5"]["recall_mean"] <= g["max_s5_recall_drop"]
        placeholder = "model" not in tracker.get("transit_source", [])
        status = "PASS" if all(checks.values()) else "FAIL"
        if placeholder and status == "PASS":
            status = "PASS (placeholder transit: optimistic)"
        res["G-STREAM"] = (status, checks)
    test = _opt(exp_dir / "eval_test.json")
    if test:
        res["G-TEST"] = (test["G-TEST"], test["primary_non_inferiority"])
    serve = _opt(exp_dir / "serve_result.json")  # I1, written by hand: {"status": "PASS"|"FAIL", ...}
    if serve:
        res["G-SERVE"] = (serve["status"], {k: v for k, v in serve.items() if k != "status"})
    statuses = {k: v[0] for k, v in res.items()}
    return {"gates": {k: {"status": v[0], "value": v[1]} for k, v in res.items()}, "auto_enabled": auto,
            "auto_enabled_by_mode": {m: [per[k]["by_mode"][m] and not per[k].get("force_disabled")
                                         for k in STATION_ORDER] for m in ("wl", "nbi")},
            "s4_fdr_source": s4_source, "outcome": outcome(statuses, auto, gates)}


def write_gates_report(exp_dir: Path, ev: dict, gr: dict) -> Path:
    lines = [f"# Gates report: {Path(exp_dir).name}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from dev OOF "
             f"({ev['clip']['n_clips']} clips, {ev['clip']['n_patients']} patients). "
             f"gates.json frozen: {read_json(GATES_PATH).get('frozen')}.", "",
             "| Gate | Status | Value |", "|---|---|---|"]
    for name, r in gr["gates"].items():
        v = r["value"]
        if name == "G-STATION":
            v = ", ".join(f"{k}:{'ok' if s['pass'] else 'manual'}" for k, s in v.items())
        lines.append(f"| {name} | {r['status']} | `{_short(v)}` |")
    c = ev["clip"]
    lines += ["", "## Clip level (10 stations)", "",
              f"macro-F1 {c['macro_f1']['value']:.3f} [{c['macro_f1']['lo']:.3f}, {c['macro_f1']['hi']:.3f}] "
              f"(group CI [{c['macro_f1']['group_lo']:.3f}, {c['macro_f1']['group_hi']:.3f}]); "
              f"balanced accuracy {c['bal_acc']['value']:.3f}; top-2 {c.get('top2_acc', float('nan')):.3f}; "
              f"coverage {c['coverage']:.3f}; patient-level macro-F1 {ev['patient_level']['macro_f1']:.3f}", "",
              "| station | n | precision [CI] | recall [Wilson] | F1 |", "|---|---|---|---|---|"]
    for k, v in c["per_class"].items():
        lines.append(f"| {k} | {v['n']} | {v['precision']:.3f} [{v['precision_lo']:.3f}, {v['precision_hi']:.3f}] | "
                     f"{v['recall']:.3f} [{v['recall_lo']:.3f}, {v['recall_hi']:.3f}] | {v['f1']:.3f} |")
    m = ev["clip_merged"]
    lines += ["", f"Merged hierarchy macro-F1 {m['macro_f1']['value']:.3f} [{m['macro_f1']['lo']:.3f}, "
                  f"{m['macro_f1']['hi']:.3f}]", "",
              f"Frame: ECE {ev['frame']['calibration']['ece']:.3f}, NLL {ev['frame']['calibration']['nll']:.3f}, "
              f"Brier {ev['frame']['calibration']['brier']:.3f}; centre-frame macro-F1 "
              f"{ev['frame']['centre_quality']['macro_f1']:.3f}", "",
              f"Mode-standardised macro-F1 {ev['mode_standardised_macro_f1']['value']:.3f}", "",
              "Confusable pairs: " + ", ".join(f"{k} {v['rate']:.3f} (n={v['n']})"
                                               for k, v in ev["confusable"].items() if v["rate"] is not None),
              "", f"auto_enabled: {gr['auto_enabled']}",
              f"auto_enabled_by_mode (informational; serving has one flag per station): "
              f"{gr.get('auto_enabled_by_mode')}", "",
              f"**Outcome: {gr.get('outcome')}**", ""]
    path = Path(exp_dir) / "gates_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _short(v, n: int = 160) -> str:
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


# -- CI-width simulation -------------------------------------------------------------------------------------

def ci_simulation(exp_dir: Path, n_test_patients: int = 19, n_sim: int = 500, seed: int = 0,
                  manifest: pd.DataFrame | None = None) -> dict:
    """Spread of the G-TEST primary endpoints on test-sized patient samples drawn from dev OOF."""
    cp = pd.read_csv(Path(exp_dir) / "clip_predictions_oof.csv.gz")
    ev = read_json(Path(exp_dir) / "eval_oof.json")
    gates = read_json(GATES_PATH)["G-TEST"]["primary_endpoints"]
    m = D.load_manifest() if manifest is None else manifest
    cp["patient_id"] = cp["clip_uid"].map(dict(zip(m["clip_uid"], m["patient_id"])))
    pats = cp["patient_id"].unique()
    rng = np.random.default_rng(seed)
    rows_of = {p: g.index.to_numpy() for p, g in cp.groupby("patient_id")}
    f1s = []
    for _ in range(n_sim):
        # a NEW test-sized sample: patients drawn WITH replacement (cluster bootstrap of size
        # n_test); without replacement from 72 the spread shrinks by the finite-population factor
        s = cp.loc[np.concatenate([rows_of[p] for p in rng.choice(pats, n_test_patients, replace=True)])]
        f1s.append(macro_f1(s["y"], s["pred"]))
    f1s = np.asarray(f1s)
    ref = ev["clip"]["macro_f1"]["value"]
    margin = gates["clip_macro_f1"]["non_inferiority_margin"]
    out = {"n_test_patients": n_test_patients, "n_sim": n_sim,
           "clip_macro_f1": {"dev_point": ref, "sim_p2.5": float(np.percentile(f1s, 2.5)),
                             "sim_p97.5": float(np.percentile(f1s, 97.5)),
                             "half_width": float((np.percentile(f1s, 97.5) - np.percentile(f1s, 2.5)) / 2),
                             "p_fail_at_true_dev": float((f1s < ref - margin).mean())}}
    # the G-TEST references are tune_tracker's nested held-out replays (auto_enabled per fold)
    st_path = Path(exp_dir) / "tracker_heldout_S2_stations.csv.gz"
    s4_path = Path(exp_dir) / "tracker_heldout_S4_stations.csv.gz"
    if not (st_path.exists() and s4_path.exists()):
        st_path, s4_path = Path(exp_dir) / "stream_S2_stations.csv.gz", Path(exp_dir) / "stream_S4_stations.csv.gz"
    if st_path.exists() and s4_path.exists():
        st, s4 = pd.read_csv(st_path), pd.read_csv(s4_path)
        if "auto" in st:
            st, s4 = st[st["auto"].astype(bool)], s4[s4["auto"].astype(bool)]
        s4 = s4[s4["station"] == s4["removed"]]
        st = st[st["present"]].reset_index(drop=True)
        s4 = s4.reset_index(drop=True)
        st_of = {p: g.index.to_numpy() for p, g in st.groupby("patient_id")}
        s4_of = {p: g.index.to_numpy() for p, g in s4.groupby("patient_id")}
        none = np.zeros(0, int)
        recs, fdrs = [], []
        for _ in range(n_sim):
            pick = rng.choice(pats, n_test_patients, replace=True)
            a = st.loc[np.concatenate([st_of.get(p, none) for p in pick])]
            recs.append(a.groupby("station")["observed"].mean().mean() if len(a) else np.nan)
            b = s4.loc[np.concatenate([s4_of.get(p, none) for p in pick])]
            fdrs.append(b["observed"].mean() if len(b) else np.nan)
        recs, fdrs = np.asarray(recs), np.asarray(fdrs)
        r_ref = st.groupby("station")["observed"].mean().mean()
        f_ref = s4["observed"].mean()
        out["recall_mean"] = {"dev_point": float(r_ref), "sim_p2.5": float(np.nanpercentile(recs, 2.5)),
                              "sim_p97.5": float(np.nanpercentile(recs, 97.5)),
                              "p_fail_at_true_dev": float(
                                  (recs < r_ref - gates["recall_mean"]["non_inferiority_margin"]).mean())}
        lim = min(f_ref + gates["s4_fdr_overall"]["non_inferiority_margin"], gates["s4_fdr_overall"]["absolute_max"])
        out["s4_fdr_overall"] = {"dev_point": float(f_ref), "sim_p2.5": float(np.nanpercentile(fdrs, 2.5)),
                                 "sim_p97.5": float(np.nanpercentile(fdrs, 97.5)),
                                 "p_fail_at_true_dev": float((fdrs > lim).mean())}
        out["stream_source"] = st_path.name
    out["resampling"] = "patients WITH replacement (cluster bootstrap of the test size)"
    write_json(Path(exp_dir) / "ci_sim.json", out)
    return out


# -- predictions of the final model on test / shift -----------------------------------------------------------

def predict_split(run_dir: Path, split: str, device: str = "auto") -> Path:
    """Final (all-dev) model -> fp32 predictions for every cached frame of a split's streams."""
    from training.landmarks.run_utils import resolve_device
    from training.landmarks.stream_eval import build_run_net
    from training.landmarks.train import predict_table, write_predictions

    device = resolve_device(device)
    cfg = read_json(run_dir / "config.json")
    met = read_json(run_dir / "metrics.json")
    cand = cfg["candidate"]
    net = build_run_net(run_dir, f"final_{met['best']['variant']}.pt").to(device)
    m = D.load_manifest()
    frames = D.load_frames()
    patients = sorted(set(m.loc[m["split"] == split, "patient_id"]))
    clips = D.stream_clips(m, patients) if split == "test" else m[(m["split"] == split)
                                                                  & (m["exclude_reason"] == "")]
    table = D.eval_frame_table(frames, clips)
    lo, em = predict_table(net, table, D.ImageSpec(int(cfg["input"]["size"])), device)
    out = run_dir / f"{split}_preds.npz"
    write_predictions(out, table, lo, em, variant=met["best"]["variant"], fold=-1,
                      input_size=int(cfg["input"]["size"]), candidate=cand)
    return out


def check_test_protocol(exp_dir: Path, transit: str = "model") -> dict:
    """The locked test must be replayed exactly like its dev reference; checked BEFORE the look.

    The dev reference (tracker.json ``heldout_pooled``) was produced with a
    transit source (``model``: fold models classify the synthetic transit
    frames) and the shipped ``auto_enabled`` comes from gates_result.json. A
    test replayed under another protocol is not comparable, so nothing is
    scored (and the one look is not spent).
    """
    from training.landmarks.stream_sim import PLACEHOLDER

    exp_dir = Path(exp_dir)
    tracker = _opt(exp_dir / "tracker.json") or {}
    gr = _opt(exp_dir / "gates_result.json") or {}
    dev_source = sorted(tracker.get("transit_source") or [])
    test_source = "model" if transit == "model" else PLACEHOLDER
    problems = []
    if not tracker:
        problems.append("tracker.json missing (run tune_tracker)")
    elif dev_source != [test_source]:
        problems.append(f"transit source differs: dev reference {dev_source}, test {test_source!r}")
    if "auto_enabled" not in gr:
        problems.append("gates_result.json with the shipped auto_enabled missing (run evaluate --report)")
    return {"test": test_source, "dev_reference": dev_source, "problems": problems}


def score_test(exp_dir: Path, force_reason: str | None = None, transit: str = "model", device: str = "auto",
               hook=None, manifest: pd.DataFrame | None = None, cache_root: Path | None = None) -> dict:
    """G-TEST: locked test scored ONCE with the shipped calibration, tracker and auto_enabled.

    Transit frames are classified by the final all-dev model (``hook``, by
    default from ``all_dev/final_<variant>.pt``) at the shipped T, the same
    protocol as the dev reference; ``check_test_protocol`` refuses anything else.
    """
    from app.ml.landmarks.tracker import TrackerParams
    from training.landmarks.common import CACHE_ROOT
    from training.landmarks.stream_eval import (
        aggregate,
        fast_observed,
        final_hook,
        replay,
        station_records,
        stream_record,
    )
    from training.landmarks.stream_sim import StreamConfig, build_streams, fill_transit

    exp_dir = Path(exp_dir)
    marker = exp_dir / "TEST_SCORED"
    if marker.exists() and not force_reason:
        raise SystemExit(f"the locked test was already scored ({marker.read_text().strip()}); "
                         "a second look needs --force-reason and is reported as such")
    proto = check_test_protocol(exp_dir, transit)
    if proto["problems"]:
        raise SystemExit("locked test NOT scored (protocol differs from the dev reference): "
                         + "; ".join(proto["problems"]))
    m = D.load_manifest() if manifest is None else manifest
    tracker = read_json(exp_dir / "tracker.json")
    auto = [bool(a) for a in read_json(exp_dir / "gates_result.json")["auto_enabled"]]
    if transit == "model" and hook is None:
        from training.landmarks.run_utils import resolve_device

        hook = final_hook(exp_dir, resolve_device(device))
    preds = load_predset(exp_dir / "all_dev" / "test_preds.npz", m)
    ship = tracker.get("shipped_calibration") or read_json(exp_dir / "calibration.json")["shipped"]
    cal = Calib(ship["T"], tuple(ship["tau"]), ship["q_min"])
    params = QualityParams.from_dict(read_json(exp_dir / "calibration.json")["quality_params"])
    tp = TrackerParams.from_dict(tracker["shipped"]["tracker"])
    cp = clip_predictions(preds, cal.T, params, cal.q_min)
    out = {"clip": clip_metrics(cp), "patient_level": patient_level(cp),
           "frame": frame_metrics({-1: preds}, {-1: cal}, params)}
    lay = dict(zip(m["clip_uid"], m["layout"]))
    for scen in ("S2", "S4", "S5", "S2@1hz"):
        name, _, rate = scen.partition("@")
        st_rows, sr_rows = [], []
        for seed in range(5):
            streams = build_streams(preds, StreamConfig(name, hz=1.0 if rate else 2.0, seed=seed), cal.T, params)
            fill_transit(streams, cal.T, params, hook if transit == "model" else None,
                         cache_root=CACHE_ROOT if cache_root is None else cache_root, layout_by_clip=lay)
            for s in streams:
                rep = replay(s, tp, cal.tau, cal.q_min, auto)
                st_rows += station_records(s, rep.observed_at, auto_enabled=auto, observed_if_enabled=fast_observed(
                    rep.q, tp.evidence_needed, tp.evidence_window))
                sr_rows.append(stream_record(s, rep))
        out[scen] = aggregate(pd.DataFrame(st_rows), pd.DataFrame(sr_rows), auto)
    ev = read_json(exp_dir / "eval_oof.json")
    gates = read_json(GATES_PATH)["G-TEST"]["primary_endpoints"]
    tr = tracker["heldout_pooled"]
    ni = {
        "clip_macro_f1": out["clip"]["macro_f1"]["value"] >= ev["clip"]["macro_f1"]["value"]
        - gates["clip_macro_f1"]["non_inferiority_margin"],
        "s4_fdr_overall": (out["S4"]["fdr_overall"] or 0) <= min(
            (tr["S4"].get("fdr_overall") or 0) + gates["s4_fdr_overall"]["non_inferiority_margin"],
            gates["s4_fdr_overall"]["absolute_max"]),
        "recall_mean": (out["S2"]["recall_mean"] or 0) >= (tr["S2"]["recall_mean"] or 0)
        - gates["recall_mean"]["non_inferiority_margin"],
    }
    test_patients = sorted(set(preds.frames["patient_id"]))
    out["primary_non_inferiority"] = ni
    out["G-TEST"] = "PASS" if all(ni.values()) else "FAIL"
    out["ece_fallback_triggered"] = out["frame"]["calibration"]["ece"] > read_json(GATES_PATH)["G-TEST"][
        "ece_fallback"]["max_ece"]
    out["test_patients_sha256"] = hashlib.sha256(",".join(test_patients).encode()).hexdigest()
    out["fallback_fold"] = fallback_fold(exp_dir)
    out["transit_source"] = {"test": proto["test"], "dev_reference": proto["dev_reference"]}
    out["auto_enabled"] = auto
    write_json(exp_dir / ("eval_test.json" if not force_reason else "eval_test_second_look.json"), out)
    marker.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {force_reason or 'first look'}\n")
    return out


# -- temporal hold-out split (laptop; reads private OSD dates) ---------------------------------------------------

def temporal_split(cutoff: str = "2026-01-01") -> Path:
    """patient_id -> early|late by median OSD date; only the period leaves this function."""
    from training.landmarks.common import AUDIT_DIR, drop_full_length

    audit = drop_full_length(read_csv(AUDIT_DIR / "clips_audit.csv"))
    date_by_md5 = {r["md5"]: r["osd_date"] for r in audit if r.get("osd_date")}
    m = D.load_manifest()
    m = m[(m["split"] == "dev") & (m["exclude_reason"] == "")]
    m = m.assign(date=m["md5"].map(date_by_md5))
    m["date"] = pd.to_datetime(m["date"], errors="coerce", dayfirst=True)
    med = m.dropna(subset=["date"]).groupby("patient_id")["date"].median()
    period = np.where(med < pd.Timestamp(cutoff), "early", "late")
    out = pd.DataFrame({"patient_id": med.index, "period": period})
    out.to_csv(TEMPORAL_PATH, index=False)
    return TEMPORAL_PATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path)
    ap.add_argument("--report", action="store_true", help="also write gates_report.md and gates_result.json")
    ap.add_argument("--ci-sim", action="store_true")
    ap.add_argument("--test", action="store_true", help="score the locked test (once)")
    ap.add_argument("--force-reason", help="second look at the test set (logged)")
    ap.add_argument("--transit", choices=("model", "placeholder"), default="model",
                    help="--test: transit classification; must match tracker.json's (refused otherwise)")
    ap.add_argument("--predict", choices=("test", "shift"))
    ap.add_argument("--run", type=Path, help="all_dev run dir for --predict")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temporal-split", action="store_true")
    ap.add_argument("--cutoff", default="2026-01-01")
    a = ap.parse_args(argv)
    if a.temporal_split:
        print(temporal_split(a.cutoff))
        return 0
    if a.predict:
        print(predict_split(a.run, a.predict, a.device))
        return 0
    if a.exp is None:
        ap.error("--exp is required")
    if a.test:
        out = score_test(a.exp, a.force_reason, a.transit, a.device)
        print(f"G-TEST {out['G-TEST']}: {out['primary_non_inferiority']}")
        return 0
    if a.ci_sim:
        print(ci_simulation(a.exp))
        return 0
    ev = evaluate_oof(a.exp)
    c = ev["clip"]["macro_f1"]
    print(f"clip macro-F1 {c['value']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]; merged "
          f"{ev['clip_merged']['macro_f1']['value']:.3f}; ECE {ev['frame']['calibration']['ece']:.3f}")
    if a.report:
        gr = gate_results(a.exp, ev)
        write_json(a.exp / "gates_result.json", gr)
        print(write_gates_report(a.exp, ev, gr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
