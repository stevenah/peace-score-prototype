"""Calibration, rejection thresholds and the quality gate (M3), from OOF predictions only.

    python -m training.landmarks.calibrate --exp runs/<exp>        # writes runs/<exp>/calibration.json

- Temperature T: NLL-optimal on quality-passing frames over ALL u (the frame
  population serving sees), hard labels of primary clips.
- Per-station tau_c: the smallest threshold on calibrated p_top such that
  frames predicted c with p >= tau_c have CLOSED-SET precision >= target
  (0.90 default; every evaluated frame belongs to one of the 10 stations, so
  live precision will be lower). Fit on u in [0.1, 0.9] quality-passing
  frames, floor 0.5, at least ``min_accept`` accepted frames; a station that
  cannot reach the target gets tau = 1.0 (never "ok") and ``reached=False``.
- q_min: a percentile (P10 default) of clip-centre frame quality.
- sharp_ref: median sharpness of clip-centre frames at 224 px (dev clips of
  the cache, never test): the QualityParams reference shipped in the bundle.

Honest reporting (critic C4): for outer fold k, T_k is fit on model k's
inner-val predictions; tau_k and q_min_k are fit on the OTHER folds' OOF only
(each calibrated with its own T_j). Shipped values are fit on all dev OOF.
Inner-val of model j is fold (j + 1) % 5, so T_{k-1} was fit on fold k's
labels: whenever fold k is reported, that one temperature is refit on model
(k-1)'s own outer fold instead (``heldout_temperatures``, stored as
``temps_heldout`` in calibration.json and used by tune_tracker).

Frame status mirrors ``RealLandmarkClassifier``: low_quality if quality <
q_min (or the layout was not verified), else uncertain if p_top < tau[top],
else ok.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from training.landmarks import dataset as D
from training.landmarks.common import CACHE_ROOT, MANIFEST_PATH
from training.landmarks.run_utils import fold_dirs, log_softmax, softmax, write_json

TAU_FLOOR = 0.5
TAU_U_RANGE = (0.1, 0.9)
PRECISION_TARGET = 0.90
Q_PCT = 10
MIN_ACCEPT = 20


# -- prediction sets ----------------------------------------------------------------------------------

@dataclass
class PredSet:
    """Per-frame predictions of one model on a set of patients, joined with clip info."""

    fold: int
    frames: pd.DataFrame  # clip_uid, src_idx, u, t_s, metrics, layout_ok, clip columns, y, primary
    logits: np.ndarray
    emb: np.ndarray | None = None

    def subset(self, mask: np.ndarray) -> "PredSet":
        return PredSet(self.fold, self.frames[mask].reset_index(drop=True), self.logits[mask],
                       None if self.emb is None else self.emb[mask])


CLIP_COLS = ["label", "soft_label", "exclude_reason", "eval_exclude_b", "patient_id", "group_id", "fold_i",
             "rel_t_s_f", "dur_s_f", "mode_major", "nbi_frac_f", "cap_b", "room_hash", "cohort", "rel_path",
             "folder_class", "split"]


def attach_clip_info(frames: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    info = manifest.set_index("clip_uid")[CLIP_COLS]
    f = frames.join(info, on="clip_uid")
    f["y"] = f["label"].map(STATION_INDEX).fillna(-1).astype(int)
    f["primary"] = (f["exclude_reason"] == "") & ~f["eval_exclude_b"].astype(bool)
    return f


def load_predset(path: Path, manifest: pd.DataFrame, with_emb: bool = False) -> PredSet:
    from training.landmarks.train import load_predictions, predictions_frame

    p = load_predictions(path)
    frames = attach_clip_info(predictions_frame(p), manifest)
    return PredSet(int(p["fold"]), frames, p["logits"].astype(np.float64), p.get("emb") if with_emb else None)


def load_experiment(exp_dir: Path, manifest: pd.DataFrame, kind: str = "oof",
                    with_emb: bool = False) -> dict[int, PredSet]:
    """{fold: PredSet} for runs/<exp>/fold*/{kind}_fold{k}.npz."""
    out = {}
    for k, d in fold_dirs(exp_dir).items():
        path = d / f"{kind}_fold{k}.npz"
        if path.exists():
            out[k] = load_predset(path, manifest, with_emb)
    return out


# -- quality ----------------------------------------------------------------------------------------------

def centre_rows(frames: pd.DataFrame) -> np.ndarray:
    """Index of the frame nearest u = 0.5 in each clip."""
    d = (frames["u"] - 0.5).abs()
    return d.groupby(frames["clip_uid"]).idxmin().to_numpy()


def sharp_ref_from_cache(manifest: pd.DataFrame | None = None, frames: pd.DataFrame | None = None) -> float:
    """Median centre-frame sharpness at 224 px over usable DEV clips (never test)."""
    manifest = D.load_manifest(MANIFEST_PATH) if manifest is None else manifest
    frames = D.load_frames(CACHE_ROOT) if frames is None else frames
    dev = manifest[(manifest["split"] == "dev") & (manifest["exclude_reason"] == "")]
    f = frames[frames["clip_uid"].isin(set(dev["clip_uid"])) & frames["layout_ok"]].reset_index(drop=True)
    return float(np.median(f.loc[centre_rows(f), "sharpness"]))


def quality_of(frames: pd.DataFrame, params: QualityParams) -> np.ndarray:
    return D.recompute_quality(frames, params)


def fit_q_min(frames: pd.DataFrame, params: QualityParams, pct: float = Q_PCT) -> float:
    """``pct``-th percentile of clip-centre frame quality (primary clips)."""
    f = frames[frames["primary"]].reset_index(drop=True) if "primary" in frames else frames
    q = quality_of(f.loc[centre_rows(f)], params)
    return float(np.percentile(q, pct))


# -- temperature ------------------------------------------------------------------------------------------

def nll(logits: np.ndarray, y: np.ndarray, t: float = 1.0) -> float:
    """Mean NLL; ``y`` is int labels or (n, C) soft targets."""
    lp = log_softmax(logits, t)
    if y.ndim == 1:
        return float(-lp[np.arange(len(y)), y].mean())
    return float(-(y * lp).sum(1).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    from scipy.optimize import minimize_scalar

    if len(logits) == 0:
        return 1.0
    res = minimize_scalar(lambda lt: nll(logits, y, float(np.exp(lt))),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded",
                          options={"xatol": 1e-4})
    return float(np.exp(res.x))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        s = bins == b
        if s.any():
            total += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


# -- thresholds ---------------------------------------------------------------------------------------------

def fit_tau(probs: np.ndarray, y: np.ndarray, precision: float = PRECISION_TARGET, floor: float = TAU_FLOOR,
            min_accept: int = MIN_ACCEPT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tau[C], reached[C], accepted[C]) at closed-set frame precision >= ``precision``."""
    top = probs.argmax(1)
    conf = probs.max(1)
    tau = np.ones(NUM_STATIONS)
    reached = np.zeros(NUM_STATIONS, bool)
    accepted = np.zeros(NUM_STATIONS, int)
    for c in range(NUM_STATIONS):
        sel = top == c
        if not sel.any():
            continue
        p = conf[sel]
        ok = (y[sel] == c).astype(float)
        order = np.argsort(-p, kind="stable")
        p, ok = p[order], ok[order]
        n = np.arange(1, len(p) + 1)
        prec = np.cumsum(ok) / n
        last_of_tie = np.r_[p[:-1] > p[1:], True]  # a threshold at p[i] also accepts its ties
        valid = (prec >= precision) & (n >= min_accept) & (p >= floor) & last_of_tie
        if valid.any():
            i = int(np.flatnonzero(valid).max())
            tau[c], reached[c], accepted[c] = float(p[i]), True, int(n[i])
    return tau, reached, accepted


def frame_status(probs: np.ndarray, quality: np.ndarray, tau: np.ndarray, q_min: float,
                 layout_ok: np.ndarray | None = None, gate: bool = True) -> np.ndarray:
    """ok | uncertain | low_quality per frame, exactly as RealLandmarkClassifier."""
    top = probs.argmax(1)
    conf = probs[np.arange(len(probs)), top]
    status = np.where(conf < np.asarray(tau)[top], "uncertain", "ok").astype(object)
    if gate:
        low = quality < q_min
        if layout_ok is not None:
            low |= ~np.asarray(layout_ok, bool)
        status[low] = "low_quality"
    return status


# -- cross-fitting --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Calib:
    T: float
    tau: tuple[float, ...]
    q_min: float
    reached: tuple[bool, ...] = ()
    accepted: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {"T": self.T, "tau": list(self.tau), "q_min": self.q_min, "tau_reached": list(self.reached),
                "tau_accepted": list(self.accepted)}


def temperature_frames(ps: PredSet, params: QualityParams, q_min: float) -> tuple[np.ndarray, np.ndarray]:
    f = ps.frames
    m = (f["primary"] & f["layout_ok"]).to_numpy() & (quality_of(f, params) >= q_min)
    return ps.logits[m], f["y"].to_numpy()[m]


def tau_frames(ps: PredSet, T: float, params: QualityParams, q_min: float,
               u_range: tuple[float, float] = TAU_U_RANGE) -> tuple[np.ndarray, np.ndarray]:
    f = ps.frames
    m = (f["primary"] & f["layout_ok"] & f["u"].between(*u_range)).to_numpy()
    m = m & (quality_of(f, params) >= q_min)
    return softmax(ps.logits[m], T), f["y"].to_numpy()[m]


def fit_thresholds(oof: dict[int, PredSet], use_folds, temps: dict[int, float], params: QualityParams,
                   precision: float = PRECISION_TARGET, q_pct: float = Q_PCT,
                   u_range: tuple[float, float] = TAU_U_RANGE) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """(tau, q_min, reached, accepted) from the OOF of ``use_folds`` only."""
    use = [k for k in use_folds if k in oof]
    q_min = fit_q_min(pd.concat([oof[k].frames for k in use], ignore_index=True), params, q_pct)
    probs, ys = zip(*(tau_frames(oof[k], temps[k], params, q_min, u_range) for k in use))
    tau, reached, accepted = fit_tau(np.concatenate(probs), np.concatenate(ys), precision)
    return tau, q_min, reached, accepted


def fold_temperatures(oof: dict[int, PredSet], inner: dict[int, PredSet], params: QualityParams,
                      q_min: float) -> dict[int, float]:
    """T_k from model k's inner-val frames (falls back to its OOF only if inner is missing)."""
    out = {}
    for k in oof:
        src = inner.get(k, oof[k])
        out[k] = fit_temperature(*temperature_frames(src, params, q_min))
    return out


def inner_fold_of(ps: PredSet) -> int | None:
    """The dev fold a model's inner-val predictions were made on (majority ``fold_i``), if known."""
    if "fold_i" not in ps.frames:
        return None
    folds = ps.frames["fold_i"].to_numpy(int)
    folds = folds[folds >= 0]
    return int(np.bincount(folds).argmax()) if len(folds) else None


def heldout_temperatures(oof: dict[int, PredSet], inner: dict[int, PredSet], params: QualityParams,
                         q_min: float, temps: dict[int, float] | None = None) -> dict[int, dict[int, float]]:
    """{k: {j: T_j}}: the temperatures to use when fold k is the reported (held-out) fold.

    T_j normally comes from model j's inner-val fold. For the model whose
    inner-val fold IS k, T_j is refit on its own outer fold j instead, so no
    temperature that feeds fold k's thresholds, tracker selection or
    auto_enabled has seen fold k's labels.
    """
    temps = fold_temperatures(oof, inner, params, q_min) if temps is None else temps
    out = {}
    for k in oof:
        tk = dict(temps)
        for j in oof:
            if j != k and j in inner and inner_fold_of(inner[j]) == k:
                tk[j] = fit_temperature(*temperature_frames(oof[j], params, q_min))
        out[k] = tk
    return out


def crossfit(oof: dict[int, PredSet], inner: dict[int, PredSet], params: QualityParams,
             precision: float = PRECISION_TARGET, q_pct: float = Q_PCT,
             temps_heldout: dict[int, dict[int, float]] | None = None) -> dict[int, Calib]:
    """Per outer fold k: T_k (inner-val of model k), tau_k and q_min_k from folds != k
    (calibrated with ``heldout_temperatures``, i.e. never with a T fit on fold k)."""
    if temps_heldout is None:
        temps_heldout = heldout_temperatures(oof, inner, params, crossfit_q_min(oof, params, q_pct))
    out = {}
    for k in oof:
        tau, q_min, reached, acc = fit_thresholds(oof, [j for j in oof if j != k], temps_heldout[k], params,
                                                  precision, q_pct)
        out[k] = Calib(temps_heldout[k][k], tuple(tau), q_min, tuple(reached), tuple(acc))
    return out


def crossfit_q_min(oof: dict[int, PredSet], params: QualityParams, q_pct: float = Q_PCT) -> float:
    """The quality gate used to select temperature frames (pooled OOF; quality needs no labels)."""
    return fit_q_min(pd.concat([p.frames for p in oof.values()], ignore_index=True), params, q_pct)


def shipped(oof: dict[int, PredSet], params: QualityParams, precision: float = PRECISION_TARGET,
            q_pct: float = Q_PCT) -> Calib:
    """Values for the bundle: one T on pooled dev OOF, tau and q_min on all folds."""
    frames = pd.concat([p.frames for p in oof.values()], ignore_index=True)
    q_min = fit_q_min(frames, params, q_pct)
    pooled = PredSet(-1, frames, np.concatenate([p.logits for p in oof.values()]))
    T = fit_temperature(*temperature_frames(pooled, params, q_min))
    tau, _, reached, acc = fit_thresholds({-1: pooled}, [-1], {-1: T}, params, precision, q_pct)
    return Calib(T, tuple(tau), q_min, tuple(reached), tuple(acc))


def calibration_report(oof: dict[int, PredSet], cal: dict[int, Calib], params: QualityParams) -> dict:
    """ECE / NLL / Brier on quality-passing u in [0.1, 0.9] frames, before and after cross-fit T."""
    rows = {"raw": [], "cal": []}
    ys = []
    for k, ps in oof.items():
        f = ps.frames
        m = (f["primary"] & f["layout_ok"] & f["u"].between(*TAU_U_RANGE)).to_numpy()
        m = m & (quality_of(f, params) >= cal[k].q_min)
        rows["raw"].append(softmax(ps.logits[m]))
        rows["cal"].append(softmax(ps.logits[m], cal[k].T))
        ys.append(f["y"].to_numpy()[m])
    y = np.concatenate(ys)
    out = {"n_frames": int(len(y))}
    for name, parts in rows.items():
        p = np.concatenate(parts)
        out[name] = {"ece": ece(p, y), "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                     "brier": brier(p, y)}
    return out


def calibrate_experiment(exp_dir: Path, manifest: pd.DataFrame | None = None, q_pct: float = Q_PCT,
                         precision: float = PRECISION_TARGET) -> dict:
    manifest = D.load_manifest() if manifest is None else manifest
    oof = load_experiment(exp_dir, manifest, "oof")
    inner = load_experiment(exp_dir, manifest, "inner")
    if not oof:
        raise SystemExit(f"no OOF predictions under {exp_dir}")
    params = replace(QualityParams(), sharp_ref=sharp_ref_from_cache(manifest))
    held = heldout_temperatures(oof, inner, params, crossfit_q_min(oof, params, q_pct))
    cal = crossfit(oof, inner, params, precision, q_pct, held)
    ship = shipped(oof, params, precision, q_pct)
    frames = pd.concat([p.frames for p in oof.values()], ignore_index=True)
    pooled = {-1: PredSet(-1, frames, np.concatenate([p.logits for p in oof.values()]))}
    sens = {name: fit_thresholds(pooled, [-1], {-1: ship.T}, params, precision, q_pct, rng)[0].tolist()
            for name, rng in (("u_all", (0.0, 1.0)), ("u_0.2_0.8", (0.2, 0.8)))}
    grid = {str(prec): fit_thresholds(pooled, [-1], {-1: ship.T}, params, prec, q_pct)[0].tolist()
            for prec in (0.85, 0.90, 0.95)}
    out = {
        "quality_params": params.to_dict(),
        "sharp_ref_source": "median centre-frame sharpness at 224 px, usable dev clips",
        "precision_target": precision, "q_pct": q_pct, "tau_u_range": list(TAU_U_RANGE),
        "tau_floor": TAU_FLOOR, "min_accept": MIN_ACCEPT,
        "per_fold": {str(k): c.to_dict() for k, c in cal.items()},
        "temps_heldout": {str(k): {str(j): t for j, t in tk.items()} for k, tk in held.items()},
        "shipped": ship.to_dict(),
        "sensitivity_tau": sens, "tau_by_precision": grid,
        "stations": list(STATION_ORDER),
        "report": calibration_report(oof, cal, params),
    }
    write_json(Path(exp_dir) / "calibration.json", out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--q-pct", type=float, default=Q_PCT)
    ap.add_argument("--precision", type=float, default=PRECISION_TARGET)
    a = ap.parse_args(argv)
    out = calibrate_experiment(a.exp, q_pct=a.q_pct, precision=a.precision)
    s = out["shipped"]
    print(f"T={s['T']:.3f} q_min={s['q_min']:.3f} sharp_ref={out['quality_params']['sharp_ref']:.1f}")
    for k, t, r in zip(STATION_ORDER, s["tau"], s["tau_reached"]):
        print(f"  {k:28s} tau={t:.3f}{'' if r else '  (precision target not reached)'}")
    print(f"ECE raw {out['report']['raw']['ece']:.3f} -> cross-fit {out['report']['cal']['ece']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
