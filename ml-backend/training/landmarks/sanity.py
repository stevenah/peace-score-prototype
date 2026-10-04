"""Sanity and shortcut checks (G-PIPELINE, G-SHORTCUT) and the serving blank-frame check.

    python -m training.landmarks.sanity perm --config sanity_perm              # GPU: label-permutation run
    python -m training.landmarks.sanity shortcuts --exp runs/<exp>             # NBI / mode-switch / cap effects
    python -m training.landmarks.sanity unmasked-cache                         # laptop: positive-control canvases
    python -m training.landmarks.train --config sanity_unmasked --fold 0       # GPU: the unmasked model
    python -m training.landmarks.sanity positive-control --run runs/sanity_unmasked_s0/fold0 --exp runs/<exp>
    python -m training.landmarks.sanity blank --bundle <bundle dir>            # blank interior -> low_quality

Results are merged into ``runs/<exp>/sanity.json`` (read by evaluate.py --report).

- perm: fold 0 trained on clip labels shuffled among the training clips; clip
  macro-F1 must be <= 0.15 (a pipeline-bug test, not a leakage test).
- nbi: P(oesophageal region) on NBI minus WL frames of the same TRUE
  non-oesophageal station, pooled with weights min(n_wl, n_nbi) clips;
  plus WL vs NBI accuracy of the oesophageal stations.
- mode_switch: paired within-clip difference (NBI minus WL frames) on
  mixed-mode clips.
- cap: P(z_line) on frames of cap vs no-cap clips of the same non-Z-line station.
- positive control: a model trained on UNMASKED canvases (the badge in the
  aperture's corner cut is visible) must trip at least one check, else the
  checks are not sensitive enough to trust; plus a badge-paste
  counterfactual (paste an NBI badge into WL frames).
- blank: a frame that passes layout detection but has a constant interior
  must come out ``low_quality`` from the real serving classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.ml.landmarks.layouts import CANON_SIZE, OLYMPUS_1920, Layout, canvas_geometry
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER, STATION_REGION
from training.landmarks import dataset as D
from training.landmarks.calibrate import PredSet, load_experiment, load_predset, quality_of
from training.landmarks.common import DATA_ROOT, ML_BACKEND, decode_frames, is_full_length, write_csv
from training.landmarks.run_utils import (
    RUNS_ROOT,
    cluster_bootstrap,
    fp32_inference,
    macro_f1,
    read_json,
    softmax,
    write_json,
)

OES = np.array([STATION_REGION[k] == "esophagus" for k in STATION_ORDER])
Z = STATION_INDEX["z_line"]
UNMASKED_ROOT = ML_BACKEND / "training" / "landmarks" / "cache" / "v1_unmasked"
LIMITS = {"nbi": 0.05, "mode_switch": 0.05, "cap": 0.05, "badge_paste": 0.05}


def merge_sanity(exp_dir: Path, key: str, value: dict) -> None:
    path = Path(exp_dir) / "sanity.json"
    cur = read_json(path) if path.exists() else {}
    cur[key] = value
    write_json(path, cur)


# -- frame tables ------------------------------------------------------------------------------------------

def frame_probs(oof: dict[int, PredSet], temps: dict[int, float] | None = None, q_min: float = 0.0,
                params=None) -> pd.DataFrame:
    """Quality-passing frames of primary clips with p(oesophageal), p(z_line), p(true)."""
    parts = []
    for k, ps in oof.items():
        f = ps.frames
        keep = (f["primary"] & f["layout_ok"]).to_numpy()
        if params is not None:
            keep = keep & (quality_of(f, params) >= q_min)
        p = softmax(ps.logits[keep], (temps or {}).get(k, 1.0))
        sub = f[keep].reset_index(drop=True)
        parts.append(pd.DataFrame({
            "clip_uid": sub["clip_uid"], "patient_id": sub["patient_id"], "y": sub["y"],
            "nbi": sub["mode_frame"].isin(["nbi", "rdi"]).to_numpy(), "cap": sub["cap_b"].to_numpy(bool),
            "p_oes": p[:, OES].sum(1), "p_z": p[:, Z], "p_true": p[np.arange(len(p)), sub["y"].to_numpy()],
            "pred": p.argmax(1)}))
    return pd.concat(parts, ignore_index=True)


def _stratified_diff(fp: pd.DataFrame, flag: str, value: str, stations: list[int]) -> tuple[float, dict]:
    """Weighted mean over stations of mean(value | flag) - mean(value | not flag); clip-level means."""
    clip = fp.groupby(["clip_uid", flag]).agg(v=(value, "mean"), y=("y", "first"), pat=("patient_id", "first"))
    clip = clip.reset_index()
    diffs, weights, per = [], [], {}
    for s in stations:
        a = clip[(clip["y"] == s) & clip[flag]]
        b = clip[(clip["y"] == s) & ~clip[flag]]
        if len(a) and len(b):
            d = float(a["v"].mean() - b["v"].mean())
            w = min(len(a), len(b))
            diffs.append(d)
            weights.append(w)
            per[STATION_ORDER[s]] = {"diff": d, "n_flag": int(len(a)), "n_other": int(len(b))}
    eff = float(np.average(diffs, weights=weights)) if diffs else float("nan")
    return eff, per


def _effect_ci(fp: pd.DataFrame, flag: str, value: str, stations: list[int]) -> dict:
    pats = fp["patient_id"].unique()
    groups = fp["patient_id"].to_numpy()

    def stat(idx):
        return _stratified_diff(fp.iloc[idx], flag, value, stations)[0]

    eff, per = _stratified_diff(fp, flag, value, stations)
    _, lo, hi = cluster_bootstrap(stat, groups, n_boot=300) if len(pats) > 3 else (eff, np.nan, np.nan)
    return {"effect": eff, "lo": lo, "hi": hi, "per_station": per}


def nbi_shortcut(fp: pd.DataFrame) -> dict:
    non_oes = [i for i in range(NUM_STATIONS) if not OES[i]]
    out = _effect_ci(fp, "nbi", "p_oes", non_oes)
    oes = fp[OES[fp["y"].to_numpy()]]
    acc = {}
    for s in np.flatnonzero(OES):
        for mode, sel in (("wl", ~oes["nbi"]), ("nbi", oes["nbi"])):
            f = oes[sel & (oes["y"] == s)]
            acc.setdefault(STATION_ORDER[s], {})[mode] = {
                "n_frames": int(len(f)), "n_clips": int(f["clip_uid"].nunique()),
                "accuracy": float((f["pred"] == s).mean()) if len(f) else None}
    wl = oes[~oes["nbi"]]
    out["wl_oesophageal_region_recall"] = float(OES[wl["pred"].to_numpy()].mean()) if len(wl) else None
    out["oesophagus_by_mode"] = acc
    return out


def mode_switch(fp: pd.DataFrame, min_frames: int = 3) -> dict:
    """Paired within-clip NBI-minus-WL differences on mixed clips."""
    rows = []
    for uid, g in fp.groupby("clip_uid"):
        a, b = g[g["nbi"]], g[~g["nbi"]]
        if len(a) >= min_frames and len(b) >= min_frames:
            rows.append({"patient_id": g["patient_id"].iloc[0], "y": int(g["y"].iloc[0]),
                         "d_oes": a["p_oes"].mean() - b["p_oes"].mean(),
                         "d_true": a["p_true"].mean() - b["p_true"].mean()})
    if not rows:
        return {"effect": None, "n_clips": 0}
    r = pd.DataFrame(rows)
    non = r[~OES[r["y"].to_numpy()]]
    out = {"n_clips": int(len(r)), "n_non_oesophageal": int(len(non)),
           "effect": float(non["d_oes"].mean()) if len(non) else None,
           "d_true_mean": float(r["d_true"].mean())}
    if len(non) > 3:
        v = non["d_oes"].to_numpy()
        _, out["lo"], out["hi"] = cluster_bootstrap(lambda i: float(v[i].mean()), non["patient_id"].to_numpy(),
                                                    n_boot=500)
    return out


def cap_shortcut(fp: pd.DataFrame) -> dict:
    return _effect_ci(fp, "cap", "p_z", [i for i in range(NUM_STATIONS) if i != Z])


def shortcuts(exp_dir: Path, manifest: pd.DataFrame | None = None) -> dict:
    from training.landmarks.stream_eval import load_calibration

    manifest = D.load_manifest() if manifest is None else manifest
    oof = load_experiment(exp_dir, manifest, "oof")
    cal, params, _ = load_calibration(exp_dir)
    q_min = float(np.median([c.q_min for c in cal.values()]))
    fp = frame_probs(oof, {k: c.T for k, c in cal.items()}, q_min, params)
    return {"nbi": nbi_shortcut(fp), "mode_switch": mode_switch(fp), "cap": cap_shortcut(fp)}


# -- G-PIPELINE ------------------------------------------------------------------------------------------------

def perm_run(config: str = "sanity_perm", device: str = "auto", exp: str | None = None) -> dict:
    from training.landmarks.train import resolve_config, train

    cfg = resolve_config(config)
    out = RUNS_ROOT / (exp or f"{cfg['name']}_s{cfg['seed']}") / "fold0"
    met = train(cfg, "fold", out, fold=0, device=device)
    m = D.load_manifest()
    ps = load_predset(out / "oof_fold0.npz", m)
    f = ps.frames
    keep = (f["primary"] & f["u"].between(0.2, 0.8)).to_numpy()
    lp = pd.DataFrame(ps.logits[keep])
    lp["clip_uid"] = f.loc[keep, "clip_uid"].to_numpy()
    clip = lp.groupby("clip_uid").mean()
    y = f.drop_duplicates("clip_uid").set_index("clip_uid").loc[clip.index, "y"].to_numpy()
    f1 = macro_f1(y, clip.to_numpy().argmax(1))
    return {"clip_macro_f1": f1, "n_clips": int(len(clip)), "run": str(out.relative_to(ML_BACKEND)),
            "steps": met["steps_run"], "pass": f1 <= 0.15}


# -- positive control --------------------------------------------------------------------------------------------

def canonicalize_unmasked(full_rgb: np.ndarray, layout: Layout) -> np.ndarray:
    """``preprocess.canonicalize`` WITHOUT the aperture mask (positive control only)."""
    s = layout.scale(full_rgb.shape[1])
    x0, y0, x1, y1 = (int(round(v * s)) for v in layout.roi)
    _, h, pad_top = canvas_geometry(layout)
    small = cv2.resize(full_rgb[y0:y1, x0:x1], (CANON_SIZE, h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANON_SIZE, CANON_SIZE, 3), np.uint8)
    canvas[pad_top:pad_top + h] = small
    return canvas


def badge_canvas_box(layout: Layout = OLYMPUS_1920, pad: int = 2) -> tuple[int, int, int, int]:
    scale, _, pad_top = canvas_geometry(layout)
    bx0, by0, bx1, by1 = layout.badge_box
    x0, y0 = layout.roi[0], layout.roi[1]
    return (max(0, int((bx0 - x0) * scale) - pad), max(0, int((by0 - y0) * scale + pad_top) - pad),
            min(CANON_SIZE, int(np.ceil((bx1 - x0) * scale)) + pad),
            min(CANON_SIZE, int(np.ceil((by1 - y0) * scale + pad_top)) + pad))


def build_unmasked_cache(patients: list[str] | None = None, every: int = 15, root: Path = UNMASKED_ROOT) -> dict:
    """Unmasked canvases for dev clips at 2 fps (every 15th source frame = every 5th cached frame).

    Laptop only (reads the raw clips); the canvases stay in the gitignored cache.
    """
    from concurrent.futures import ProcessPoolExecutor

    m = D.load_manifest()
    frames = D.load_frames()
    clips = m[(m["split"] == "dev") & (m["exclude_reason"] == "") & (m["cohort"] == "new_1920")]
    if patients:
        clips = clips[clips["patient_id"].isin(set(patients))]
    jobs = [(r.clip_uid, r.rel_path, int(r.width), int(r.height), every, str(root)) for r in clips.itertuples()
            if not is_full_length(r.rel_path)]
    with ProcessPoolExecutor(6) as ex:
        done = list(ex.map(_unmasked_clip, jobs))
    keys = {(u, s) for u, idx in done for s in idx}
    sub = frames[[(u, s) in keys for u, s in zip(frames["clip_uid"], frames["src_idx"])]]
    write_csv(root / "frames.csv.gz", sub.to_dict("records"), list(frames.columns))
    info = {"what": "UNMASKED canvases for the G-SHORTCUT positive control; never train a real model on these",
            "every_nth_source_frame": every, "clips": len(done), "frames": int(len(sub)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (root / "CACHE_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _unmasked_clip(job) -> tuple[str, list[int]]:
    uid, rel, w, h, every, root = job
    out = Path(root) / "frames" / uid
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    enc = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    for src, rgb in decode_frames(DATA_ROOT / rel, w, h, every):
        c = canonicalize_unmasked(rgb, OLYMPUS_1920)
        cv2.imwrite(str(out / f"{src:05d}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR), enc)
        kept.append(src)
    return uid, kept


def unmasked_hook(net, device: str = "cpu", size: int = 224, batch: int = 64):
    import torch

    from app.ml.landmarks.preprocess import normalize

    def hook(canvases):
        out = []
        with torch.inference_mode(), fp32_inference():
            for s in range(0, len(canvases), batch):
                x = torch.stack([normalize(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA))
                                 for c in canvases[s:s + batch]])
                out.append(net(x.to(device).float())[0].float().cpu().numpy())
        return np.concatenate(out)

    return hook


def badge_paste_effect(hook, ps: PredSet, cache_root: Path, n: int = 300, seed: int = 0) -> dict:
    """Paste the badge region of NBI canvases into WL canvases of non-oesophageal stations."""
    f = ps.frames
    rng = np.random.default_rng(seed)
    nbi_rows = np.flatnonzero(f["mode_frame"].isin(["nbi"]).to_numpy())
    wl_rows = np.flatnonzero((f["mode_frame"] == "wl").to_numpy() & ~OES[f["y"].clip(0).to_numpy()]
                             & f["primary"].to_numpy())
    if not len(nbi_rows) or not len(wl_rows):
        return {"effect": None, "n": 0}
    x0, y0, x1, y1 = badge_canvas_box()
    wl_pick = rng.choice(wl_rows, min(n, len(wl_rows)), replace=False)
    src_pick = rng.choice(nbi_rows, len(wl_pick), replace=True)
    base, pasted = [], []
    for a, b in zip(wl_pick, src_pick):
        c = D.read_canvas(cache_root, f.at[a, "clip_uid"], int(f.at[a, "src_idx"]))
        badge = D.read_canvas(cache_root, f.at[b, "clip_uid"], int(f.at[b, "src_idx"]))[y0:y1, x0:x1]
        p = c.copy()
        p[y0:y1, x0:x1] = badge
        base.append(c)
        pasted.append(p)
    d = softmax(hook(pasted))[:, OES].sum(1) - softmax(hook(base))[:, OES].sum(1)
    return {"effect": float(d.mean()), "n": int(len(d)), "box": [x0, y0, x1, y1]}


def positive_control(run_dir: Path, device: str = "cpu") -> dict:
    import torch

    from app.ml.landmarks.architectures import build
    from training.landmarks.backbones import get_candidate

    cfg = read_json(run_dir / "config.json")
    met = read_json(run_dir / "metrics.json")
    if cfg["input"].get("masked", True):
        raise SystemExit("positive control needs a run trained with input.masked=false")
    m = D.load_manifest()
    ps = load_predset(run_dir / "oof_fold0.npz", m)
    fp = frame_probs({0: ps})
    res = {"nbi": nbi_shortcut(fp), "mode_switch": mode_switch(fp), "cap": cap_shortcut(fp)}
    cand = cfg["candidate"]
    net = build(cand.split(":", 1)[1] if cand.startswith("scratch:") else get_candidate(cand).arch)
    net.load_state_dict(torch.load(run_dir / f"best_{met['best']['variant']}.pt", map_location="cpu",
                                   weights_only=True))
    root = Path(cfg["input"]["cache_root"])
    root = root if root.is_absolute() else ML_BACKEND / root
    res["badge_paste"] = badge_paste_effect(unmasked_hook(net.to(device).eval(), device, cfg["input"]["size"]),
                                            ps, root)
    effects = {"nbi": res["nbi"]["effect"], "mode_switch": res["mode_switch"].get("effect"),
               "cap": res["cap"]["effect"], "badge_paste": res["badge_paste"].get("effect")}
    res["tripped_checks"] = [k for k, v in effects.items() if v is not None and np.isfinite(v)
                             and abs(v) > LIMITS[k]]
    res["tripped"] = bool(res["tripped_checks"])
    res["run"] = str(run_dir)
    return res


# -- serving system check ------------------------------------------------------------------------------------------

def blank_frames() -> dict[str, np.ndarray]:
    """Frames that pass layout detection but whose aperture interior is constant."""
    from app.ml.landmarks.bench import synthetic_olympus_frame

    base = synthetic_olympus_frame(1920, seed=0)
    inside = np.zeros(base.shape[:2], np.uint8)
    cv2.fillPoly(inside, [np.array(OLYMPUS_1920.polygon, np.int32)], 1)
    out = {}
    for name, rgb in (("grey", (128, 128, 128)), ("pink", (190, 95, 70)), ("dark", (12, 8, 8)),
                      ("white", (250, 250, 250))):
        f = base.copy()
        f[inside > 0] = rgb
        out[name] = f
    return out


def blank_check(bundle_dir: Path) -> dict:
    """System check: a blank interior must yield low_quality from RealLandmarkClassifier."""
    from app.ml.landmarks.bench import synthetic_olympus_frame
    from app.ml.landmarks.bundle import load_bundle
    from app.ml.landmarks.classifier import LandmarkModelManager, RealLandmarkClassifier

    clf = RealLandmarkClassifier(LandmarkModelManager(load_bundle(bundle_dir, "cpu")))
    for i in range(3):  # lock the layout on normal frames first
        clf.predict(synthetic_olympus_frame(1920, seed=10 + i))
    statuses = {name: clf.predict(f).status for name, f in blank_frames().items()}
    return {"statuses": statuses, "pass": all(s == "low_quality" for s in statuses.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("perm")
    p.add_argument("--config", default="sanity_perm")
    p.add_argument("--exp-out", help="experiment whose sanity.json receives the result")
    p.add_argument("--device", default="auto")
    s = sub.add_parser("shortcuts")
    s.add_argument("--exp", type=Path, required=True)
    sub.add_parser("unmasked-cache")
    c = sub.add_parser("positive-control")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--exp", type=Path, required=True)
    c.add_argument("--device", default="cpu")
    b = sub.add_parser("blank")
    b.add_argument("--bundle", type=Path, required=True)
    b.add_argument("--exp", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "perm":
        res = perm_run(a.config, a.device)
        if a.exp_out:
            merge_sanity(Path(a.exp_out), "perm", res)
    elif a.cmd == "shortcuts":
        res = shortcuts(a.exp)
        for k, v in res.items():
            merge_sanity(a.exp, k, v)
    elif a.cmd == "unmasked-cache":
        res = build_unmasked_cache()
    elif a.cmd == "positive-control":
        res = positive_control(a.run, a.device)
        merge_sanity(a.exp, "positive_control", res)
    else:
        res = blank_check(a.bundle)
        if a.exp:
            merge_sanity(a.exp, "blank", res)
    print(json.dumps(res, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
