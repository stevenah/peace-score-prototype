"""Score an exported bundle on labelled clips it never trained on.

    python -m training.landmarks.bundle_eval --bundle runs/<exp>/fold0/bundle/lm-0.1.0 \\
        [--split heldout|shift] [--device auto] [--out result.json]

The bundle is verified and loaded like a served one (``load_bundle``) and run
over every cached frame of the split's primary clips. Clip, patient and frame
metrics are evaluate.py's, computed with the bundle's own temperature, quality
parameters, q_min and tau, so they describe what the bundle would serve.

Splits:

    heldout   the outer dev fold a fold (or smoke) run held out. Refused for
              an all-dev model, which has trained on every dev patient.
    shift     the legacy-cohort shift set, quality gate off as in
              evaluate.py's ``legacy_shift``. Another processor layout: a
              robustness figure, not serving accuracy.

The locked test set is not an option here. It is scored once, by
``evaluate.py --test``.

Only aggregate numbers leave this module: no clip, patient or path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.landmarks.bundle import LandmarkBundle, load_bundle
from app.ml.landmarks.layouts import LAYOUTS_BY_NAME
from app.ml.landmarks.progress import report
from app.ml.landmarks.registry import training_split
from app.ml.landmarks.stations import STATION_ORDER
from training.landmarks import dataset as D
from training.landmarks.calibrate import Calib, PredSet, attach_clip_info, frame_status, quality_of
from training.landmarks.common import CACHE_ROOT, MANIFEST_PATH, sha256_file
from training.landmarks.evaluate import clip_metrics, clip_predictions, frame_metrics, patient_level
from training.landmarks.run_utils import cache_fingerprint, resolve_device, softmax, write_json

SPLITS = ("heldout", "shift")


def select_clips(m: pd.DataFrame, split: str, meta: dict, bundle_dir: Path) -> tuple[pd.DataFrame, dict]:
    """(primary clips of the split, a description of it). Exits when the split is not clean."""
    primary = (m["exclude_reason"] == "") & ~m["eval_exclude_b"]
    if split == "shift":
        return m[primary & (m["split"] == "shift")].reset_index(drop=True), {
            "name": "shift", "label": "legacy shift set", "fold": None}
    run = training_split(meta, bundle_dir)
    if run["mode"] == "all_dev":
        raise SystemExit("this bundle was trained on every dev patient, so no dev fold is held out; "
                         "score it on the shift set, or on the locked test set with evaluate.py --test")
    if run["mode"] not in ("fold", "smoke") or run["fold"] is None:
        raise SystemExit("cannot tell which dev fold this bundle held out: meta.train names no fold run")
    _, _, outer = D.fold_patients(m, run["fold"])
    clips = m[primary & (m["split"] == "dev") & m["patient_id"].isin(set(outer))].reset_index(drop=True)
    return clips, {"name": "heldout", "label": f"dev fold {run['fold']} (held out)", "fold": run["fold"]}


def acceptance(ps: PredSet, bundle: LandmarkBundle, q_min: float) -> dict:
    """How the served thresholds sort the frames: ok / uncertain / low_quality.

    ``precision_ok`` is closed-set (every frame shows one of the 10 stations),
    so precision on a live stream will be lower.
    """
    f = ps.frames
    prim = f["primary"].to_numpy()
    probs = softmax(ps.logits[prim], bundle.temperature)
    y = f["y"].to_numpy()[prim]
    status = frame_status(probs, quality_of(f, bundle.quality)[prim], np.asarray(bundle.tau), q_min,
                          f["layout_ok"].to_numpy()[prim])
    top = probs.argmax(1)
    ok = status == "ok"
    per = {}
    for c, key in enumerate(STATION_ORDER):
        sel = ok & (top == c)
        per[key] = {"accepted": int(sel.sum()), "precision": float((y[sel] == c).mean()) if sel.any() else None}
    return {
        "n_frames": int(len(y)),
        "share": {s: float((status == s).mean()) for s in ("ok", "uncertain", "low_quality")},
        "precision_ok": float((top[ok] == y[ok]).mean()) if ok.any() else None,
        "per_station": per,
    }


def data_warnings(meta: dict, manifest_path: Path, cache_root: Path, split: str) -> list[str]:
    train = meta.get("train") if isinstance(meta.get("train"), dict) else {}
    out = []
    trained_on = train.get("manifest_sha256")
    if split == "heldout" and trained_on and trained_on != sha256_file(manifest_path):
        out.append("The manifest has changed since this model was trained, so its fold may no longer be held out.")
    clip_set = train.get("clip_set_sha256")
    if clip_set and clip_set != cache_fingerprint(cache_root)["clip_set_sha256"]:
        out.append("The frame cache holds a different clip set from the one this model was trained on.")
    if isinstance(meta.get("metrics"), dict) and meta["metrics"].get("WARNING"):
        out.append("Uncalibrated export: temperature, tau and q_min are defaults, not fitted values.")
    return out


def evaluate_bundle(bundle_dir: str | Path, split: str = "heldout", device: str = "auto",
                    manifest_path: Path = MANIFEST_PATH, cache_root: Path = CACHE_ROOT,
                    batch: int = 128, workers: int = 4, progress=report) -> dict:
    from training.landmarks.train import predict_table

    t0 = time.perf_counter()
    bundle_dir = Path(bundle_dir)
    device = resolve_device(device)
    bundle = load_bundle(bundle_dir, device=device)
    m = D.load_manifest(manifest_path)
    clips, described = select_clips(m, split, bundle.meta, bundle_dir)
    known = clips["layout"].isin(set(LAYOUTS_BY_NAME))
    skipped, clips = int((~known).sum()), clips[known].reset_index(drop=True)
    if clips.empty:
        raise SystemExit(f"no primary clips in the {described['label']}")

    table = D.eval_frame_table(D.load_frames(cache_root), clips)
    progress(0, len(table), "frames")
    logits, _ = predict_table(bundle.model, table, D.ImageSpec(bundle.input_size, cache_root=cache_root), device,
                              batch=batch, workers=workers, with_emb=False,
                              on_batch=lambda done: progress(done, len(table), "frames"))
    ps = PredSet(-1 if described["fold"] is None else described["fold"], attach_clip_info(table, m),
                 logits.astype(np.float64))

    # The shift set is another processor: its sharpness scale is not the one q_min was fitted on.
    q_min = 0.0 if split == "shift" else bundle.q_min
    cp = clip_predictions(ps, bundle.temperature, bundle.quality, q_min)
    try:
        frame = frame_metrics({ps.fold: ps}, {ps.fold: Calib(bundle.temperature, bundle.tau, q_min)}, bundle.quality)
    except (ValueError, IndexError, ZeroDivisionError):  # no frame passes the quality gate
        frame = None
    return {
        "version": bundle.version,
        "arch": bundle.arch,
        "input_size": bundle.input_size,
        "device": device,
        "split": {**described, "n_clips": int(len(cp)), "n_patients": int(cp["patient_id"].nunique()),
                  "n_frames": int(len(table)), "skipped_clips": skipped},
        "calibration": {"temperature": bundle.temperature, "q_min": q_min, "quality_gate": split != "shift"},
        "stations": list(STATION_ORDER),
        "clip": clip_metrics(cp),
        "patient_level": patient_level(cp),
        "frame": frame,
        "acceptance": acceptance(ps, bundle, q_min),
        "warnings": data_warnings(bundle.meta, manifest_path, cache_root, split),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--split", choices=SPLITS, default="heldout")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, help="write the result JSON here")
    a = ap.parse_args(argv)
    res = evaluate_bundle(a.bundle, a.split, a.device)
    f1 = res["clip"]["macro_f1"]
    print(f"{res['version']} on the {res['split']['label']}: clip macro-F1 {f1['value']:.3f} "
          f"[{f1['lo']:.3f}, {f1['hi']:.3f}], {res['split']['n_clips']} clips, "
          f"{res['split']['n_patients']} patients, {res['seconds']} s on {res['device']}")
    for w in res["warnings"]:
        print(f"warning: {w}")
    if a.out:
        write_json(a.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
