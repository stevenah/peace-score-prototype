"""Export a trained landmark model as a serving bundle ``lm-<semver>/`` (M4).

    python -m training.landmarks.export_bundle --exp runs/<exp> --version lm-1.0.0
    python -m training.landmarks.export_bundle --run runs/smoke_x/smoke --version lm-0.0.1-smoke --smoke

Builds the bundle with the backend's own ``bundle.write_bundle`` (so the
format is the one ``load_bundle`` verifies), from:

    weights      runs/<exp>/all_dev/final_<variant>.pt (or --run <dir>, best_<variant>.pt)
    T, tau       tracker.json "shipped_calibration" (the tuned precision target and
                 q_min percentile) or calibration.json "shipped"
    quality      QualityParams with sharp_ref = calibration.json (median centre-frame
                 sharpness of dev clips at 224 px) + q_min
    tracker      tracker.json "shipped" (default TrackerParams otherwise)
    auto_enabled gates_result.json (G-STATION); required unless --smoke
    metrics      aggregate dev OOF / test summaries; train = provenance

``write_bundle`` computes the CPU fp32 selftest from the reloaded weights;
the bundle is then re-verified with ``load_bundle`` on CPU and exercised
once through ``RealLandmarkClassifier`` on a synthetic processor frame.
Finally the directory is tarred (``pack_bundle``) and the sha256 printed
with the ``aws s3 cp`` command for the PRIVATE bucket and the
``BUNDLE.lock`` JSON to commit. Nothing is uploaded or committed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.ml.landmarks.bundle import BundleLock, load_bundle, pack_bundle, write_bundle
from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, STATION_ORDER
from app.ml.landmarks.tracker import TrackerParams
from training.landmarks import dataset as D
from training.landmarks.run_utils import read_json

S3_PREFIX = "models/landmarks"
LOCK_PATH = "ml-backend/app/models/landmarks/BUNDLE.lock"


def _opt(path: Path):
    return read_json(path) if path.exists() else None


def load_run_model(run_dir: Path, variant: str | None = None):
    """(model, config, variant) from a train.py run directory (all_dev, fold or smoke)."""
    import torch

    from app.ml.landmarks.architectures import build
    from training.landmarks.backbones import get_candidate

    cfg = read_json(run_dir / "config.json")
    met = read_json(run_dir / "metrics.json")
    variant = variant or met["best"]["variant"] or "ema"
    cand = cfg["candidate"]
    arch = cand.split(":", 1)[1] if cand.startswith("scratch:") else get_candidate(cand).arch
    weights = run_dir / f"final_{variant}.pt"
    if not weights.exists():
        weights = run_dir / f"best_{variant}.pt"
    net = build(arch)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval(), cfg, variant, weights


def _metrics_summary(exp_dir: Path | None) -> dict:
    if exp_dir is None:
        return {}
    out = {}
    ev = _opt(exp_dir / "eval_oof.json")
    if ev:
        out["dev_oof"] = {"clip_macro_f1": ev["clip"]["macro_f1"],
                          "clip_merged_macro_f1": ev["clip_merged"]["macro_f1"],
                          "patient_macro_f1": ev["patient_level"]["macro_f1"],
                          "ece": ev["frame"]["calibration"]["ece"],
                          "n_clips": ev["clip"]["n_clips"], "n_patients": ev["clip"]["n_patients"],
                          "per_station_recall": {k: v["recall"] for k, v in ev["clip"]["per_class"].items()},
                          "per_station_precision": {k: v["precision"] for k, v in ev["clip"]["per_class"].items()}}
    tr = _opt(exp_dir / "tracker.json")
    if tr:
        out["replay_dev_nested"] = {"S2_recall_mean": tr["heldout_pooled"]["S2"]["recall_mean"],
                                    "S4_fdr_overall": tr["heldout_pooled"]["S4"].get("fdr_overall"),
                                    "tto_median_s": tr["heldout_pooled"]["S2"]["tto_median"],
                                    "transit_source": tr.get("transit_source")}
    te = _opt(exp_dir / "eval_test.json")
    if te:
        out["test"] = {"clip_macro_f1": te["clip"]["macro_f1"], "G-TEST": te["G-TEST"],
                       "primary_non_inferiority": te["primary_non_inferiority"],
                       "test_patients_sha256": te["test_patients_sha256"]}
    gr = _opt(exp_dir / "gates_result.json")
    if gr:
        out["gates"] = {k: v["status"] for k, v in gr["gates"].items()}
    return out


def build_meta(run_dir: Path, exp_dir: Path | None, version: str, cfg: dict, variant: str, weights: Path,
               smoke: bool = False, manifest=None, sharp_ref: float | None = None) -> dict:
    """meta_fields for ``write_bundle`` (derived fields are left to it)."""
    from training.landmarks.calibrate import sharp_ref_from_cache
    from training.landmarks.run_utils import provenance

    tracker_j = _opt(exp_dir / "tracker.json") if exp_dir else None
    cal_j = _opt(exp_dir / "calibration.json") if exp_dir else None
    gates_j = _opt(exp_dir / "gates_result.json") if exp_dir else None
    if not smoke and not (tracker_j or cal_j):
        raise SystemExit("no calibration: run calibrate.py (and tune_tracker.py) first, or pass --smoke")
    if not smoke and not gates_j:
        raise SystemExit("no gates_result.json: run evaluate.py --report first (auto_enabled comes from G-STATION)")

    if cal_j:
        params = QualityParams.from_dict(cal_j["quality_params"])
    else:
        params = replace(QualityParams(), sharp_ref=sharp_ref or sharp_ref_from_cache())
    ship = (tracker_j or {}).get("shipped_calibration") or (cal_j or {}).get("shipped")
    if ship:
        T, tau, q_min = float(ship["T"]), [float(t) for t in ship["tau"]], float(ship["q_min"])
    else:  # smoke: uncalibrated, clearly labelled
        T, tau, q_min = 1.0, [0.5] * NUM_STATIONS, 0.2
    tracker = TrackerParams.from_dict(tracker_j["shipped"]["tracker"]) if tracker_j else TrackerParams()
    auto = [bool(a) for a in gates_j["auto_enabled"]] if gates_j else [True] * NUM_STATIONS

    m = D.load_manifest() if manifest is None else manifest
    test_patients = sorted(set(m.loc[m["split"] == "test", "patient_id"]))
    prov = read_json(run_dir / "provenance.json") if (run_dir / "provenance.json").exists() else provenance()
    from app.ml.landmarks.bundle import sha256_file

    metrics = _metrics_summary(exp_dir)
    if smoke or not ship:
        metrics["WARNING"] = "smoke/uncalibrated export: not for serving"
    return {
        "version": version,
        "arch": None,  # filled by the caller from the model
        "input_size": int(cfg["input"]["size"]),
        "temperature": T,
        "tau": tau,
        "quality": {**params.to_dict(), "q_min": q_min},
        "tracker": tracker.to_dict(),
        "auto_enabled": auto,
        "rejection": {"method": "msp"},
        "metrics": metrics,
        "train": {
            "candidate": cfg["candidate"], "variant": variant, "run": run_dir.name,
            "weights_sha256": sha256_file(weights), "git_sha": prov.get("git_sha"),
            "git_dirty": prov.get("git_dirty"), "manifest_sha256": prov.get("manifest_sha256"),
            "clip_set_sha256": prov.get("clip_set_sha256"), "frames_csv_sha256": prov.get("frames_csv_sha256"),
            "torch": prov.get("torch"), "device": prov.get("device"),
            "test_patients_sha256": hashlib.sha256(",".join(test_patients).encode()).hexdigest(),
            "calibration_source": "tracker.json" if (tracker_j or {}).get("shipped_calibration")
            else "calibration.json" if cal_j else "none (smoke)",
        },
    }


def smoke_serve(bundle_dir: Path) -> dict:
    """Load on CPU and classify one synthetic processor frame through the serving classes."""
    from app.ml.landmarks.bench import synthetic_olympus_frame
    from app.ml.landmarks.classifier import LandmarkModelManager, RealLandmarkClassifier

    b = load_bundle(bundle_dir, "cpu")
    clf = RealLandmarkClassifier(LandmarkModelManager(b))
    out = [clf.predict(synthetic_olympus_frame(1920, seed=i)) for i in range(4)]
    return {"version": b.version, "statuses": [o.status for o in out], "top": [o.top for o in out]}


def export(run_dir: Path, version: str, out_root: Path, exp_dir: Path | None = None, variant: str | None = None,
           smoke: bool = False, manifest=None, sharp_ref: float | None = None) -> dict:
    model, cfg, variant, weights = load_run_model(run_dir, variant)
    meta = build_meta(run_dir, exp_dir, version, cfg, variant, weights, smoke, manifest, sharp_ref)
    meta["arch"] = model.arch
    bundle_dir = write_bundle(out_root / version, model, meta)
    check = smoke_serve(bundle_dir)
    tarball = out_root / f"landmarks-{version}.tar.gz"
    digest = pack_bundle(bundle_dir, tarball)
    lock = BundleLock(version=version, s3_key=f"{S3_PREFIX}/landmarks-{version}.tar.gz", sha256=digest)
    return {"bundle_dir": str(bundle_dir), "tarball": str(tarball), "sha256": digest, "lock": vars(lock),
            "selftest_logits": read_json(bundle_dir / "meta.json")["selftest"]["logits"], "serve_check": check}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, help="runs/<exp> (uses <exp>/all_dev unless --run)")
    ap.add_argument("--run", type=Path, help="explicit run dir (smoke or fold runs)")
    ap.add_argument("--version", required=True, help="lm-<semver>")
    ap.add_argument("--variant", choices=("raw", "ema"))
    ap.add_argument("--out", type=Path, help="default: <run dir>/bundle")
    ap.add_argument("--smoke", action="store_true", help="allow an uncalibrated export (never serve it)")
    a = ap.parse_args(argv)
    run = a.run or (a.exp / "all_dev" if a.exp else None)
    if run is None:
        ap.error("--exp or --run is required")
    res = export(run, a.version, a.out or run / "bundle", a.exp, a.variant, a.smoke)
    print(json.dumps({k: v for k, v in res.items() if k != "selftest_logits"}, indent=2))
    print("\n# Upload to the PRIVATE bucket (never public, never GitHub Releases/LFS):")
    print(f'aws s3 cp {res["tarball"]} "s3://$PEACE_S3_BUCKET/{res["lock"]["s3_key"]}" --sse AES256')
    print(f"\n# Then commit {LOCK_PATH}:")
    print(json.dumps(res["lock"], indent=2))
    stations = ", ".join(f"{k}={t:.3f}" for k, t in zip(STATION_ORDER, read_json(Path(res["bundle_dir"]) /
                                                                                   "meta.json")["tau"]))
    print(f"\n# tau: {stations}")
    return 0 if np.isfinite(res["selftest_logits"]).all() else 1


if __name__ == "__main__":
    sys.exit(main())
