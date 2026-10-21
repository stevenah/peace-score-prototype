"""Self-test of one landmark bundle, reported check by check.

    python -m app.ml.landmarks.selfcheck --bundle runs/<exp>/all_dev/bundle/lm-1.0.0 [--out result.json]

The checks are the ones a bundle must pass before it is served
(``bundle.load_bundle`` and ``LandmarkModelManager.warmup``), run one at a
time so a failure names its step:

    meta       meta.json parses and matches this code's schema, stations,
               preprocessing and layout versions
    files      every listed file matches its sha256
    selftest   the weights load and reproduce the reference logits (CPU fp32)
    live       synthetic processor frames go through layout detection,
               preprocessing, the quality gate, the forward, the wire checks
               and the bundle's tracker

The first failing check stops the run; later ones are reported as skipped.
Exit code 0 whenever the checks ran (``ok`` in the result says how they went).
No patient data is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

CHECKS = (
    ("meta", "Metadata"),
    ("files", "File hashes"),
    ("selftest", "Reference logits"),
    ("live", "Live path"),
)
LIVE_FRAMES = 4


def _check_meta(path: Path, ctx: dict) -> str:
    from app.ml.landmarks.bundle import check_meta, read_meta

    ctx["meta"] = read_meta(path)
    parsed = check_meta(path)
    return (f"{parsed['version']}: {parsed['arch']} at {parsed['input_size']} px, "
            f"{ctx['meta']['layouts_version']}, {ctx['meta']['preprocess_version']}")


def _check_files(path: Path, ctx: dict) -> str:
    from app.ml.landmarks.bundle import verify_files

    verify_files(path, ctx["meta"])
    files = ctx["meta"]["files"]
    size_mb = sum((path / name).stat().st_size for name in files) / (1024 * 1024)
    return f"{len(files)} file{'s' if len(files) != 1 else ''}, {size_mb:.1f} MB, sha256 match"


def _check_selftest(path: Path, ctx: dict) -> str:
    from app.ml.landmarks.bundle import SELFTEST_ATOL, SELFTEST_RTOL, load_bundle, selftest_error

    ctx["bundle"] = bundle = load_bundle(path, device="cpu")
    st = ctx["meta"]["selftest"]
    diff = selftest_error(bundle.model, st)
    return (f"max |logit difference| {diff:.2g} "
            f"(tolerance rtol {st.get('rtol', SELFTEST_RTOL):g}, atol {st.get('atol', SELFTEST_ATOL):g})")


def _check_live(path: Path, ctx: dict) -> str:
    from app.ml.landmarks.bench import synthetic_olympus_frame
    from app.ml.landmarks.classifier import LandmarkModelManager, RealLandmarkClassifier
    from app.ml.landmarks.payload import validate_output
    from app.ml.landmarks.tracker import StationTracker

    bundle = ctx["bundle"]
    manager = LandmarkModelManager(bundle)
    manager.warmup()
    clf = RealLandmarkClassifier(manager)
    tracker = StationTracker(bundle.tracker, bundle.auto_enabled)
    statuses: Counter[str] = Counter()
    latencies = []
    for seed in range(LIVE_FRAMES):
        out = clf.predict(synthetic_olympus_frame(1920, seed=seed))
        validate_output(out)
        tracker.update(out.evidence())
        statuses[out.status] += 1
        latencies.append(out.latency_ms)
    if statuses.get("unsupported_layout"):
        raise RuntimeError(f"{statuses['unsupported_layout']} of {LIVE_FRAMES} synthetic frames "
                           f"were not recognised as {', '.join(bundle.serve_layouts)}")
    counts = ", ".join(f"{n} {s}" for s, n in sorted(statuses.items()))
    return f"{LIVE_FRAMES} synthetic frames ({counts}), median {float(np.median(latencies)):.0f} ms on CPU"


_RUN = {"meta": _check_meta, "files": _check_files, "selftest": _check_selftest, "live": _check_live}


def run(bundle_dir: str | Path) -> dict:
    """Run every check on ``bundle_dir``; never raises for a bad bundle."""
    path = Path(bundle_dir)
    t0 = time.perf_counter()
    ctx: dict = {}
    checks = []
    failed = False
    for name, label in CHECKS:
        if failed:
            checks.append({"name": name, "label": label, "status": "skipped", "detail": ""})
            continue
        try:
            detail = _RUN[name](path, ctx)
            checks.append({"name": name, "label": label, "status": "passed", "detail": detail})
        except Exception as exc:
            failed = True
            detail = str(exc).removeprefix(f"landmark bundle {path}: ")
            checks.append({"name": name, "label": label, "status": "failed",
                           "detail": detail or type(exc).__name__})
    version = (ctx.get("meta") or {}).get("version")
    return {
        "ok": not failed,
        "version": version if isinstance(version, str) else path.name,
        "checks": checks,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--out", type=Path, help="also write the result JSON here")
    args = p.parse_args(argv)

    result = run(args.bundle)
    for c in result["checks"]:
        print(f"{c['status']:8s} {c['label']}: {c['detail']}")
    print(f"{result['version']}: {'OK' if result['ok'] else 'FAILED'} in {result['seconds']} s")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
