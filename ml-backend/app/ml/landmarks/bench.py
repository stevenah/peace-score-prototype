"""Stage-0 latency benchmark: PEACE alone vs PEACE + landmark, serving path.

Every measured frame goes through the live websocket's own per-frame
callable (JPEG decode -> PEACE 224 squash + full-frame landmark path ->
tracker -> payload) plus the JSON encode, on synthetic 1920x1080 processor
frames encoded as JPEG q0.8 like the browser. No patient data is involved,
and landmark weights are random (latency does not depend on them).

    # quick screen: one stream, back-to-back frames per config
    python -m app.ml.landmarks.bench --mode screen \\
        --archs effb0,r50,regy32,cnxt_t,dinov2_s14 --sizes 224,288 --frames 60

    # sustained gate (I1 G-SERVE): N concurrent sockets at 2 Hz
    python -m app.ml.landmarks.bench --mode sustained --archs effb0 --sizes 224 \\
        --minutes 30 --rate 2 --streams 2

Prints a table, then one JSON document (also written to --out if given).
The PEACE-only baseline is always measured first. ``drift`` is the median
latency of the last third of frames over the first third (throttling shows
up as drift > 1). ``rss_peak_mb`` is the process peak so far (cumulative).
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.ml.landmarks.layouts import OLYMPUS_1920

# -- synthetic processor frames ------------------------------------------------


def synthetic_olympus_frame(
    width: int = 1920,
    seed: int = 0,
    brightness: float = 1.0,
    badge: str | None = None,
) -> np.ndarray:
    """RGB frame in the Olympus 1920x1080 layout with a textured aperture.

    Black background, a lit octagon filled with smooth mucosa-like colour
    blobs plus fine texture, and dim OSD text on the dark left panel. Passes
    ``detect_layout``; no patient data.
    """
    rng = np.random.default_rng(seed)
    h = int(round(width * 1080 / 1920))
    s = width / 1920
    coarse = rng.random((h // 60 + 2, width // 60 + 2, 3)).astype(np.float32)
    blobs = cv2.resize(coarse, (width, h), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((h // 3, width // 3), dtype=np.float32), (width, h),
                      interpolation=cv2.INTER_LINEAR)[..., None]
    base = np.array([190.0, 95.0, 70.0], np.float32)  # reddish-pink mucosa
    img = base * (0.55 + 0.45 * blobs) * (0.8 + 0.4 * fine) * brightness
    img = np.clip(img, 0, 255).astype(np.uint8)

    mask = np.zeros((h, width), np.uint8)
    poly = np.array([(x * s, y * s) for x, y in OLYMPUS_1920.polygon], np.int32)
    cv2.fillPoly(mask, [poly], 1)
    frame = np.where(mask[..., None] > 0, img, 0).astype(np.uint8)
    for i in range(6):
        cv2.putText(frame, "ID 000000 12:00:00", (int(10 * s), int((40 + 30 * i) * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (200, 200, 200), 1)
    if badge:
        x0, y0, x1, y1 = OLYMPUS_1920.badge_box
        x1 = x1 + 18 if badge == "rdi" else x1
        cv2.rectangle(frame, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (255, 255, 255), -1)
    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    buf = BytesIO()
    Image.fromarray(frame, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# -- measurement ---------------------------------------------------------------


def _rss_peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def _pcts(xs: list[float]) -> dict:
    if not xs:
        return {"p50": None, "p95": None, "p99": None}
    a = np.asarray(xs)
    return {f"p{q}": round(float(np.percentile(a, q)), 1) for q in (50, 95, 99)}


def _drift(xs: list[float]) -> float | None:
    if len(xs) < 6:
        return None
    k = len(xs) // 3
    return round(float(np.median(xs[-k:]) / max(np.median(xs[:k]), 1e-9)), 3)


@dataclass
class _Stream:
    frame_ms: list[float]
    landmark_ms: list[float]


def _make_peace(peace: str):
    """Fresh per-socket PEACE pipeline (the managers are shared singletons)."""
    from app.ml.mock_models import MockMotionDetector, MockPEACEClassifier, MockRegionDetector
    from app.ml.pipeline import AnalysisPipeline

    if peace == "mock":
        return AnalysisPipeline(MockPEACEClassifier(), MockMotionDetector(), MockRegionDetector())
    from app.config import settings
    from app.ml.real_models import (ModelManager, RealMotionDetector, RealPEACEClassifier,
                                    RealRegionDetector, SessionState)

    model_path = settings.model_path if Path(settings.model_path).is_file() else ""
    manager = ModelManager.get_instance(model_path, "cpu")
    session = SessionState()
    return AnalysisPipeline(RealPEACEClassifier(manager, session), RealMotionDetector(),
                            RealRegionDetector(session))


def _landmark_manager(arch: str, size: int, workdir: Path, channels_last: bool):
    """Random-weight bundle written and loaded through the real bundle path."""
    from app.ml.landmarks.architectures import build
    from app.ml.landmarks.bundle import load_bundle, write_bundle
    from app.ml.landmarks.classifier import LandmarkModelManager

    version = "lm-0.0.0-bench"
    out = write_bundle(workdir / f"{arch}-{size}" / version, build(arch),
                       {"version": version, "input_size": size})
    return LandmarkModelManager(load_bundle(out, device="cpu"), channels_last=channels_last)


def _attach(pipe, manager) -> None:
    from app.ml.landmarks.classifier import RealLandmarkClassifier
    from app.ml.landmarks.tracker import StationTracker

    clf = RealLandmarkClassifier(manager)
    pipe.landmark_classifier = clf
    pipe.station_tracker = StationTracker(clf.tracker_params, clf.auto_enabled)
    pipe.landmark_meta = {"model_version": clf.model_version}


def _run_stream(pipe, jpegs: list[bytes], n_frames: int, rate: float, deadline: float | None,
                warmup: int, out: _Stream) -> None:
    from app.api.websocket import _analyse

    prev = None
    period = 1.0 / rate if rate > 0 else 0.0
    next_t = time.perf_counter()
    i = 0
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if deadline is None and i >= n_frames + warmup:
            break
        if period:
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            next_t += period
        t0 = time.perf_counter()
        prev, result = _analyse(pipe, jpegs[i % len(jpegs)], prev, time.time(), i)
        message = {"peace_score": result.peace_score, "motion": result.motion, "region": result.region}
        if result.landmark is not None:
            message["landmark"] = result.landmark
        json.dumps(message)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            out.frame_ms.append(ms)
            if result.landmark is not None and result.landmark["status"] != "skipped":
                out.landmark_ms.append(float(result.landmark["landmark_ms"]))
        i += 1


def run_config(label: str, manager, args, jpegs: list[bytes]) -> dict:
    sustained = args.mode == "sustained"
    streams = args.streams if sustained else 1
    rate = args.rate if sustained else args.screen_rate
    # --frames bounds each stream by count, otherwise by wall time (--minutes).
    n_frames = args.frames
    deadline = None if n_frames else time.perf_counter() + args.minutes * 60
    results = [_Stream([], []) for _ in range(streams)]
    threads = []
    for k in range(streams):
        pipe = _make_peace(args.peace)
        if manager is not None:
            _attach(pipe, manager)
        t = threading.Thread(target=_run_stream,
                             args=(pipe, jpegs, n_frames, rate, deadline, args.warmup, results[k]))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    frame_ms = [x for r in results for x in r.frame_ms]
    landmark_ms = [x for r in results for x in r.landmark_ms]
    row = {
        "config": label,
        "streams": streams,
        "frames": len(frame_ms),
        "frame_ms": _pcts(frame_ms),
        "landmark_ms": _pcts(landmark_ms),
        "drift": _drift(results[0].frame_ms),
        "rss_peak_mb": _rss_peak_mb(),
    }
    return row


def _print_table(rows: list[dict]) -> None:
    head = f"{'config':<22}{'n':>6}{'frame p50':>11}{'p95':>8}{'p99':>8}{'lm p50':>9}{'lm p95':>8}{'drift':>7}{'rss MB':>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        f, lm = r["frame_ms"], r["landmark_ms"]

        def c(v):
            return "-" if v is None else f"{v:.1f}"

        print(f"{r['config']:<22}{r['frames']:>6}{c(f['p50']):>11}{c(f['p95']):>8}{c(f['p99']):>8}"
              f"{c(lm['p50']):>9}{c(lm['p95']):>8}{c(r['drift']):>7}{r['rss_peak_mb']:>9.1f}")


def main(argv: list[str] | None = None) -> dict:
    import torch

    from app.ml.landmarks.architectures import available_archs
    from app.config import settings

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("screen", "sustained"), default="screen")
    p.add_argument("--archs", default="effb0,r50,regy32,cnxt_t,dinov2_s14")
    p.add_argument("--sizes", default="224")
    p.add_argument("--minutes", type=float, default=1.0)
    p.add_argument("--frames", type=int, default=0, help="frames per stream (overrides --minutes)")
    p.add_argument("--rate", type=float, default=2.0, help="frames/s per stream (sustained)")
    p.add_argument("--screen-rate", type=float, default=0.0, help="frames/s in screen mode (0 = back-to-back)")
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--threads", type=int, default=settings.torch_threads)
    p.add_argument("--peace", choices=("real", "mock"), default="real")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--channels-last", action="store_true", help="NHWC landmark forwards (convnets)")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    archs = [a for a in args.archs.split(",") if a]
    have = available_archs()
    skipped = [a for a in archs if a not in have]
    archs = [a for a in archs if a in have]
    sizes = [int(s) for s in args.sizes.split(",") if s]
    jpegs = [encode_jpeg(synthetic_olympus_frame(args.width, seed=k)) for k in range(8)]

    rows = [run_config("peace_only", None, args, jpegs)]
    _print_progress(rows[-1])
    with tempfile.TemporaryDirectory(prefix="lm-bench-") as tmp:
        for arch in archs:
            for size in sizes:
                manager = _landmark_manager(arch, size, Path(tmp), args.channels_last)
                label = f"{arch}@{size}" + ("/nhwc" if manager._channels_last else "")
                rows.append(run_config(label, manager, args, jpegs))
                _print_progress(rows[-1])
                del manager

    report = {
        "mode": args.mode,
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version(), "torch": torch.__version__,
                 "torch_threads": torch.get_num_threads()},
        "params": {k: v for k, v in vars(args).items() if k != "out"},
        "skipped_archs": skipped,
        "results": rows,
    }
    print()
    _print_table(rows)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print_progress(row: dict) -> None:
    print(f"# {row['config']}: frame p50 {row['frame_ms']['p50']} ms, "
          f"landmark p50 {row['landmark_ms']['p50']} ms", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
