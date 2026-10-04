"""Online-equals-offline check (I1): replay a patient through a running /api/v1/ws/live.

    # terminal 1 (local, isolated stack; same bundle, 1 torch thread both sides):
    PEACE_LANDMARKS_ENABLED=true PEACE_LANDMARK_BUNDLE_DIR=<bundle> PEACE_TORCH_THREADS=2 \
        uvicorn app.main:app --port 8000
    # terminal 2:
    python -m training.landmarks.replay_ws --patient HT### --bundle <bundle> [--url ws://127.0.0.1:8000/api/v1/ws/live]

The patient's ORIGINAL clips (never the full-length recordings) are decoded
chronologically at full resolution with the explicit BT.709 chain, sampled
at 2 Hz with a fixed phase, JPEG-encoded at quality 80 with PIL (what the
browser's ``canvas.toBlob('image/jpeg', 0.8)`` sends) and pushed through the
websocket one frame at a time. The same JPEG bytes are decoded in process and
run through ``RealLandmarkClassifier`` + ``StationTracker`` from the same
bundle. Per frame the status, top station, station states, current station
and events must match; probabilities must agree within ``--atol``. Needs
``landmark_every_n=1`` on the server (skipped frames have no offline twin).

Only aggregate agreement is printed; frame-level diffs stay in the scratch
output file you pass with ``--out`` (keep it out of the repo).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.landmarks.stations import STATION_ORDER
from training.landmarks import dataset as D
from training.landmarks.common import DATA_ROOT, decode_frames, is_full_length


def patient_jpegs(patient_id: str, hz: float = 2.0, phase: int = 0, quality: int = 80,
                  max_frames: int | None = None) -> list[bytes]:
    """Chronological 2 Hz full-resolution JPEGs of one patient's station clips."""
    m = D.load_manifest()
    clips = D.stream_clips(m, [patient_id])
    clips = clips.sort_values(["rel_t_s_f", "clip_uid"], na_position="last")
    out: list[bytes] = []
    for r in clips.itertuples():
        if is_full_length(r.rel_path):  # defensive: the manifest never lists them
            continue
        every = max(1, int(round(float(r.fps or 30) / hz)))
        for src, rgb in decode_frames(DATA_ROOT / r.rel_path, int(r.width), int(r.height), 1):
            if src % every != phase % every:
                continue
            buf = BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
            out.append(buf.getvalue())
            if max_frames and len(out) >= max_frames:
                return out
    return out


def offline(jpegs: list[bytes], bundle_dir: Path, threads: int = 2) -> list[dict]:
    import torch

    from app.ml.landmarks.bundle import load_bundle
    from app.ml.landmarks.classifier import LandmarkModelManager, RealLandmarkClassifier
    from app.ml.landmarks.tracker import StationTracker

    torch.set_num_threads(threads)
    b = load_bundle(bundle_dir, "cpu")
    clf = RealLandmarkClassifier(LandmarkModelManager(b))
    tracker = StationTracker(b.tracker, b.auto_enabled)
    rows = []
    for data in jpegs:
        full = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
        out = clf.predict(full)
        step = tracker.update(out.evidence()).to_dict()
        rows.append({"status": out.status, "top": STATION_ORDER[out.top] if out.top is not None else None,
                     "probs": list(out.probs) if out.probs else None, **step})
    return rows


async def online(jpegs: list[bytes], url: str) -> list[dict]:
    import websockets

    rows = []
    async with websockets.connect(url, max_size=None) as ws:
        for data in jpegs:
            await ws.send(data)
            msg = json.loads(await ws.recv())
            lm = msg.get("landmark")
            if lm is None:
                raise SystemExit(f"no landmark block in the reply ({msg.get('type')}); is PEACE_LANDMARKS_ENABLED on?")
            rows.append(lm)
    return rows


def compare(on: list[dict], off: list[dict], atol: float = 1e-3) -> dict:
    keys = ("status", "top", "current", "stations")
    agree = {k: 0 for k in keys}
    events_equal = 0
    dp = []
    first = None
    for i, (a, b) in enumerate(zip(on, off)):
        for k in keys:
            if a.get(k) == b.get(k):
                agree[k] += 1
            elif first is None:
                first = {"frame": i, "field": k}
        if a.get("events") == b.get("events"):
            events_equal += 1
        if a.get("probs") and b.get("probs"):
            dp.append(float(np.max(np.abs(np.asarray(a["probs"]) - np.asarray(b["probs"])))))
    n = min(len(on), len(off))
    return {"n_frames": n, **{f"{k}_agreement": agree[k] / n if n else None for k in keys},
            "events_agreement": events_equal / n if n else None,
            "max_abs_dp": max(dp) if dp else None, "probs_within_atol": all(d <= atol for d in dp),
            "first_mismatch": first,
            "equal": n > 0 and all(agree[k] == n for k in keys) and events_equal == n and all(d <= atol for d in dp)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/api/v1/ws/live")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, help="scratch JSON with both sequences (never commit)")
    a = ap.parse_args(argv)
    jpegs = patient_jpegs(a.patient, a.hz, a.phase, max_frames=a.max_frames)
    off = offline(jpegs, a.bundle, a.threads)
    on = asyncio.run(online(jpegs, a.url))
    res = compare(on, off, a.atol)
    if a.out:
        a.out.write_text(json.dumps({"online": on, "offline": off, "result": res}, default=str))
    print(json.dumps(res, indent=2))
    return 0 if res["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
