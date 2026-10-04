"""PEACE non-regression check across environments (torch builds, images).

Dumps the raw (pre-smoothing) region/score probabilities of the PEACE model
for a fixed set of deterministic frames, then compares two dumps.

    # inside each environment / image
    python scripts/peace_regression.py dump out.json [--model app/models/best_model.pt]
    # anywhere
    python scripts/peace_regression.py compare before.json after.json

The comparison passes when every argmax is identical and the largest absolute
probability difference is within --atol (default 1e-4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_FRAMES = 8


def _frames() -> list[np.ndarray]:
    """Deterministic, endoscopy-ish frames: smooth colour fields plus texture."""
    rng = np.random.default_rng(1234)
    h, w = 1080, 1920
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for i in range(N_FRAMES):
        base = np.stack([
            150 + 60 * np.sin(xx / (180 + 20 * i)),
            80 + 40 * np.cos(yy / (140 + 15 * i)),
            60 + 30 * np.sin((xx + yy) / (220 + 10 * i)),
        ], axis=-1)
        noise = rng.normal(0, 12, size=(h, w, 3))
        frames.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    return frames


def dump(out: Path, model_path: str) -> None:
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.ml.real_models import ModelManager

    torch.set_num_threads(2)
    manager = ModelManager(model_path, device="cpu")
    rows = []
    for frame in _frames():
        # Same squash-to-224 path as the websocket (PIL resize of the full frame).
        small = np.array(Image.fromarray(frame).resize((224, 224)))
        region, score = manager.predict(small)
        rows.append({"region": region.tolist(), "score": score.tolist()})
    out.write_text(json.dumps({"torch": torch.__version__, "rows": rows}, indent=1))
    print(f"wrote {out} (torch {torch.__version__})")


def compare(a: Path, b: Path, atol: float) -> int:
    da, db = json.loads(a.read_text()), json.loads(b.read_text())
    max_diff = 0.0
    argmax_mismatch = 0
    for ra, rb in zip(da["rows"], db["rows"], strict=True):
        for key in ("region", "score"):
            pa, pb = np.asarray(ra[key]), np.asarray(rb[key])
            max_diff = max(max_diff, float(np.abs(pa - pb).max()))
            argmax_mismatch += int(pa.argmax() != pb.argmax())
    ok = argmax_mismatch == 0 and max_diff <= atol
    print(
        f"torch {da['torch']} vs {db['torch']}: max|dp|={max_diff:.2e} "
        f"argmax mismatches={argmax_mismatch} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_dump = sub.add_parser("dump")
    p_dump.add_argument("out", type=Path)
    p_dump.add_argument("--model", default="app/models/best_model.pt")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a", type=Path)
    p_cmp.add_argument("b", type=Path)
    p_cmp.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.cmd == "dump":
        dump(args.out, args.model)
        return 0
    return compare(args.a, args.b, args.atol)


if __name__ == "__main__":
    raise SystemExit(main())
