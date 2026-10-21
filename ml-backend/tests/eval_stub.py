"""A stand-in eval module for test_evals.py, run as ``python -m tests.eval_stub``."""

from __future__ import annotations

import argparse
import json
import sys
import time

from app.ml.landmarks.progress import report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--mode", default="ok")
    a = p.parse_args()
    print("starting", flush=True)
    report(1, 2, "steps")
    if a.mode == "fail":
        print("ValueError: it broke", file=sys.stderr)
        return 3
    if a.mode == "hang":
        time.sleep(120)
    if a.mode == "no_result":
        return 0
    report(2, 2, "steps")
    with open(a.out, "w") as f:
        json.dump({"value": 42, "target": a.target}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
