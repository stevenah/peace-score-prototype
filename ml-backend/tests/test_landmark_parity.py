"""Train/serve parity of the landmark path, and the blank-frame system check.

- blank interior -> low_quality through the real serving classifier (fast).
- parity (slow; skipped without the private dataset and an exported bundle):
  the frame-cache path (q95 canvas -> to_model_input) and the serving path
  (original frame -> PIL JPEG q80, what the browser sends -> decode ->
  canonicalize -> to_model_input) must give the same top-1 on >= 99% of
  frames with mean |dp| <= 0.02, and the same quality-gate decisions.

Pick the bundle with PEACE_LANDMARK_PARITY_BUNDLE=<bundle dir>; otherwise
the newest bundle under ml-backend/runs/ is used. Only aggregate numbers
are printed.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from app.ml.landmarks.layouts import OLYMPUS_1920, aperture_mask

BACKEND = Path(__file__).resolve().parents[1]


def test_blank_interior_is_low_quality_through_the_serving_classifier(tiny_bundle):
    from training.landmarks.sanity import blank_check

    res = blank_check(tiny_bundle)
    assert res["pass"], res["statuses"]
    assert set(res["statuses"]) == {"grey", "pink", "dark", "white"}


def test_unmasked_positive_control_canvas_shows_the_badge_the_mask_removes():
    from app.ml.landmarks.bench import synthetic_olympus_frame
    from app.ml.landmarks.preprocess import canonicalize
    from training.landmarks.sanity import badge_canvas_box, canonicalize_unmasked

    frame = synthetic_olympus_frame(1920, seed=4, badge="NBI")
    raw = canonicalize_unmasked(frame, OLYMPUS_1920)
    masked = canonicalize(frame, OLYMPUS_1920)
    inside = aperture_mask(OLYMPUS_1920)
    np.testing.assert_array_equal(raw[inside], masked[inside])
    x0, y0, x1, y1 = badge_canvas_box()
    assert not inside[y0:y1, x0:x1].any()  # the badge is outside the served aperture ...
    assert raw[y0:y1, x0:x1].max() > 150  # ... and visible without the mask


# -- slow: cache path vs serving path on real clips --------------------------------------------------------

def _is_smoke(meta: dict) -> bool:
    """Smoke bundles are ~uniform, so their top-1 is noise (their |dp| and quality still count)."""
    return "WARNING" in (meta.get("metrics") or {}) or str((meta.get("train") or {}).get("run", "")).startswith("smoke")


def _bundle_dir() -> Path | None:
    """PEACE_LANDMARK_PARITY_BUNDLE, else the newest trained bundle, else the newest smoke bundle."""
    env = os.environ.get("PEACE_LANDMARK_PARITY_BUNDLE")
    if env:
        return Path(env)
    found = sorted((BACKEND / "runs").glob("**/bundle/lm-*/meta.json"), key=lambda p: p.stat().st_mtime)
    real = [p for p in found if not _is_smoke(json.loads(p.read_text()))]
    pick = (real or found)[-1:]
    return pick[0].parent if pick else None


@pytest.mark.slow
def test_cache_path_matches_serving_path():
    from PIL import Image

    from app.ml.landmarks.bundle import load_bundle
    from app.ml.landmarks.preprocess import canonicalize, normalize, to_model_input
    from app.ml.landmarks.quality import QUALITY_SIZE, quality_metrics, quality_score
    from training.landmarks import dataset as D
    from training.landmarks.common import CACHE_ROOT, DATA_ROOT, MANIFEST_PATH, decode_frames, is_full_length
    from training.landmarks.run_utils import softmax

    bundle_dir = _bundle_dir()
    if not (MANIFEST_PATH.exists() and (CACHE_ROOT / "frames.csv.gz").exists() and DATA_ROOT.exists()):
        pytest.skip("private landmark dataset not available")
    if bundle_dir is None:
        pytest.skip("no exported landmark bundle (set PEACE_LANDMARK_PARITY_BUNDLE)")
    torch = pytest.importorskip("torch")
    b = load_bundle(bundle_dir, "cpu")
    size = b.input_size

    m = D.load_manifest()
    frames = D.load_frames()
    clips = m[(m["split"] == "dev") & (m["exclude_reason"] == "") & (m["cohort"] == "new_1920")]
    clips = clips.sample(n=min(60, len(clips)), random_state=0)
    serve, cache = [], []
    for r in clips.itertuples():
        assert not is_full_length(r.rel_path)
        f = frames[frames["clip_uid"] == r.clip_uid]
        want = {int(f.iloc[(f["u"] - u).abs().argmin()]["src_idx"]) for u in (0.2, 0.35, 0.5, 0.65, 0.8)}
        for src, rgb in decode_frames(DATA_ROOT / r.rel_path, int(r.width), int(r.height), 3):
            if src not in want:
                continue
            buf = BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
            full = np.asarray(Image.open(BytesIO(buf.getvalue())).convert("RGB"))
            serve.append(canonicalize(full, OLYMPUS_1920))
            cache.append(D.read_canvas(CACHE_ROOT, r.clip_uid, src))

    def run(canvases):
        imgs = [to_model_input(c, OLYMPUS_1920, size) for c in canvases]
        with torch.inference_mode():
            logits = b.model(torch.stack([normalize(i) for i in imgs]))[0].numpy()
        # quality is measured at QUALITY_SIZE whatever the model size (as the serving classifier does)
        q = np.array([quality_score(quality_metrics(to_model_input(c, OLYMPUS_1920, QUALITY_SIZE), OLYMPUS_1920,
                                                    b.quality), b.quality) for c in canvases])
        return softmax(logits, b.temperature), q

    ps, qs = run(serve)
    pc, qc = run(cache)
    top1 = float((ps.argmax(1) == pc.argmax(1)).mean())
    confident = pc.max(1) >= 0.5  # where the cache path would pass the tau floor
    top1_conf = float((ps.argmax(1) == pc.argmax(1))[confident].mean()) if confident.any() else float("nan")
    dp = float(np.abs(ps - pc).mean())
    gate = float(((qs >= b.q_min) == (qc >= b.q_min)).mean())
    dq = float(np.abs(qs - qc).mean())
    print(f"parity over {len(ps)} frames: top-1 {top1:.3f} (p>=0.5: {top1_conf:.3f} on {int(confident.sum())}), "
          f"mean|dp| {dp:.4f}, gate agreement {gate:.3f}, mean|dq| {dq:.3f} ({b.version})")
    assert dp <= 0.02
    assert gate >= 0.95 and dq <= 0.05
    if not _is_smoke(b.meta):
        assert top1 >= 0.99
