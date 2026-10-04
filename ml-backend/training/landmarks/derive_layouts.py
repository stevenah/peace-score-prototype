"""Measure screen-layout geometry and scan for shortcuts in the aperture.

    python -m training.landmarks.derive_layouts legacy    # LEGACY_1350 geometry
    python -m training.landmarks.derive_layouts static    # overlay scan (FAILS on a hit)
    python -m training.landmarks.derive_layouts cap       # cap-ring score distribution

``legacy`` measures, over frames of every legacy (1350x1080) clip, the
aperture polygon and the OSD text drawn inside it, and prints the constants
to paste into ``app/ml/landmarks/layouts.py`` (bump LAYOUTS_VERSION when
they change). Everything is derived from the station clips only.

``static`` looks for anything drawn at a fixed position inside the
canonical aperture of the new cohort (zoom indicators, freeze markers,
icons): globally (temporal std/mean over >= 2,000 cached canvases) and per
clip (pixels static within a clip, recurring at the same position in many
clips). Either finding exits non-zero.

The cap-ring detector (``cap_ring_score``) finds the transparent distal
attachment cap: a thin circular edge, fixed relative to the scope tip, so it
persists in every frame while the mucosa moves. Score = the share of the
circle's in-aperture arc where the temporal-median edge map has a thin ring
(strong edge, much weaker 5 px inside and outside), maximised over centre
and radius.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from app.ml.landmarks.layouts import (
    CANON_SIZE,
    LAYOUTS_BY_NAME,
    LAYOUTS_VERSION,
    LEGACY_1350,
    OLYMPUS_1920,
    Layout,
    aperture_mask,
)
from training.landmarks.common import (
    AUDIT_DIR,
    CACHE_ROOT,
    MANIFEST_PATH,
    REPORTS_DIR,
    decode_frames,
    read_csv,
    write_csv,
)

# --- cap ring ----------------------------------------------------------------------

CAP_VERSION = "cap-2"
CAP_THRESHOLD = 0.35  # separates the reviewed cap / no-cap clips (README "Numbers")
# A cap stays on for the whole procedure: when most of a patient's clips
# score above threshold, all of that patient's clips are flagged.
CAP_PATIENT_SHARE = 0.5
CAP_MAX_FRAMES = 24
_CAP_SIZE = 160
_CAP_ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
_CAP_RADII = np.arange(44, 83)  # at 160 px; tested radii are 50..76, +-6 for context
_CAP_CENTRES = range(66, 95, 2)


def _edge_map(canvases: list[np.ndarray], inner: np.ndarray) -> np.ndarray:
    """Temporal median of the per-frame normalised gradient magnitude."""
    stack = []
    for c in canvases:
        g = cv2.cvtColor(cv2.resize(c, (_CAP_SIZE, _CAP_SIZE), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        stack.append(np.minimum(m / (float(np.median(m[inner])) + 1e-3), 10.0))
    e = np.median(np.stack(stack), axis=0) if len(stack) > 1 else stack[0]
    return np.where(inner, e, np.nan).astype(np.float32)


def cap_ring_score(canvases: list[np.ndarray], layout: Layout = OLYMPUS_1920) -> float:
    """0..1 cap evidence for one clip from its canonical 320 px canvases."""
    if not canvases:
        return 0.0
    inner = cv2.erode(aperture_mask(layout, _CAP_SIZE).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    e = _edge_map(canvases, inner)
    cos, sin = np.cos(_CAP_ANGLES), np.sin(_CAP_ANGLES)
    best = 0.0
    for cy in _CAP_CENTRES:
        for cx in _CAP_CENTRES:
            xs = np.clip(np.round(cx + _CAP_RADII[:, None] * cos).astype(int), 0, _CAP_SIZE - 1)
            ys = np.clip(np.round(cy + _CAP_RADII[:, None] * sin).astype(int), 0, _CAP_SIZE - 1)
            s = e[ys, xs]  # (radii, angles), nan outside the aperture
            with np.errstate(invalid="ignore"):
                # Tolerate +-1 px of radius (the ring is not a perfect circle).
                v = np.fmax(np.fmax(s[:-2], s[1:-1]), s[2:])  # radii 45..81
                ring, lo, hi = v[5:-5], v[:-10], v[10:]  # r vs r-5 vs r+5, r = 50..76
                valid = ~(np.isnan(ring) | np.isnan(lo) | np.isnan(hi))
                hit = (ring > 1.5) & (ring > 1.6 * np.fmax(lo, hi)) & valid
            n_valid = valid.sum(axis=1)
            ok = n_valid >= 0.25 * len(_CAP_ANGLES)
            if ok.any():
                best = max(best, float((hit.sum(axis=1)[ok] / n_valid[ok]).max()))
    return best


def _cap_job(args: tuple[str, str, list[str]]) -> tuple[str, float]:
    uid, layout_name, paths = args
    frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    return uid, cap_ring_score(frames, LAYOUTS_BY_NAME[layout_name])


def _clip_frame_paths(cache_root: Path, uid: str, src_idx: list[int], k: int) -> list[str]:
    """Up to k evenly spaced frames among the clip's UNIQUE frames: a freeze
    hold would otherwise make every edge (e.g. duodenal folds) look static."""
    if len(src_idx) > k:
        src_idx = [src_idx[i] for i in np.linspace(0, len(src_idx) - 1, k).round().astype(int)]
    return [str(cache_root / "frames" / uid / f"{i:05d}.jpg") for i in src_idx]


def cap_scores(cache_root: Path, rows: list[dict], workers: int = 8) -> dict[str, float]:
    """clip_uid -> cap score, computed from cached canvases (cached per version)."""
    import pandas as pd

    cache_file = AUDIT_DIR / "cap_scores.csv"
    known = {}
    if cache_file.exists():
        known = {r["clip_uid"]: float(r["cap_score"]) for r in read_csv(cache_file)
                 if r["cap_version"] == CAP_VERSION and r["layouts_version"] == LAYOUTS_VERSION}
    fr = pd.read_csv(cache_root / "frames.csv.gz", usecols=["clip_uid", "src_idx", "dup_of"],
                     dtype={"clip_uid": str, "dup_of": "Int64"})
    uniq = fr[fr["dup_of"].isna()].groupby("clip_uid")["src_idx"].apply(list).to_dict()
    todo = [(r["clip_uid"], r["layout"], _clip_frame_paths(cache_root, r["clip_uid"], uniq.get(r["clip_uid"], []),
                                                           CAP_MAX_FRAMES))
            for r in rows if r["clip_uid"] not in known and r.get("layout")]
    todo = [t for t in todo if t[2]]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for uid, s in ex.map(_cap_job, todo, chunksize=4):
                known[uid] = s
        write_csv(cache_file, [{"clip_uid": u, "cap_score": round(s, 4), "cap_version": CAP_VERSION,
                                "layouts_version": LAYOUTS_VERSION} for u, s in sorted(known.items())])
    return known


# --- legacy geometry ----------------------------------------------------------------

def _luma(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _legacy_clip_stats(args: tuple[str, int, int, int]) -> dict:
    """Per clip: bright-frame count and a text map from the temporal median."""
    path, w, h, every = args
    frames = [_luma(f) for _, f in decode_frames(Path(path), w, h, every=every)]
    if not frames:
        return {"n": 0}
    y = np.stack(frames)
    med = np.median(y, axis=0)
    # OSD text: brighter than its surroundings in the temporal median. Moving
    # content blurs out of the median; drawn text does not.
    bg = cv2.medianBlur(med.astype(np.uint8), 21).astype(np.float32)
    text = (med - bg) > 10
    return {"n": len(frames), "bright": (y > 25).sum(axis=0).astype(np.int32), "text": text}


def measure_legacy(rows: list[dict], workers: int = 8, every: int = 15) -> dict:
    """Aperture polygon, ROI and in-aperture OSD boxes of the legacy layout."""
    from training.landmarks.common import DATA_ROOT

    lay = LEGACY_1350
    jobs = [(str(DATA_ROOT / r["rel_path"]), lay.ref_width, lay.ref_height, every)
            for r in rows if r["cohort"] == "legacy_1350" and r["exclude_reason"] in ("", "combo",
                                                                                      "quarantine_suffix_mismatch")]
    bright = np.zeros((lay.ref_height, lay.ref_width), np.int64)
    text = np.zeros_like(bright)
    n_frames = n_clips = 0
    with ProcessPoolExecutor(workers) as ex:
        for st in ex.map(_legacy_clip_stats, jobs):
            if st["n"]:
                bright += st["bright"]
                text += st["text"]
                n_frames += st["n"]
                n_clips += 1
    frac = bright / max(n_frames, 1)
    # Drawn at this spot in >= 12% of clips. Moving mucosa reaches <= 9% at
    # any interior pixel; the rarest overlay (an icon under the badge) ~15%.
    text_map = text >= max(3, 0.12 * n_clips)
    poly = fit_octagon(frac > 0.5)
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0) + 1
    # In-aperture OSD: one box per text line (wide, short merge kernel),
    # padded 8 px, clipped to the ROI, kept if it reaches inside the aperture.
    # The aperture's own rim (+-7 px) is not text.
    filled = cv2.fillPoly(np.zeros(frac.shape, np.uint8), [poly.astype(np.int32)], 1)
    rim = cv2.dilate(filled, np.ones((15, 15), np.uint8)) & ~cv2.erode(filled, np.ones((15, 15), np.uint8)) & 1
    core = text_map & ~rim.astype(bool)
    inside = filled.astype(bool)
    n, lab, _, _ = cv2.connectedComponentsWithStats(cv2.dilate(core.astype(np.uint8), np.ones((9, 31), np.uint8)))
    boxes = []
    for k in range(1, n):
        comp = (lab == k) & core
        if not (comp & inside).any():
            continue
        ys, xs = np.nonzero(comp)
        boxes.append((max(int(xs.min()) - 8, int(x0)), max(int(ys.min()) - 8, int(y0)),
                      min(int(xs.max()) + 9, int(x1)), min(int(ys.max()) + 9, int(y1))))
    return {"n_clips": n_clips, "n_frames": n_frames, "polygon": [tuple(map(float, p)) for p in poly],
            "roi": (int(x0), int(y0), int(x1), int(y1)), "text_boxes": sorted(boxes),
            "aperture_px": int(inside.sum()), "text_px_inside": int((core & inside).sum())}


def _outward(profile: np.ndarray, start: int, step: int) -> int:
    """Index of the last lit pixel scanning from ``start`` in direction ``step``."""
    i = start
    while 0 <= i + step < len(profile) and profile[i + step]:
        i += step
    return i


def fit_octagon(lit: np.ndarray) -> np.ndarray:
    """Axis-aligned octagon (4 sides + 4 corner cuts) fitted to a lit mask.

    Every row and column is scanned from the centre outward to the last lit
    pixel, so bright OSD text outside the aperture (separated by dark gaps)
    never extends the edge. Sides are medians over the middle band; each
    corner cut is a robust line fit to the scan points inside that corner.
    Vertices clockwise from the top edge's left end, like OLYMPUS_1920.
    """
    h, w = lit.shape
    ys, xs = np.nonzero(lit)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    left = np.array([_outward(lit[y], cx, -1) for y in range(h)], float)
    right = np.array([_outward(lit[y], cx, 1) for y in range(h)], float)
    top = np.array([_outward(lit[:, x], cy, -1) for x in range(w)], float)
    bot = np.array([_outward(lit[:, x], cy, 1) for x in range(w)], float)
    band_y = slice(cy - (cy - int(np.median(top[cx - 50:cx + 50]))) // 3,
                   cy + (int(np.median(bot[cx - 50:cx + 50])) - cy) // 3)
    band_x = slice(cx - 100, cx + 100)
    # Vertices sit on the last lit pixel (inclusive), like OLYMPUS_1920.
    x_l, x_r = np.median(left[band_y]), np.median(right[band_y])
    y_t, y_b = np.median(top[band_x]), np.median(bot[band_x])
    rows, cols = np.arange(h), np.arange(w)

    def cut(sel_rows, xs_rows, sel_cols, ys_cols):
        pts = np.concatenate([np.stack([xs_rows[sel_rows], rows[sel_rows]], 1),
                              np.stack([cols[sel_cols], ys_cols[sel_cols]], 1)]).astype(np.float32)
        return tuple(cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())

    m = 4  # a point belongs to a cut if it is > m px inside the side lines
    in_y = (rows > y_t + m) & (rows < y_b - m)
    in_x = (cols > x_l + m) & (cols < x_r - m)
    tl = cut(in_y & (rows < cy) & (left > x_l + m), left, in_x & (cols < cx) & (top > y_t + m), top)
    tr = cut(in_y & (rows < cy) & (right < x_r - m), right, in_x & (cols > cx) & (top > y_t + m), top)
    br = cut(in_y & (rows > cy) & (right < x_r - m), right, in_x & (cols > cx) & (bot < y_b - m), bot)
    bl = cut(in_y & (rows > cy) & (left > x_l + m), left, in_x & (cols < cx) & (bot < y_b - m), bot)
    top_l, bot_l = (1.0, 0.0, 0.0, y_t), (1.0, 0.0, 0.0, y_b)
    left_l, right_l = (0.0, 1.0, x_l, 0.0), (0.0, 1.0, x_r, 0.0)
    seq = [tl, top_l, tr, right_l, br, bot_l, bl, left_l]
    verts = [_intersect(seq[i], seq[(i + 1) % 8]) for i in range(8)]
    return np.array(verts).round()


def _intersect(l1, l2) -> tuple[float, float]:
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    t = np.linalg.solve(a, np.array([x2 - x1, y2 - y1], np.float64))
    return x1 + t[0] * vx1, y1 + t[0] * vy1


# --- static overlay scan -------------------------------------------------------------

STATIC_STD = 2.0
STATIC_MEAN = 30.0
MIN_GLOBAL_FRAMES = 2000
PER_CLIP_MIN_UNIQUE = 8
RECURRENT_CLIP_FRAC = 0.02  # a per-clip static pixel recurring in >= 2% of clips


def _clip_static(args: tuple[str, list[str]]) -> tuple[str, np.ndarray | None, int]:
    uid, paths = args
    if len(paths) < PER_CLIP_MIN_UNIQUE:
        return uid, None, 0
    y = np.stack([_luma(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)) for p in paths])
    return uid, (y.std(axis=0) < STATIC_STD) & (y.mean(axis=0) > STATIC_MEAN), len(paths)


def static_scan(cache_root: Path = CACHE_ROOT, workers: int = 8, seed: int = 0) -> dict:
    """Global and per-clip static-pattern scan over cached OLYMPUS canvases."""
    import pandas as pd

    man = {r["clip_uid"]: r for r in read_csv(MANIFEST_PATH)}
    fr = pd.read_csv(cache_root / "frames.csv.gz", dtype={"clip_uid": str})
    fr = fr[fr["clip_uid"].map(lambda u: man.get(u, {}).get("layout") == OLYMPUS_1920.name)]
    uniq = fr[fr["dup_of"].isna()]
    mask = aperture_mask(OLYMPUS_1920, CANON_SIZE)
    path = lambda u, i: str(cache_root / "frames" / u / f"{int(i):05d}.jpg")  # noqa: E731

    # Global: temporal mean/std over a random sample of unique canvases.
    rng = np.random.default_rng(seed)
    sample = uniq.iloc[rng.permutation(len(uniq))[: max(MIN_GLOBAL_FRAMES, 4000)]]
    s1 = np.zeros((CANON_SIZE, CANON_SIZE)), np.zeros((CANON_SIZE, CANON_SIZE))
    acc, acc2 = s1
    for u, i in zip(sample["clip_uid"], sample["src_idx"]):
        y = _luma(cv2.cvtColor(cv2.imread(path(u, i)), cv2.COLOR_BGR2RGB)).astype(np.float64)
        acc += y
        acc2 += y * y
    n = len(sample)
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 0))
    global_hits = int(((std < STATIC_STD) & (mean > STATIC_MEAN) & mask).sum())

    # Per clip: static pixels within each clip's unique frames; an overlay
    # recurs at the same canvas position across many clips.
    jobs = [(u, [path(u, i) for i in g["src_idx"]]) for u, g in uniq.groupby("clip_uid")]
    counts = np.zeros((CANON_SIZE, CANON_SIZE), np.int32)
    clip_hits = []
    n_scanned = 0
    with ProcessPoolExecutor(workers) as ex:
        for u, st, _ in ex.map(_clip_static, jobs, chunksize=4):
            if st is None:
                continue
            n_scanned += 1
            st &= mask
            counts += st
            if st.sum() >= 20:
                clip_hits.append((u, int(st.sum())))
    recurrent = counts >= max(5, RECURRENT_CLIP_FRAC * max(n_scanned, 1))
    rec_hits = int((recurrent & mask).sum())
    return {"frames_global": n, "global_static_px": global_hits, "global_std_min_in_mask": float(std[mask].min()),
            "clips_scanned": n_scanned, "clips_with_static_px>=20": len(clip_hits),
            "recurrent_static_px": rec_hits, "max_clips_same_px": int(counts.max()),
            "top_clips": sorted(clip_hits, key=lambda t: -t[1])[:10],
            "ok": global_hits == 0 and rec_hits == 0}


# --- main -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["legacy", "static", "cap"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    rows = read_csv(MANIFEST_PATH)
    if args.cmd == "legacy":
        m = measure_legacy(rows, args.workers)
        print(f"legacy: {m['n_clips']} clips, {m['n_frames']} frames")
        print(f"  roi={m['roi']}")
        print(f"  polygon={tuple(m['polygon'])}")
        print(f"  text_boxes={tuple(m['text_boxes'])}  (text px inside aperture: {m['text_px_inside']})")
        print(f"  current LAYOUTS_VERSION={LAYOUTS_VERSION}")
        return 0
    if args.cmd == "static":
        res = static_scan(CACHE_ROOT, args.workers)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "static_scan.txt").write_text("\n".join(f"{k}: {v}" for k, v in res.items()) + "\n")
        for k, v in res.items():
            print(f"  {k}: {v}")
        if not res["ok"]:
            print("FAIL: static bright pattern inside the aperture mask", file=sys.stderr)
            return 1
        print("OK: no static overlay inside the aperture")
        return 0
    scores = cap_scores(CACHE_ROOT, [r for r in rows if r["layout"]], args.workers)
    v = np.array(sorted(scores.values()))
    print(f"cap scores: n={len(v)}  >= {CAP_THRESHOLD}: {(v >= CAP_THRESHOLD).sum()}")
    print("  quantiles 10/50/90/95/99:", np.percentile(v, [10, 50, 90, 95, 99]).round(3).tolist())
    hist, edges = np.histogram(v, bins=np.linspace(0, 1, 21))
    for h, e in zip(hist, edges):
        print(f"  {e:4.2f} {'#' * int(h // 5)} {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
