"""Extract the canonical frame cache: training/landmarks/cache/v1/.

    python -m training.landmarks.extract_cache [--workers 6] [--limit N]

For every manifest clip that is not an exact duplicate or too short
(quarantined, combo, id-uncertain and legacy clips are included, so later
adjudication never needs re-extraction):

    ffmpeg (explicit BT.709 tv->pc -> rgb24), every 3rd frame (10 fps)
      -> detect_mode on the FULL frame (badge; metadata only)
      -> canonicalize -> frames/{clip_uid}/{src_idx:05d}.jpg (q95, 4:4:4)
      -> to_model_input(224) -> quality metrics

Outputs (only masked 320 px canvases leave the laptop; see README):

    frames/{clip_uid}/{src_idx:05d}.jpg
    frames.csv.gz   clip_uid, src_idx, u, t_s, sharpness, dark_frac, sat_frac,
                    redout, quality, sharp_ratio_to_clip_median, dhash, dup_of,
                    mode_frame, layout_ok
    thumbs.npy      uint8 (n_frames, 32, 32): grey centre crop of each canvas,
                    row-aligned with frames.csv.gz (near-duplicate search)
    CACHE_INFO.json versions, manifest sha256, clip-set sha256, ffmpeg, counts

``dhash`` and the thumbnails use the canvas centre (x, y in [64, 256)), which
is inside the aperture for every layout, so the constant black pad and mask
cannot dominate them. ``dup_of`` is the src_idx of an earlier frame of the
same clip that is near-identical (frozen video); empty for unique frames.

Extraction is incremental: per-clip parts under ``_parts/`` are reused when
the clip's md5 and the preprocessing/layout versions are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.ml.landmarks.layouts import LAYOUTS_BY_NAME, LAYOUTS_VERSION, detect_layout, detect_mode
from app.ml.landmarks.preprocess import PREPROCESS_VERSION, canonicalize, to_model_input
from app.ml.landmarks.quality import QUALITY_SIZE, QualityParams, quality_metrics, quality_score
from training.landmarks.common import (
    CACHE_ROOT,
    DATA_ROOT,
    FFMPEG_COLOR_VF,
    MANIFEST_PATH,
    REPO_ROOT,
    decode_frames,
    ffmpeg_version,
    is_full_length,
    read_csv,
    sha256_file,
    write_csv,
    write_json,
)

EVERY = 3  # keep every 3rd source frame: 10 fps at 30 fps
JPEG_QUALITY = 95
THUMB = 32
CENTRE = (64, 256)  # canvas centre crop used for thumbnails and dHash
DUP_MAD = 1.0  # mean |diff| (0..255) of two thumbnails below which frames are "the same"
# Near-duplicate CLIP rule (near_duplicate_pairs; manifest near_dup_group, leak audit):
NEAR_MAX_HAMMING = 8  # dHash bits
NEAR_MAX_MAD = 4.0  # thumbnail mean |diff|
NEAR_MIN_FRAC = 0.3  # share of one clip's sampled frames with a partner in the other clip
SKIP_REASONS = {"dup_exact", "too_short"}

FRAME_FIELDS = ["clip_uid", "src_idx", "u", "t_s", "sharpness", "dark_frac", "sat_frac", "redout",
                "quality", "sharp_ratio_to_clip_median", "dhash", "dup_of", "mode_frame", "layout_ok"]


def centre_thumb(canvas: np.ndarray) -> np.ndarray:
    a, b = CENTRE
    grey = cv2.cvtColor(canvas[a:b, a:b], cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, (THUMB, THUMB), interpolation=cv2.INTER_AREA)


def dhash(thumb: np.ndarray) -> str:
    """64-bit difference hash of a grey thumbnail, as 16 hex chars."""
    small = cv2.resize(thumb, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return f"{int(np.packbits(bits).view('>u8')[0]):016x}"


def frame_dups(thumbs: np.ndarray, src: list[int], mad: float = DUP_MAD) -> list[int | None]:
    """For each frame, the src_idx of an earlier unique near-identical frame."""
    out: list[int | None] = []
    uniq: list[int] = []
    t = thumbs.astype(np.int16)
    for i in range(len(t)):
        hit = None
        if uniq:
            d = np.abs(t[uniq] - t[i]).mean(axis=(1, 2))
            j = int(np.argmin(d))
            if d[j] < mad:
                hit = src[uniq[j]]
        out.append(hit)
        if hit is None:
            uniq.append(i)
    return out


def _part_dir(cache_root: Path) -> Path:
    return cache_root / "_parts"


def _part_key(row: dict) -> str:
    return f"{row['md5']}|{PREPROCESS_VERSION}|{LAYOUTS_VERSION}|{EVERY}|{JPEG_QUALITY}"


def extract_clip(row: dict, cache_root: Path = CACHE_ROOT, data_root: Path = DATA_ROOT) -> dict:
    """Decode one clip and write its canvases and per-frame part files."""
    uid = row["clip_uid"]
    if is_full_length(row["rel_path"]):
        raise ValueError(f"refusing full-length path {row['rel_path']}")
    layout = LAYOUTS_BY_NAME[row["layout"]]
    w, h = int(row["width"]), int(row["height"])
    n_src = max(int(row["n_frames"]), 1)
    fps = float(row["fps"]) or 30.0
    out_dir = cache_root / "frames" / uid
    tmp_dir = cache_root / "frames" / f".{uid}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    params = QualityParams()
    recs, thumbs = [], []
    enc = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
           cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    for idx, full in decode_frames(data_root / row["rel_path"], w, h, every=EVERY):
        mode = detect_mode(full, layout)
        det = detect_layout(full)
        canvas = canonicalize(full, layout)
        m = quality_metrics(to_model_input(canvas, layout, QUALITY_SIZE), layout, params)
        ok = cv2.imwrite(str(tmp_dir / f"{idx:05d}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), enc)
        if not ok:
            raise OSError(f"could not write frame {idx} of {uid}")
        th = centre_thumb(canvas)
        thumbs.append(th)
        recs.append({"clip_uid": uid, "src_idx": idx, "u": round(min(idx / max(n_src - 1, 1), 1.0), 4),
                     "t_s": round(idx / fps, 3), "sharpness": round(m.sharpness, 2),
                     "dark_frac": round(m.dark_frac, 4), "sat_frac": round(m.sat_frac, 4),
                     "redout": round(m.redout, 4), "quality": round(quality_score(m, params), 4),
                     "dhash": dhash(th), "mode_frame": mode,
                     "layout_ok": det is not None and det.name == layout.name})
    if recs:
        med = float(np.median([r["sharpness"] for r in recs]))
        dups = frame_dups(np.stack(thumbs), [r["src_idx"] for r in recs])
        for r, d in zip(recs, dups):
            r["sharp_ratio_to_clip_median"] = round(r["sharpness"] / med, 4) if med > 0 else 0.0
            r["dup_of"] = d
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    parts = _part_dir(cache_root)
    parts.mkdir(parents=True, exist_ok=True)
    write_csv(parts / f"{uid}.csv", recs, FRAME_FIELDS)
    np.save(parts / f"{uid}.npy", np.stack(thumbs) if thumbs else np.zeros((0, THUMB, THUMB), np.uint8))
    (parts / f"{uid}.key").write_text(_part_key(row))
    return {"clip_uid": uid, "n": len(recs)}


def _is_current(row: dict, cache_root: Path) -> bool:
    parts = _part_dir(cache_root)
    key = parts / f"{row['clip_uid']}.key"
    return key.exists() and key.read_text() == _part_key(row) and (cache_root / "frames" / row["clip_uid"]).is_dir()


def select_rows(manifest: list[dict]) -> list[dict]:
    return [r for r in manifest if r["exclude_reason"] not in SKIP_REASONS and r["layout"]
            and not is_full_length(r["rel_path"])]


def consolidate(rows: list[dict], cache_root: Path = CACHE_ROOT) -> tuple[int, int]:
    """Merge per-clip parts into frames.csv.gz + thumbs.npy (manifest order)."""
    parts = _part_dir(cache_root)
    all_recs, all_thumbs = [], []
    for r in sorted(rows, key=lambda r: r["clip_uid"]):
        p = parts / f"{r['clip_uid']}.csv"
        if not p.exists():
            continue
        recs = read_csv(p)
        th = np.load(parts / f"{r['clip_uid']}.npy")
        assert len(recs) == len(th), r["clip_uid"]
        all_recs += recs
        all_thumbs.append(th)
    write_csv(cache_root / "frames.csv.gz", all_recs, FRAME_FIELDS)
    thumbs = np.concatenate(all_thumbs) if all_thumbs else np.zeros((0, THUMB, THUMB), np.uint8)
    np.save(cache_root / "thumbs.npy", thumbs)
    return len({r["clip_uid"] for r in all_recs}), len(all_recs)


def clip_set_sha256(rows: list[dict]) -> str:
    """Stable id of WHICH clips were extracted (unaffected by later manifest
    edits such as split/fold assignment, which change the file's sha256)."""
    s = "\n".join(sorted(f"{r['clip_uid']}:{r['md5']}" for r in rows))
    return hashlib.sha256(s.encode()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*.jpg"))


def run(workers: int = 6, limit: int | None = None, cache_root: Path = CACHE_ROOT,
        manifest_path: Path = MANIFEST_PATH) -> dict:
    manifest = read_csv(manifest_path)
    rows = select_rows(manifest)
    if limit:
        rows = rows[:limit]
    todo = [r for r in rows if not _is_current(r, cache_root)]
    print(f"cache: {len(rows)} clips selected, {len(todo)} to extract -> {cache_root}", flush=True)
    t0 = time.time()
    done = 0
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(extract_clip, r, cache_root) for r in todo]
            for f in as_completed(futs):
                f.result()
                done += 1
                if done % 50 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)} clips  {el:.0f}s  eta {el / done * (len(todo) - done):.0f}s",
                          flush=True)
    n_clips, n_frames = consolidate(rows, cache_root)
    info = {
        "cache_version": "v1",
        "preprocess_version": PREPROCESS_VERSION,
        "layouts_version": LAYOUTS_VERSION,
        "manifest_sha256_at_extraction": sha256_file(manifest_path),
        "clip_set_sha256": clip_set_sha256(rows),
        "ffmpeg": ffmpeg_version(),
        "ffmpeg_vf": FFMPEG_COLOR_VF,
        "every_nth_frame": EVERY,
        "jpeg_quality": JPEG_QUALITY,
        "jpeg_sampling": "4:4:4",
        "canvas_px": 320,
        "dup_mad": DUP_MAD,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {"clips": n_clips, "frames": n_frames, "jpeg_bytes": _du(cache_root / "frames"),
                   "extracted_this_run": len(todo)},
        "frames_csv_sha256": sha256_file(cache_root / "frames.csv.gz"),
    }
    write_json(cache_root / "CACHE_INFO.json", info)
    return info


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(axis=-1)


def thumb_mad(a: np.ndarray, b: np.ndarray, gain_invariant: bool = False) -> np.ndarray:
    """Mean |diff| of thumbnails (broadcasting over leading axes); ``gain_invariant``
    removes each thumbnail's mean first (re-exports with a brightness change)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    if gain_invariant:
        a = a - a.mean(axis=(-2, -1), keepdims=True)
        b = b - b.mean(axis=(-2, -1), keepdims=True)
    return np.abs(a - b).mean(axis=(-2, -1))


def is_near_duplicate(n_a: int, n_b: int, matched_a: int, matched_b: int, min_frac: float = NEAR_MIN_FRAC) -> bool:
    """The clip-level rule: >= 2 matching frames and >= ``min_frac`` of either clip's frames."""
    return max(matched_a, matched_b) >= 2 and max(matched_a / max(n_a, 1), matched_b / max(n_b, 1)) >= min_frac


def near_duplicate_pairs(cache_root: Path, frames, rows: list[dict], per_clip: int = 30,
                         max_hamming: int = NEAR_MAX_HAMMING, max_mad: float = NEAR_MAX_MAD,
                         min_frac: float = NEAR_MIN_FRAC, min_thumb_std: float = 6.0, min_sharpness: float = 150.0,
                         gain_invariant: bool = False):
    """Clip pairs sharing near-identical content, from cached thumbnails.

    Frames are matched when their dHash is within ``max_hamming`` bits AND
    their 32x32 thumbnails differ by < ``max_mad`` (mean |diff|; after
    removing each thumbnail's mean with ``gain_invariant``). Textureless
    frames (thumbnail std < ``min_thumb_std`` or sharpness < ``min_sharpness``:
    red-out, mucosal contact, smooth vignetted red) match anything and are
    skipped. Returns (uid_a, uid_b, frac_a, frac_b): the
    share of each clip's sampled unique frames that match a frame of the
    other clip; pairs with a share >= ``min_frac`` on either side and >= 2
    matching frames are reported.
    """
    thumbs = np.load(cache_root / "thumbs.npy")
    keep = set(r["clip_uid"] for r in rows)
    fr = frames.reset_index(drop=True)
    textured = (thumbs.reshape(len(thumbs), -1).std(axis=1) >= min_thumb_std) & \
        (fr["sharpness"].to_numpy() >= min_sharpness)
    sel = []
    live = fr["clip_uid"].isin(keep) & fr["dup_of"].isna() & textured[fr.index.to_numpy()]
    for _, g in fr[live].groupby("clip_uid"):
        idx = g.index.to_numpy()
        if len(idx) > per_clip:
            idx = idx[np.linspace(0, len(idx) - 1, per_clip).round().astype(int)]
        sel.append(idx)
    if not sel:
        return []
    sel = np.concatenate(sel)
    uids = fr.loc[sel, "clip_uid"].to_numpy()
    codes = np.array([int(h, 16) for h in fr.loc[sel, "dhash"].astype(str)], dtype=np.uint64)
    th = thumbs[sel].astype(np.int16)
    n_per = {u: c for u, c in zip(*np.unique(uids, return_counts=True))}
    matches: dict[tuple[str, str], tuple[set, set]] = {}
    chunk = 2048
    for s in range(0, len(sel), chunk):
        a = codes[s:s + chunk]
        ham = _popcount(a[:, None] ^ codes[None, :])
        ii, jj = np.nonzero(ham <= max_hamming)
        ii = ii + s
        m = (jj > ii) & (uids[ii] != uids[jj])
        for i, j in zip(ii[m], jj[m]):
            if thumb_mad(th[i], th[j], gain_invariant) >= max_mad:
                continue
            ua, ub = uids[i], uids[j]
            key = (ua, ub) if ua < ub else (ub, ua)
            fa, fb = matches.setdefault(key, (set(), set()))
            (fa if ua == key[0] else fb).add(i)
            (fb if ua == key[0] else fa).add(j)
    out = []
    for (ua, ub), (fa, fb) in matches.items():
        if is_near_duplicate(n_per[ua], n_per[ub], len(fa), len(fb), min_frac):
            out.append((ua, ub, round(len(fa) / n_per[ua], 3), round(len(fb) / n_per[ub], 3)))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--limit", type=int, default=None, help="first N clips only (smoke test)")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    args = ap.parse_args(argv)
    info = run(args.workers, args.limit, args.cache_root)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
