"""Leakage audit across splits: embedding nearest neighbours AND the dHash near-duplicate rule.

    python -m training.landmarks.leak_audit            # after folds.py

Two detectors look at every clip pair across splits (dev vs test/shift):

1. Embeddings (below): catch re-encoded, re-trimmed or colour-shifted
   re-exports whose frames are no longer pixel-similar.
2. The thumbnail dHash rule of ``extract_cache.near_duplicate_pairs``, run
   GAIN-INVARIANT here (each 32 px thumbnail's mean removed): catch
   re-exports that were shifted/cropped by a few pixels, which moves
   ImageNet features a lot (cosine ~0.87, far below T_dup).

This audit embeds 3 cached frames
per clip (u ~ 0.25 / 0.5 / 0.75) with torchvision ResNet-50 IMAGENET1K_V2
(BSD-3, checked by sha256) and compares clips by cosine similarity:
clip-pair similarity = mean over clip A's frames of the best cosine to a
frame of clip B, symmetrised by max. Pair distributions:

    W  same patient, distinct takes (not duplicates, no shared time)
    B  different patients, same split          (the null)
    X  across splits (test/shift vs dev)

Duplicate threshold, from W: the most similar pair of DISTINCT takes of one
patient is as close as legitimate content gets, so
T_dup = max(W) + (1 - max(W)) / 2. Any pair across splits (or across
patients) at or above T_dup is flagged. B above T_dup is the false-alarm rate.

Positive controls, all planted from one seeded test clip; REQUIRED ones make
the audit exit 1 when missed:

1. planted_copy (required, embedding): the clip's frames re-exported at JPEG
   q90 with +2% gain, injected into dev under a new id;
2. planted_stress_shift2_q75 (required, EITHER detector): the same frames
   with a 2 px shift, JPEG q75 and +3% gain. Embeddings miss this (measured
   0.82-0.91 on 40 test clips); the gain-invariant dHash rule caught 59/60;
3. planted_recut (informative): the clip embedded from different frames
   (u = 0.35 / 0.6 / 0.85), as if re-trimmed;
4. known_reencode_* (required, embedding): the real re-encoded duplicates
   the manifest excluded (dup_near) against their canonical copies.

Patient identity: whether a test patient reappears in dev under another ID
is also measured (per-station best-match similarity averaged over shared
stations, halves of one patient vs different patients). ImageNet features
turn out not to identify patients, so this part is descriptive only; the
custodian's repeat-procedure mapping remains the control for that risk.

Outputs data/landmarks_private/reports/leak_audit.{md,csv}.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.ml.landmarks.layouts import LAYOUTS_BY_NAME, LAYOUTS_VERSION
from app.ml.landmarks.preprocess import PREPROCESS_VERSION, normalize, to_model_input
from training.landmarks.common import AUDIT_DIR, CACHE_ROOT, MANIFEST_PATH, REPORTS_DIR, read_csv, write_csv
from training.landmarks.extract_cache import (
    NEAR_MAX_HAMMING,
    NEAR_MAX_MAD,
    _popcount,
    centre_thumb,
    dhash,
    is_near_duplicate,
    near_duplicate_pairs,
    thumb_mad,
)

WEIGHTS = "IMAGENET1K_V2"
WEIGHTS_SHA256_PREFIX = "11ad3fa6"  # torchvision resnet50-11ad3fa6.pth
U_POINTS = (0.25, 0.5, 0.75)


# --- embeddings -------------------------------------------------------------------------

def load_model(device: str):
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    w = getattr(ResNet50_Weights, WEIGHTS)
    if not w.url.rsplit("-", 1)[-1].startswith(WEIGHTS_SHA256_PREFIX):
        raise RuntimeError(f"unexpected weights {w.url}")
    model = resnet50(weights=w)
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / w.url.rsplit("/", 1)[-1]
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if not digest.startswith(WEIGHTS_SHA256_PREFIX):
        raise RuntimeError(f"weights sha256 {digest} does not match {WEIGHTS_SHA256_PREFIX}")
    model.fc = torch.nn.Identity()
    return model.eval().to(device), digest


def embed_images(model, images: list[np.ndarray], layouts: list[str], device: str, batch: int = 64) -> np.ndarray:
    """L2-normalised 2048-d embeddings of canonical canvases (fp32)."""
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(images), batch):
            x = torch.stack([normalize(to_model_input(img, LAYOUTS_BY_NAME[lay], 224))
                             for img, lay in zip(images[s:s + batch], layouts[s:s + batch])])
            e = model(x.to(device)).float().cpu().numpy()
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)


def pick_frames(frames, uid: str) -> list[int]:
    """src_idx of the cached frames nearest u = 0.25/0.5/0.75 (unique frames preferred)."""
    g = frames.get_group(uid)
    uniq = g[g["dup_of"].isna()]
    g = uniq if len(uniq) >= 3 else g
    picks = []
    for u in U_POINTS:
        i = int((g["u"] - u).abs().to_numpy().argmin())
        picks.append(int(g["src_idx"].iloc[i]))
    return picks


def _read(cache_root: Path, uid: str, idx: int) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(cache_root / "frames" / uid / f"{idx:05d}.jpg")), cv2.COLOR_BGR2RGB)


def clip_embeddings(rows: list[dict], cache_root: Path, device: str) -> tuple[np.ndarray, str]:
    """(n_clips, 3, 2048) embeddings, cached in AUDIT_DIR by (clip_uid, frames)."""
    import pandas as pd

    fr = pd.read_csv(cache_root / "frames.csv.gz", dtype={"clip_uid": str, "dup_of": "Int64"}).groupby("clip_uid")
    cache_file = AUDIT_DIR / "embeddings_resnet50.npz"
    known = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        known = {k: v for k, v in zip(z["keys"], z["emb"])}
    # versions in the key: a mask/preprocessing change re-embeds everything
    keys = [f"{r['clip_uid']}:" + ",".join(map(str, pick_frames(fr, r["clip_uid"])))
            + f"@{PREPROCESS_VERSION}/{LAYOUTS_VERSION}" for r in rows]
    todo = [(i, k) for i, k in enumerate(keys) if k not in known]
    model, digest = load_model(device)
    if todo:
        imgs, lays = [], []
        for i, k in todo:
            uid, idxs = k.split("@")[0].split(":")
            for idx in idxs.split(","):
                imgs.append(_read(cache_root, uid, int(idx)))
                lays.append(rows[i]["layout"])
        e = embed_images(model, imgs, lays, device).reshape(len(todo), 3, -1)
        for (_, k), v in zip(todo, e):
            known[k] = v
        np.savez(cache_file, keys=np.array(list(known)), emb=np.stack(list(known.values())))
    return np.stack([known[k] for k in keys]).astype(np.float32), digest


# --- similarities --------------------------------------------------------------------------

def clip_sim(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """(na, 3, d) x (nb, 3, d) -> (na, nb) symmetrised mean-of-best-frame cosine."""
    s = np.einsum("aid,bjd->abij", ea, eb)
    ab = s.max(axis=3).mean(axis=2)
    ba = s.max(axis=2).mean(axis=2)
    return np.maximum(ab, ba)


def _related(rows: list[dict]) -> np.ndarray:
    """Pairs that legitimately share content: same dup/near-dup group or
    overlapping OSD time (they must not calibrate the "distinct takes" ceiling)."""
    n = len(rows)
    rel = np.zeros((n, n), bool)
    idx = {r["clip_uid"]: i for i, r in enumerate(rows)}
    for key in ("dup_group", "near_dup_group"):
        vals = np.array([r.get(key) or f"_{i}" for i, r in enumerate(rows)])
        rel |= vals[:, None] == vals[None, :]
    for i, r in enumerate(rows):
        for u in (r.get("overlap_with") or "").split("|"):
            if u in idx:
                rel[i, idx[u]] = rel[idx[u], i] = True
    return rel


def audit(rows: list[dict], emb: np.ndarray) -> dict:
    """Distributions, T_dup and flags. ``rows`` align with ``emb``."""
    n = len(rows)
    sim = clip_sim(emb, emb)
    pid = np.array([r["patient_id"] for r in rows])
    split = np.array([r["split"] for r in rows])
    upper = np.triu(np.ones((n, n), bool), 1)
    same_p = pid[:, None] == pid[None, :]
    W = sim[upper & same_p & ~_related(rows)]
    B = sim[upper & ~same_p & (split[:, None] == split[None, :])]
    cross = upper & (split[:, None] != split[None, :]) & ((split[:, None] == "dev") | (split[None, :] == "dev"))
    X = sim[cross]
    w_max = float(W.max())
    t_dup = w_max + (1.0 - w_max) / 2
    flags = []
    for i, j in zip(*np.nonzero(upper & ~same_p & (sim >= t_dup))):
        flags.append({"kind": "cross_split" if split[i] != split[j] else "cross_patient_same_split",
                      "sim": round(float(sim[i, j]), 4),
                      "split_a": split[i], "clip_a": rows[i]["clip_uid"], "patient_a": pid[i],
                      "label_a": rows[i]["label"], "split_b": split[j], "clip_b": rows[j]["clip_uid"],
                      "patient_b": pid[j], "label_b": rows[j]["label"]})
    q = [50, 90, 99, 99.9, 100]
    return {"n_clips": n, "w_max": w_max, "t_dup": t_dup,
            "W": np.percentile(W, q).round(4).tolist(), "B": np.percentile(B, q).round(4).tolist(),
            "X": np.percentile(X, q).round(4).tolist() if len(X) else [],
            "n_W": int(W.size), "n_B": int(B.size), "n_X": int(X.size),
            "B_above_t_dup": int((B >= t_dup).sum()), "X_above_t_dup": int((X >= t_dup).sum()),
            "flags": flags, "sim": sim}


def patient_identity(rows: list[dict], sim: np.ndarray, n_null: int = 3000, seed: int = 0) -> dict:
    """Can embeddings tell the same patient from another? Statistic for two
    clip sets = mean over shared stations of the best same-station
    similarity. Same patient: alternate clips per station split into halves.
    Null: halves of two different dev patients."""
    lab = np.array([r["label"] for r in rows])
    pid = np.array([r["patient_id"] for r in rows])
    half: dict[tuple[str, int], list[int]] = defaultdict(list)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    for i, r in enumerate(rows):
        k = (r["patient_id"], r["label"])
        half[(r["patient_id"], seen[k] % 2)].append(i)
        seen[k] += 1

    def stat(a: list[int], b: list[int]) -> float | None:
        vals = []
        for k in set(lab[a]) & set(lab[b]):
            ia, ib = [i for i in a if lab[i] == k], [j for j in b if lab[j] == k]
            vals.append(sim[np.ix_(ia, ib)].max())
        return float(np.mean(vals)) if len(vals) >= 3 else None

    same = [v for p in sorted(set(pid)) if (v := stat(half[(p, 0)], half.get((p, 1), []))) is not None]
    dev = sorted({r["patient_id"] for r in rows if r["split"] == "dev"})
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        p, q = rng.choice(dev, 2, replace=False)
        if (v := stat(half[(p, 0)], half.get((q, 1), []))) is not None:
            null.append(v)
    same, null = np.array(same), np.array(null)
    t = float(np.percentile(same, 10))
    return {"n_same": len(same), "n_null": len(null), "same_p50": float(np.median(same)),
            "null_p50": float(np.median(null)), "t_at_90_sensitivity": t,
            "false_alarm_at_90_sensitivity": float((null >= t).mean())}


def dhash_match(thumbs_a: np.ndarray, thumbs_b: np.ndarray, gain_invariant: bool = True) -> dict:
    """The near_duplicate_pairs rule applied to two frame sets (32 px grey thumbnails)."""
    code = lambda th: np.array([int(dhash(t), 16) for t in th], dtype=np.uint64)  # noqa: E731
    ham = _popcount(code(thumbs_a)[:, None] ^ code(thumbs_b)[None, :])
    match = (ham <= NEAR_MAX_HAMMING) & (thumb_mad(thumbs_a[:, None], thumbs_b[None, :], gain_invariant) < NEAR_MAX_MAD)
    ma, mb = int(match.any(1).sum()), int(match.any(0).sum())
    na, nb = len(thumbs_a), len(thumbs_b)
    return {"frac": max(ma / max(na, 1), mb / max(nb, 1)), "caught": is_near_duplicate(na, nb, ma, mb)}


def textured_frames(frames, uid: str, per_clip: int = 30, min_sharpness: float = 150.0) -> list[int]:
    """src_idx of the unique, sharp frames of a clip that the dHash rule samples."""
    g = frames.get_group(uid)
    g = g[g["dup_of"].isna() & (g["sharpness"] >= min_sharpness)]
    idx = g["src_idx"].to_numpy()
    if len(idx) > per_clip:
        idx = idx[np.linspace(0, len(idx) - 1, per_clip).round().astype(int)]
    return [int(i) for i in idx]


def degrade(img: np.ndarray, quality: int = 90, gain: float = 1.02, shift: int = 0) -> np.ndarray:
    """Re-export-like degradation: content shift (mask re-applied downstream),
    gain, JPEG re-encode."""
    if shift:
        img = np.roll(img, (shift, shift), axis=(0, 1))
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _frames_at(frames, uid: str, us: tuple[float, ...]) -> list[int]:
    g = frames.get_group(uid)
    return [int(g["src_idx"].iloc[int((g["u"] - u).abs().to_numpy().argmin())]) for u in us]


def controls(rows: list[dict], emb: np.ndarray, t_dup: float, frames, model, device: str,
             cache_root: Path, extra: list[tuple[str, dict, dict]], seed: int = 0) -> dict:
    """Similarity of each positive control to its source (embedding: >= t_dup) and, for the
    planted copies, the dHash rule's verdict against the source clip (``<name>_dhash``)."""
    rng = np.random.default_rng(seed)
    test_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
    i = int(rng.choice(test_idx))
    src = rows[i]

    def emb_of(images: list[np.ndarray]) -> np.ndarray:
        return embed_images(model, images, [src["layout"]] * len(images), device)[None]

    frames_src = [_read(cache_root, src["clip_uid"], k) for k in src["_frames"]]
    variants = {
        # Required: the same frames re-exported (JPEG q90, +2% gain).
        "planted_copy": emb_of([degrade(f) for f in frames_src]),
        # Informative stress tests.
        "planted_stress_shift2_q75": emb_of([degrade(f, 75, 1.03, 2) for f in frames_src]),
        "planted_recut": emb_of([_read(cache_root, src["clip_uid"], k)
                                 for k in _frames_at(frames, src["clip_uid"], (0.35, 0.6, 0.85))]),
    }
    out: dict = {}
    for name, e in variants.items():
        out[name] = float(clip_sim(emb[i][None], e)[0, 0])
        s = clip_sim(e, emb)[0]  # the planted clip against every real clip
        out[f"{name}_nn_is_source"] = bool(int(np.argmax(s)) == i)
    # dHash rule (gain-invariant) on the source clip's sampled frames vs their planted copies
    canv = [_read(cache_root, src["clip_uid"], k) for k in textured_frames(frames, src["clip_uid"])]
    canv = [c for c in canv if centre_thumb(c).std() >= 6.0]
    thumbs = np.stack([centre_thumb(c) for c in canv])
    for name, fn in (("planted_copy", lambda c: degrade(c)),
                     ("planted_stress_shift2_q75", lambda c: degrade(c, 75, 1.03, 2))):
        out[f"{name}_dhash"] = dhash_match(thumbs, np.stack([centre_thumb(fn(c)) for c in canv]))
    for name, a, b in extra:
        ea = embed_images(model, [_read(cache_root, a["clip_uid"], k) for k in a["_frames"]], [a["layout"]] * 3, device)
        eb = embed_images(model, [_read(cache_root, b["clip_uid"], k) for k in b["_frames"]], [b["layout"]] * 3, device)
        out[name] = float(clip_sim(ea[None], eb[None])[0, 0])
    out["source_clip"] = src["clip_uid"]
    return out


EMBEDDING_REQUIRED = ("planted_copy",)  # plus every known_reencode_*
EITHER_REQUIRED = ("planted_stress_shift2_q75",)  # embedding OR the dHash rule


def control_verdicts(ctl: dict, t_dup: float) -> dict[str, dict]:
    """{control: {similarity, embedding, dhash, caught, required}} for every similarity control."""
    out = {}
    for k, v in ctl.items():
        if not isinstance(v, float):
            continue
        emb = v >= t_dup
        dh = (ctl.get(f"{k}_dhash") or {}).get("caught")
        either = k in EITHER_REQUIRED
        required = k in EMBEDDING_REQUIRED or k.startswith("known_reencode") or either
        out[k] = {"similarity": v, "embedding": emb, "dhash": dh, "required": required,
                  "caught": bool(emb or dh) if either else emb}
    return out


def dhash_flags(pairs, rows: list[dict]) -> list[dict]:
    """near_duplicate_pairs output between different patients -> audit flags."""
    by = {r["clip_uid"]: r for r in rows}
    flags = []
    for a, b, fa, fb in pairs:
        ra, rb = by[a], by[b]
        if ra["patient_id"] == rb["patient_id"]:
            continue
        flags.append({"kind": "dhash_cross_split" if ra["split"] != rb["split"] else "dhash_cross_patient_same_split",
                      "sim": max(fa, fb), "split_a": ra["split"], "clip_a": a, "patient_a": ra["patient_id"],
                      "label_a": ra["label"], "split_b": rb["split"], "clip_b": b, "patient_b": rb["patient_id"],
                      "label_b": rb["label"]})
    return flags


def write_report(res: dict, pat: dict, ctl: dict, meta: dict, path_md: Path, path_csv: Path,
                 dflags: list[dict] | None = None) -> bool:
    dflags = dflags or []
    write_csv(path_csv, res["flags"] + dflags, ["kind", "sim", "split_a", "clip_a", "patient_a", "label_a",
                                                "split_b", "clip_b", "patient_b", "label_b"])
    verdicts = control_verdicts(ctl, res["t_dup"])
    ok = all(v["caught"] for v in verdicts.values() if v["required"]) and ctl["planted_copy_nn_is_source"]
    n_cross = sum(f["kind"] == "cross_split" for f in res["flags"])
    n_pat = sum(f["kind"] == "cross_patient_same_split" for f in res["flags"])
    n_dcross = sum(f["kind"] == "dhash_cross_split" for f in dflags)
    n_dpat = sum(f["kind"] == "dhash_cross_patient_same_split" for f in dflags)
    row = lambda name, v: f"| {name} | " + " | ".join(map(str, v)) + " |"  # noqa: E731
    lines = [
        "# Leak audit (embedding nearest neighbour)", "",
        f"torchvision ResNet-50 {WEIGHTS} (sha256 {meta['weights_sha256'][:16]}...), {res['n_clips']} clips "
        f"(dev/test/shift) x 3 frames at u = 0.25/0.5/0.75, clip-pair cosine similarity.", "",
        "| pair distribution | p50 | p90 | p99 | p99.9 | max | n pairs |", "|---|---|---|---|---|---|---|",
        row("W same patient, distinct takes", res["W"] + [res["n_W"]]),
        row("B different patients, same split", res["B"] + [res["n_B"]]),
        row("X across splits (test/shift vs dev)", res["X"] + [res["n_X"]]), "",
        f"- T_dup = max(W) + (1 - max(W))/2 = {res['w_max']:.4f} -> **{res['t_dup']:.4f}**",
        f"- B pairs >= T_dup (false alarms): {res['B_above_t_dup']}; X pairs >= T_dup: {res['X_above_t_dup']}",
        f"- embedding flags: **{n_cross} cross-split**, {n_pat} cross-patient within a split",
        f"- dHash rule (gain-invariant thumbnails, {meta.get('dhash_pairs', 0)} near-duplicate clip pairs in "
        f"total): **{n_dcross} cross-split**, {n_dpat} cross-patient within a split (all flags in {path_csv.name})",
        "",
        "## Positive controls", "",
        "| control | required | similarity | embedding (>= T_dup) | dHash rule | caught | nearest real clip is "
        "the source |", "|---|---|---|---|---|---|---|",
        *[f"| {k} | {'yes' if v['required'] else 'no'}{' (either)' if k in EITHER_REQUIRED else ''} | "
          f"{v['similarity']:.4f} | {'yes' if v['embedding'] else 'no'} | "
          f"{'' if v['dhash'] is None else ('yes' if v['dhash'] else 'no')} | {'yes' if v['caught'] else 'no'} | "
          f"{ctl.get(k + '_nn_is_source', '')} |" for k, v in verdicts.items()],
        "",
        f"Source clip {ctl['source_clip']} (test). Required: the planted copy (JPEG q90, +2% gain; embedding), "
        f"the shifted stress copy (2 px shift, JPEG q75, +3% gain; embedding OR dHash rule, since ImageNet "
        f"features move a lot under a shift) and every real re-encoded duplicate the manifest excluded "
        f"(embedding). Informative: a re-cut (different frames of the same clip).", "",
        "## Patient identity (descriptive)", "",
        f"Same patient (halves of {pat['n_same']} patients) vs different dev patients ({pat['n_null']} pairs): "
        f"median {pat['same_p50']:.3f} vs {pat['null_p50']:.3f}; at 90% sensitivity "
        f"(T = {pat['t_at_90_sensitivity']:.3f}) the false-alarm rate is "
        f"{pat['false_alarm_at_90_sensitivity']:.0%}. ImageNet features do not identify patients, so a person "
        f"filed under two IDs cannot be excluded by this audit; ask the data custodian for the "
        f"repeat-procedure mapping.", "",
        "## Verdict", "",
        ("PASS" if ok and n_cross == 0 and n_dcross == 0 else "FAIL")
        + f": {n_cross} embedding and {n_dcross} dHash cross-split duplicates; required controls "
        + ("caught." if ok else "MISSED."),
    ]
    path_md.write_text("\n".join(lines) + "\n")
    return ok and n_cross == 0 and n_dcross == 0


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args(argv)
    manifest = read_csv(MANIFEST_PATH)
    fr = pd.read_csv(CACHE_ROOT / "frames.csv.gz", dtype={"clip_uid": str, "dup_of": "Int64"})
    grouped = fr.groupby("clip_uid")
    have = set(fr["clip_uid"])
    rows = [r for r in manifest if r["split"] in ("dev", "test", "shift") and r["clip_uid"] in have]
    for r in manifest:
        if r["clip_uid"] in have:
            r["_frames"] = pick_frames(grouped, r["clip_uid"])
    emb, digest = clip_embeddings(rows, CACHE_ROOT, args.device)
    res = audit(rows, emb)
    pat = patient_identity(rows, res.pop("sim"))
    # Real positives: each excluded re-encoded copy vs its canonical clip.
    by_nd = defaultdict(list)
    for r in manifest:
        if r["near_dup_group"] and r["clip_uid"] in have:
            by_nd[r["near_dup_group"]].append(r)
    extra = []
    for g in by_nd.values():
        losers = [r for r in g if r["exclude_reason"] == "dup_near"]
        keep = [r for r in g if r["exclude_reason"] != "dup_near"]
        extra += [(f"known_reencode_{x['clip_uid']}", x, keep[0]) for x in losers if keep]
    model, _ = load_model(args.device)
    ctl = controls(rows, emb, res["t_dup"], grouped, model, args.device, CACHE_ROOT, extra)
    pairs = near_duplicate_pairs(CACHE_ROOT, fr, rows, gain_invariant=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = write_report(res, pat, ctl, {"weights_sha256": digest, "dhash_pairs": len(pairs)},
                      REPORTS_DIR / "leak_audit.md", REPORTS_DIR / "leak_audit.csv", dhash_flags(pairs, rows))
    print((REPORTS_DIR / "leak_audit.md").read_text())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
