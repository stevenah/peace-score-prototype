"""Stage A: frozen-backbone probes over the 5 dev folds (M2 model selection).

    python -m training.landmarks.embed_probe --candidates effb0_in1k,r50_in1k,... [--device mps]

For every candidate in ``backbones.ALLOWLIST``:

1. Embed 5 quality-passing frames per clip from u in [0.2, 0.8] (fp32, frozen,
   eval-mode backbone; no augmentation). Clip embedding = mean of the
   L2-normalised frame embeddings.
2. Probes, per outer dev fold k (train = the other 4 folds, patient-grouped):
   - L2 logistic regression on standardised clip embeddings, C picked on the
     inner-val fold ((k+1) % 5) by macro-F1, then refit on all 4 folds;
   - kNN, k=20, cosine, majority vote (ties -> summed similarity).
   OOF clip macro-F1 and balanced accuracy with patient-cluster bootstrap CIs.
3. Confound visibility: linear decodability of the clip's imaging mode (WL vs
   NBI), transparent cap and room (processor), overall AND within station
   (mean per-station balanced accuracy; 0.5 = not decodable beyond the
   station itself).

Only usable, primary-evaluation dev clips are used (contested soft-labelled
clips and the locked test set are never touched). Writes aggregate numbers to
``training/landmarks/results/stageA_v1.csv`` (+ ``_per_class.csv``); frame
embeddings are cached under the gitignored ``cache/embeddings/``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from training.landmarks import dataset as D
from training.landmarks.backbones import ALLOWLIST, get_candidate, load_pretrained
from training.landmarks.common import CACHE_ROOT, ML_BACKEND
from training.landmarks.run_utils import (
    RESULTS_DIR,
    RUNS_ROOT,
    balanced_accuracy,
    cluster_bootstrap,
    confusion,
    macro_f1,
    prf_from_cm,
    resolve_device,
    write_json,
)

EMB_DIR = ML_BACKEND / "training" / "landmarks" / "cache" / "embeddings"
DEFAULT_CANDIDATES = ("effb0_in1k", "r50_in1k", "regy32_in1k", "regy32_endopos", "cnxt_t_in1k", "dinov2_s14")
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
KNN_K = 20
N_FOLDS = 5


# -- frame selection and embeddings ---------------------------------------------------------------

def pick_probe_frames(frames: pd.DataFrame, clips: pd.DataFrame, n: int = 5,
                      u_range: tuple[float, float] = (0.2, 0.8), q_floor: float = 0.2) -> pd.DataFrame:
    """n frames per clip, evenly spaced in u among quality-passing frames (with fallbacks)."""
    f = frames[frames["clip_uid"].isin(set(clips["clip_uid"]))]
    lo, hi = u_range
    rows = []
    for uid, g in f.groupby("clip_uid", sort=True):
        g = g.sort_values("u")
        inr = g[(g["u"] >= lo) & (g["u"] <= hi) & g["layout_ok"]]
        tiers = (
            inr[(inr["quality"] >= q_floor) & (inr["sharp_ratio_to_clip_median"] >= 0.4) & inr["dup_of"].isna()],
            inr[(inr["quality"] >= q_floor) & (inr["sharp_ratio_to_clip_median"] >= 0.4)],
            inr,
            g,
        )
        cand = next((t for t in tiers if len(t) >= n), max(tiers, key=len))
        if len(cand) == 0:
            continue
        pos = np.unique(np.round(np.linspace(0, len(cand) - 1, min(n, len(cand)))).astype(int))
        rows.append(cand.iloc[pos])
    out = pd.concat(rows).reset_index(drop=True)
    out["layout"] = out["clip_uid"].map(dict(zip(clips["clip_uid"], clips["layout"])))
    return out


def embed_frames(candidate_id: str, table: pd.DataFrame, device: str, batch: int = 64, workers: int = 4,
                 size: int = 224, cache_root: Path = CACHE_ROOT) -> np.ndarray:
    """(n, D) float32 frozen embeddings of the listed cached frames, cached on disk."""
    import torch
    from torch.utils.data import DataLoader

    key = hashlib.sha256(("|".join(table["clip_uid"] + ":" + table["src_idx"].astype(str))
                          + f"|{size}").encode()).hexdigest()[:16]
    path = EMB_DIR / f"stageA_{candidate_id}_{key}.npz"
    if path.exists():
        return np.load(path)["emb"]
    net = load_pretrained(candidate_id, allow_probe_only=True).to(device).eval()
    loader = DataLoader(D.EvalFrames(table, D.ImageSpec(size, True, cache_root)), batch_size=batch,
                        num_workers=workers, persistent_workers=False)
    out = np.zeros((len(table), get_candidate(candidate_id).embed_dim), np.float32)
    with torch.inference_mode():
        for x, idx in loader:
            e = net.backbone(x.to(device)).float().cpu().numpy()
            out[idx.numpy()] = e
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, emb=out)
    return out


def clip_embeddings(frame_emb: np.ndarray, frame_clip: np.ndarray, n_clips: int) -> np.ndarray:
    """Mean of L2-normalised frame embeddings per clip."""
    e = frame_emb / np.maximum(np.linalg.norm(frame_emb, axis=1, keepdims=True), 1e-12)
    out = np.zeros((n_clips, e.shape[1]), np.float64)
    np.add.at(out, frame_clip, e)
    cnt = np.bincount(frame_clip, minlength=n_clips)[:, None]
    return (out / np.maximum(cnt, 1)).astype(np.float32)


# -- probes ---------------------------------------------------------------------------------------

def _lr(c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def lr_probe(x: np.ndarray, y: np.ndarray, folds: np.ndarray, c_grid=C_GRID) -> tuple[np.ndarray, list[float]]:
    """OOF class predictions; C per outer fold chosen on its inner-val fold."""
    pred = np.full(len(y), -1)
    chosen = []
    for k in range(N_FOLDS):
        inner = (k + 1) % N_FOLDS
        tr_in = (folds != k) & (folds != inner)
        va_in = folds == inner
        scores = []
        for c in c_grid:
            m = _lr(c).fit(x[tr_in], y[tr_in])
            scores.append(macro_f1(y[va_in], m.predict(x[va_in])))
        best = c_grid[int(np.argmax(scores))]
        chosen.append(best)
        m = _lr(best).fit(x[folds != k], y[folds != k])
        pred[folds == k] = m.predict(x[folds == k])
    return pred, chosen


def knn_probe(x: np.ndarray, y: np.ndarray, folds: np.ndarray, k: int = KNN_K) -> np.ndarray:
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    pred = np.full(len(y), -1)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        sims = xn[te] @ xn[tr].T
        nn = np.argsort(-sims, axis=1)[:, :k]
        ytr = y[tr]
        for i, row in enumerate(nn):
            votes = np.bincount(ytr[row], minlength=NUM_STATIONS).astype(float)
            votes += 1e-3 * np.bincount(ytr[row], weights=sims[i, row], minlength=NUM_STATIONS)
            pred[np.flatnonzero(te)[i]] = int(np.argmax(votes))
    return pred


def decodability(x: np.ndarray, target: np.ndarray, station: np.ndarray, folds: np.ndarray,
                 min_per_class: int = 5) -> dict:
    """Balanced accuracy of a linear probe for a binary/categorical confound (grouped CV)."""
    ok = target >= 0
    if len(np.unique(target[ok])) < 2:
        return {"bacc": None, "bacc_within_station": None, "n": int(ok.sum())}
    pred = np.full(len(target), -1)
    for k in range(N_FOLDS):
        tr, te = ok & (folds != k), ok & (folds == k)
        if len(np.unique(target[tr])) < 2 or not te.any():
            continue
        pred[te] = _lr(1.0).fit(x[tr], target[tr]).predict(x[te])
    has = ok & (pred >= 0)
    n_cls = int(target[ok].max()) + 1
    overall = balanced_accuracy(target[has], pred[has], n_cls)
    within = []
    for s in np.unique(station[has]):
        sel = has & (station == s)
        counts = np.bincount(target[sel], minlength=n_cls)
        if (counts >= min_per_class).sum() >= 2:
            within.append(balanced_accuracy(target[sel], pred[sel], n_cls))
    return {"bacc": overall, "bacc_within_station": float(np.mean(within)) if within else None,
            "n": int(has.sum()), "stations_used": len(within)}


def _ci(y, pred, groups, fn, seed=0):
    return cluster_bootstrap(lambda i: fn(y[i], pred[i]), groups, n_boot=2000, seed=seed)


def probe_candidate(candidate_id: str, m: pd.DataFrame, frames: pd.DataFrame, device: str,
                    workers: int = 4, batch: int = 64) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    clips = m[(m["split"] == "dev") & (m["exclude_reason"] == "") & ~m["eval_exclude_b"]
              & (m["cohort"] == "new_1920")].sort_values("clip_uid").reset_index(drop=True)
    table = pick_probe_frames(frames, clips)
    idx = pd.Series(np.arange(len(clips)), index=clips["clip_uid"])
    frame_clip = idx.loc[table["clip_uid"]].to_numpy()
    emb = embed_frames(candidate_id, table, device, batch=batch, workers=workers)
    t_embed = time.time() - t0
    have = np.bincount(frame_clip, minlength=len(clips)) > 0
    x = clip_embeddings(emb, frame_clip, len(clips))[have]
    c = clips[have].reset_index(drop=True)
    y = c["label"].map(STATION_INDEX).to_numpy()
    folds = c["fold_i"].to_numpy()
    pats = c["patient_id"].to_numpy()

    lr_pred, cs = lr_probe(x, y, folds)
    knn_pred = knn_probe(x, y, folds)
    res = {"candidate": candidate_id, "arch": get_candidate(candidate_id).arch or "probe-only",
           "weights": get_candidate(candidate_id).weights, "license": get_candidate(candidate_id).license,
           "embed_dim": int(x.shape[1]), "n_clips": int(len(c)), "n_patients": int(len(set(pats))),
           "n_frames": int(len(table)), "lr_C_per_fold": cs}
    for name, pred in (("lr", lr_pred), ("knn", knn_pred)):
        for mname, fn in (("macro_f1", macro_f1), ("bal_acc", balanced_accuracy)):
            p, lo, hi = _ci(y, pred, pats, fn)
            res[f"{name}_{mname}"], res[f"{name}_{mname}_lo"], res[f"{name}_{mname}_hi"] = p, lo, hi

    station = y
    mode = c["mode_major"].map({"wl": 0, "nbi": 1}).fillna(-1).astype(int).to_numpy()
    cap = c["cap_b"].astype(int).to_numpy()
    rooms = {h: i for i, h in enumerate(c["room_hash"].value_counts().index)}
    room = c["room_hash"].map(rooms).fillna(-1).astype(int).to_numpy()
    for name, target in (("nbi", mode), ("cap", cap), ("room", room)):
        d = decodability(x, target, station, folds)
        res[f"{name}_decode_bacc"] = d["bacc"]
        res[f"{name}_decode_bacc_within_station"] = d["bacc_within_station"]

    _, _, f1 = prf_from_cm(confusion(y, lr_pred))
    per_class = pd.DataFrame({"candidate": candidate_id, "station": STATION_ORDER,
                              "n_clips": np.bincount(y, minlength=NUM_STATIONS),
                              "lr_f1": np.round(f1, 4)})
    res["seconds_embed"] = round(t_embed, 1)
    res["seconds_total"] = round(time.time() - t0, 1)
    res["device"] = device
    return res, per_class


CSV_COLUMNS = [
    "candidate", "arch", "license", "embed_dim", "n_clips", "n_patients", "n_frames",
    "lr_macro_f1", "lr_macro_f1_lo", "lr_macro_f1_hi", "lr_bal_acc", "lr_bal_acc_lo", "lr_bal_acc_hi",
    "knn_macro_f1", "knn_macro_f1_lo", "knn_macro_f1_hi", "knn_bal_acc", "knn_bal_acc_lo", "knn_bal_acc_hi",
    "nbi_decode_bacc", "nbi_decode_bacc_within_station", "cap_decode_bacc", "cap_decode_bacc_within_station",
    "room_decode_bacc", "room_decode_bacc_within_station", "device", "seconds_embed",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "stageA_v1.csv")
    a = ap.parse_args(argv)
    device = resolve_device(a.device)
    m = D.load_manifest()
    frames = D.load_frames()
    rows, per_class = [], []
    for cid in [c for c in a.candidates.split(",") if c]:
        if cid not in ALLOWLIST:
            print(f"skip {cid}: not allowlisted", file=sys.stderr)
            continue
        res, pc = probe_candidate(cid, m, frames, device, a.workers, a.batch)
        write_json(RUNS_ROOT / "stageA" / f"{cid}.json", res)
        rows.append(res)
        per_class.append(pc)
        print(f"{cid:15s} LR F1 {res['lr_macro_f1']:.3f} [{res['lr_macro_f1_lo']:.3f},{res['lr_macro_f1_hi']:.3f}]"
              f"  kNN F1 {res['knn_macro_f1']:.3f}  NBI|station {res['nbi_decode_bacc_within_station']}"
              f"  ({res['seconds_total']}s)", flush=True)
    if not rows:
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[CSV_COLUMNS]
    if a.out.exists():  # keep other candidates' rows from earlier runs
        old = pd.read_csv(a.out)
        df = pd.concat([old[~old["candidate"].isin(df["candidate"])], df])
    df.round(4).sort_values("lr_macro_f1", ascending=False).to_csv(a.out, index=False)
    pc_path = a.out.with_name(a.out.stem + "_per_class.csv")
    pc = pd.concat(per_class)
    if pc_path.exists():
        old = pd.read_csv(pc_path)
        pc = pd.concat([old[~old["candidate"].isin(pc["candidate"])], pc])
    pc.to_csv(pc_path, index=False)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
