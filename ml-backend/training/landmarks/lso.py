"""Leave-station-out (LSO) comparison of rejection scores: MSP vs energy vs kNN (M3, GPU box).

    python -m training.landmarks.lso --config stageB_<winner> \
        [--stations lesser_curvature_retroflex,z_line,duodenum_descending] [--device cuda]

There is no "other" class, so rejection is judged by hiding a real station:
for each held-out station s, fold 0 is retrained WITHOUT s's clips
(``data.exclude_stations``). Out-of-distribution = frames of s's clips in
fold 0 (plus, with ``--transit``, synthetic transit frames); in-distribution
= quality-passing u in [0.1, 0.9] frames of the other stations' fold-0 clips.

Scores (higher = more in-distribution):
    msp     max softmax(logits / T)          (T fit on the run's inner-val, s excluded)
    energy  T * logsumexp(logits / T)
    knn     -cosine distance to the 10th nearest neighbour in a class-balanced
            bank of <= 1500 training-frame embeddings per class

Reported: AUROC and FPR at 95% TPR per score and station. MSP ships unless
kNN or energy beats it by >= 0.05 AUROC AND lowers S4 FDR in the replay
benchmark (the second condition is checked with stream_eval, not here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.landmarks.stations import STATION_INDEX
from training.landmarks import dataset as D
from training.landmarks.calibrate import fit_temperature, load_predset
from training.landmarks.run_utils import RUNS_ROOT, write_json

DEFAULT_STATIONS = ("lesser_curvature_retroflex", "z_line", "duodenum_descending")
BANK_PER_CLASS = 1500
KNN_K = 10


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    s = np.concatenate([pos, neg])
    ranks = pd.Series(s).rank().to_numpy()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fpr_at_tpr(pos: np.ndarray, neg: np.ndarray, tpr: float = 0.95) -> float:
    """Share of OOD (neg) scores above the threshold that keeps ``tpr`` of ID (pos)."""
    thr = np.quantile(pos, 1 - tpr)
    return float((neg >= thr).mean())


def energy(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / t
    mx = z.max(1, keepdims=True)
    return t * (mx[:, 0] + np.log(np.exp(z - mx).sum(1)))


def msp(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / t
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return (p / p.sum(1, keepdims=True)).max(1)


def knn_score(bank: np.ndarray, query: np.ndarray, k: int = KNN_K, chunk: int = 2048) -> np.ndarray:
    b = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
    out = np.zeros(len(query))
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk].astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        sims = q @ b.T
        kth = np.partition(-sims, k - 1, axis=1)[:, k - 1]
        out[s:s + chunk] = -(1.0 + kth)  # -(1 - cos) of the k-th neighbour
    return out


def bank_embeddings(net, m: pd.DataFrame, frames: pd.DataFrame, patients, exclude: str, spec, device: str,
                    per_class: int = BANK_PER_CLASS, seed: int = 0) -> np.ndarray:
    from training.landmarks.train import predict_table

    cfg = D.DataConfig(exclude_stations=(exclude,))
    clips = D.training_clips(m, patients, cfg)
    t = D.train_frame_table(frames, clips, cfg)
    t["y"] = clips["label"].map(STATION_INDEX).to_numpy()[t["clip_i"].to_numpy()]
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.choice(g.index.to_numpy(), min(per_class, len(g)), replace=False)
                           for _, g in t.groupby("y")])
    table = t.loc[np.sort(pick)].copy()
    table["layout"] = clips["layout"].to_numpy()[table["clip_i"].to_numpy()]
    _, emb = predict_table(net, table.reset_index(drop=True), spec, device)
    return emb.astype(np.float32)


def evaluate_station(run_dir: Path, station: str, m: pd.DataFrame, device: str = "cpu") -> dict:
    import torch

    from app.ml.landmarks.architectures import build
    from training.landmarks.backbones import get_candidate
    from training.landmarks.run_utils import read_json

    s = STATION_INDEX[station]
    oof = load_predset(run_dir / "oof_fold0.npz", m, with_emb=True)
    inner = load_predset(run_dir / "inner_fold0.npz", m)
    fi = inner.frames
    tm = (fi["primary"] & (fi["y"] != s)).to_numpy()
    t = fit_temperature(inner.logits[tm], fi["y"].to_numpy()[tm])
    f = oof.frames
    base = (f["primary"] & f["layout_ok"] & f["u"].between(0.1, 0.9)).to_numpy()
    id_m = base & (f["y"] != s).to_numpy()
    ood_m = (f["y"] == s).to_numpy() & f["layout_ok"].to_numpy()

    cfg = read_json(run_dir / "config.json")
    met = read_json(run_dir / "metrics.json")
    cand = cfg["candidate"]
    net = build(cand.split(":", 1)[1] if cand.startswith("scratch:") else get_candidate(cand).arch)
    net.load_state_dict(torch.load(run_dir / f"best_{met['best']['variant']}.pt", map_location="cpu",
                                   weights_only=True))
    net.to(device)
    train_p, _, _ = D.fold_patients(m, 0)
    spec = D.ImageSpec(int(cfg["input"]["size"]))
    bank = bank_embeddings(net, m, D.load_frames(), train_p, station, spec, device)
    scores = {
        "msp": msp(oof.logits, t),
        "energy": energy(oof.logits, t),
        "knn": knn_score(bank, oof.emb.astype(np.float32)),
    }
    return {"station": station, "T": t, "n_id": int(id_m.sum()), "n_ood": int(ood_m.sum()),
            **{name: {"auroc": auroc(v[id_m], v[ood_m]), "fpr_at_95tpr": fpr_at_tpr(v[id_m], v[ood_m])}
               for name, v in scores.items()}}


def main(argv: list[str] | None = None) -> int:
    from training.landmarks.train import resolve_config, train

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing LSO runs")
    a = ap.parse_args(argv)
    m = D.load_manifest()
    results = []
    for station in a.stations.split(","):
        cfg = resolve_config(a.config, {"data": {"exclude_stations": [station]}})
        exp = RUNS_ROOT / f"lso_{cfg['name']}_s{cfg['seed']}"
        run = exp / f"without_{station}" / "fold0"
        if not a.skip_train:
            train(cfg, "fold", run, fold=0, device=a.device)
        from training.landmarks.run_utils import resolve_device

        results.append(evaluate_station(run, station, m, resolve_device(a.device)))
        print(results[-1])
    summary = {k: {"auroc_mean": float(np.mean([r[k]["auroc"] for r in results])),
                   "fpr95_mean": float(np.mean([r[k]["fpr_at_95tpr"] for r in results]))}
               for k in ("msp", "energy", "knn")}
    best_alt = max(("energy", "knn"), key=lambda k: summary[k]["auroc_mean"])
    decision = best_alt if summary[best_alt]["auroc_mean"] >= summary["msp"]["auroc_mean"] + 0.05 else "msp"
    write_json(exp / "lso.json", {"stations": results, "summary": summary,
                                  "candidate_decision": decision,
                                  "note": "a non-MSP scorer is adopted only if it ALSO lowers S4 FDR"})
    print(summary, "->", decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
