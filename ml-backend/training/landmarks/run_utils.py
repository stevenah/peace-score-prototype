"""Run hygiene and small statistics shared by the M2-M4 training/eval tools.

- Run directories: ``ml-backend/runs/<exp_id>/{fold0..4,all_dev,...}/`` (gitignored).
  Every run records its config, git sha, manifest/cache sha256 and versions.
- Devices: CUDA on the GPU box, MPS/CPU on the laptop (smoke runs only).
- Statistics: confusion-matrix metrics and cluster bootstrap CIs. Everything
  that leaves a run directory (``results/*.csv``) is aggregate numbers only.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from app.ml.landmarks.layouts import LAYOUTS_VERSION
from app.ml.landmarks.preprocess import PREPROCESS_VERSION
from app.ml.landmarks.stations import NUM_STATIONS
from training.landmarks.common import CACHE_ROOT, MANIFEST_PATH, ML_BACKEND, REPO_ROOT, sha256_file

RUNS_ROOT = ML_BACKEND / "runs"
RESULTS_DIR = ML_BACKEND / "training" / "landmarks" / "results"
CONFIGS_DIR = ML_BACKEND / "training" / "landmarks" / "configs"
GATES_PATH = ML_BACKEND / "training" / "landmarks" / "gates.json"


# -- run directories and provenance -------------------------------------------------------

def git_sha() -> tuple[str, bool]:
    """(HEAD sha, working tree dirty?) or ("unknown", True) outside git."""
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "ml-backend"],
                                    capture_output=True, text=True, timeout=30).stdout.strip())
        return sha or "unknown", dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def cache_fingerprint(cache_root: Path = CACHE_ROOT) -> dict:
    info_path = cache_root / "CACHE_INFO.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return {
        "cache_root": cache_root.name,
        "clip_set_sha256": info.get("clip_set_sha256"),
        "frames_csv_sha256": info.get("frames_csv_sha256"),
        "cache_preprocess_version": info.get("preprocess_version"),
        "cache_layouts_version": info.get("layouts_version"),
    }


def provenance(manifest_path: Path = MANIFEST_PATH, cache_root: Path = CACHE_ROOT, device: str = "") -> dict:
    import torch

    sha, dirty = git_sha()
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        **cache_fingerprint(cache_root),
        "preprocess_version": PREPROCESS_VERSION,
        "layouts_version": LAYOUTS_VERSION,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "tf32_training": tf32_state(),
        "inference_precision": "fp32 ieee: TF32 off (fp32_inference) for every OOF/inner/test/transit logit",
        "python": platform.python_version(),
        "host": platform.platform(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_json(path: Path, obj) -> None:
    """Strict JSON (no NaN) written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def to_jsonable(obj):
    """numpy scalars/arrays -> Python; NaN/inf -> None (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_config(path: str | os.PathLike) -> dict:
    p = Path(path)
    if not p.exists() and (CONFIGS_DIR / p).exists():
        p = CONFIGS_DIR / p
    if not p.exists() and (CONFIGS_DIR / f"{p}.json").exists():
        p = CONFIGS_DIR / f"{p}.json"
    return json.loads(p.read_text())


def deep_update(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def fold_dirs(exp_dir: Path) -> dict[int, Path]:
    """{fold: run dir} for an experiment's completed outer-fold runs."""
    out = {}
    for d in sorted(Path(exp_dir).glob("fold[0-9]")):
        if (d / "metrics.json").exists():
            out[int(d.name[4:])] = d
    return out


# -- devices and seeds ----------------------------------------------------------------------

def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tf32_flags():
    """(cudnn conv, cuda matmul) flag holders of the new ``fp32_precision`` API, or None (older torch)."""
    import torch

    conv = getattr(torch.backends.cudnn, "conv", None)
    mm = torch.backends.cuda.matmul
    if hasattr(conv, "fp32_precision") and hasattr(mm, "fp32_precision"):
        return conv, mm
    return None


def tf32_state() -> dict:
    """Current TF32 settings (recorded in provenance.json)."""
    import torch

    flags = _tf32_flags()
    if flags:
        return {"cudnn_conv": flags[0].fp32_precision, "cuda_matmul": flags[1].fp32_precision}
    return {"cudnn_conv": "tf32" if torch.backends.cudnn.allow_tf32 else "ieee",
            "cuda_matmul": "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"}


@contextmanager
def fp32_inference() -> Iterator[None]:
    """True fp32 convolutions and matmuls for logits that feed T, tau, q_min or a gate.

    PyTorch enables TF32 cuDNN convolutions by default on Ampere+ GPUs (10-bit
    mantissa); serving is CPU fp32, so frames near a tau threshold could flip
    between ok and uncertain. Training keeps the default; the flags are
    restored on exit. A no-op on CPU/MPS.
    """
    import torch

    flags = _tf32_flags()
    if flags:
        conv, mm = flags
        saved = (conv.fp32_precision, mm.fp32_precision)
        conv.fp32_precision, mm.fp32_precision = "ieee", "ieee"
        try:
            yield
        finally:
            conv.fp32_precision, mm.fp32_precision = saved
        return
    saved_legacy = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved_legacy


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


# -- metrics ---------------------------------------------------------------------------------

def confusion(y: np.ndarray, yhat: np.ndarray, n: int = NUM_STATIONS, w: np.ndarray | None = None) -> np.ndarray:
    """(n, n) confusion matrix, rows = truth; optional sample weights."""
    cm = np.zeros((n, n), np.float64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(yhat, int)), 1.0 if w is None else np.asarray(w, float))
    return cm


def prf_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall, F1 (NaN where undefined)."""
    tp = np.diag(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / cm.sum(0)
        rec = tp / cm.sum(1)
        f1 = 2 * prec * rec / (prec + rec)
    f1 = np.where((cm.sum(1) > 0) & np.isnan(f1), 0.0, f1)  # present class, never right -> 0
    return prec, rec, f1


def macro_f1(y, yhat, n: int = NUM_STATIONS, w=None) -> float:
    """Macro-F1 over the classes present in ``y``."""
    cm = confusion(y, yhat, n, w)
    _, _, f1 = prf_from_cm(cm)
    present = cm.sum(1) > 0
    return float(np.nanmean(f1[present])) if present.any() else float("nan")


def balanced_accuracy(y, yhat, n: int = NUM_STATIONS, w=None) -> float:
    cm = confusion(y, yhat, n, w)
    _, rec, _ = prf_from_cm(cm)
    present = cm.sum(1) > 0
    return float(np.nanmean(rec[present])) if present.any() else float("nan")


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    groups: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point, lo, hi): resample whole clusters (patients or groups) with replacement.

    ``stat(idx)`` receives row indices (with repeats) into the arrays it closes
    over. Resamples where the statistic is undefined (NaN) are dropped.
    """
    groups = np.asarray(groups)
    point = float(stat(np.arange(len(groups))))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        v = stat(np.concatenate([members[g] for g in pick]))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_cluster_bootstrap(
    stat_a: Callable[[np.ndarray], float],
    stat_b: Callable[[np.ndarray], float],
    groups: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI of stat_a - stat_b over the same cluster resamples (model comparisons)."""
    return cluster_bootstrap(lambda i: stat_a(i) - stat_b(i), groups, n_boot, seed)


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def log_softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / t
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))
