"""Clip/frame selection, hierarchical sampling and torch datasets for the landmark model.

Inputs are the private manifest (``data/landmarks_private/manifest_v1.csv``) and
the masked frame cache (``cache/v1/frames.csv.gz`` + JPEG canvases). See
``README.md`` "Formats" for the columns.

Training frames (``train_frame_table``)
    usable dev clips only (``exclude_reason`` empty, new cohort); per frame:
    u in [0.1, 0.9], ``sharp_ratio_to_clip_median >= 0.4``, not a freeze
    duplicate (``dup_of`` empty), not gated by darkness/saturation/red-out,
    verified layout, and outside any OSD-time overlap with a different-label
    clip of the same patient (``overlap_head_s`` / ``overlap_tail_s``).
    Contested Incisura/LC clips train on their soft 0.5/0.5 target.

Sampling (``ClipSampler``), per class c (uniform over present classes):
    patient ~ min(n_clips(p, c), patient_cap)          (per-patient cap)
    clip    ~ t_c(clip) * frozen_weight? / near-dup group size
    frame   ~ 0.2 + 0.8 * (1 - |2u - 1|)               (triangular, centre)
    then frame masses are re-weighted by imaging mode within each class so
    the effective NBI share moves toward [0.3, 0.7], each weight <= 3x.
    ``mode_report`` gives the natural and achieved shares per class.
    A batch = ``clips_per_batch`` clips x ``k_frames`` frames of each clip.

Evaluation frames (``eval_frame_table``) are EVERY cached frame of the
requested clips, unfiltered, with u and the raw quality metrics, so the
quality gate, u-ranges and ambiguous segments can be applied downstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from app.ml.landmarks.layouts import LAYOUTS_BY_NAME
from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER
from training.landmarks.common import CACHE_ROOT, MANIFEST_PATH

# Clips with an exclude_reason that still belong to a patient's stream (they are
# real footage): replayed as ambiguous segments, never trained or scored on.
REPLAY_ONLY_REASONS = frozenset({"combo", "quarantine_suffix_mismatch"})
NBI_MODES = frozenset({"nbi", "rdi"})


@dataclass(frozen=True)
class DataConfig:
    k_frames: int = 8
    clips_per_batch: int = 16
    u_range: tuple[float, float] = (0.1, 0.9)
    min_sharp_ratio: float = 0.4
    drop_dups: bool = True
    drop_overlap: bool = True
    drop_gated: bool = True  # dark / saturated / red-out frames (quality gate zero)
    patient_cap: float = 4.0
    nbi_share: tuple[float, float] = (0.3, 0.7)
    mode_weight_cap: float = 3.0
    mode_balance: bool = True
    frozen_weight: float = 0.25
    permute_labels: bool = False  # G-PIPELINE sanity run only
    exclude_stations: tuple[str, ...] = ()  # leave-station-out runs only
    max_clips: int | None = None  # --smoke

    @classmethod
    def from_dict(cls, d: dict | None) -> "DataConfig":
        d = dict(d or {})
        for k in ("u_range", "nbi_share", "exclude_stations"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


# -- tables ------------------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("rel_t_s", "dur_s", "overlap_head_s", "overlap_tail_s", "nbi_frac", "fps"):
        m[col + "_f"] = pd.to_numeric(m[col], errors="coerce")
    m["fold_i"] = pd.to_numeric(m["fold"], errors="coerce").fillna(-1).astype(int)
    m["cap_b"] = m["cap"].str.lower().eq("true")
    m["frozen_b"] = m["frozen"].str.lower().eq("true")
    m["eval_exclude_b"] = m["eval_exclude"].str.lower().eq("true")
    return m


def load_frames(cache_root: Path = CACHE_ROOT) -> pd.DataFrame:
    f = pd.read_csv(cache_root / "frames.csv.gz", dtype={"clip_uid": str, "mode_frame": str, "dhash": str})
    f["layout_ok"] = f["layout_ok"].astype(str).str.lower().eq("true")
    f["src_idx"] = f["src_idx"].astype(int)
    return f


def clip_targets(clips: pd.DataFrame) -> np.ndarray:
    """(n, 10) training targets: the soft label when set, else one-hot ``label``."""
    t = np.zeros((len(clips), NUM_STATIONS), np.float32)
    for i, (label, soft) in enumerate(zip(clips["label"], clips["soft_label"])):
        if soft:
            for part in soft.split("|"):
                k, w = part.split(":")
                t[i, STATION_INDEX[k]] = float(w)
        else:
            t[i, STATION_INDEX[label]] = 1.0
    return t / t.sum(1, keepdims=True)


def station_set(row) -> frozenset[str]:
    """Stations a clip may show: >1 for contested, combo and quarantined clips."""
    if row["soft_label"]:
        return frozenset(p.split(":")[0] for p in row["soft_label"].split("|"))
    out = {row["label"]} if row["label"] else set()
    if row["exclude_reason"] in REPLAY_ONLY_REASONS:
        from training.landmarks.ids import parse_clip_name

        out.add(row["folder_class"])
        parsed = parse_clip_name(Path(row["rel_path"]).name)
        out.update(s for s in parsed.suffix_stations if s)
    return frozenset(out)


def is_ambiguous(row) -> bool:
    return len(station_set(row)) > 1 or row["exclude_reason"] in REPLAY_ONLY_REASONS


def fold_patients(m: pd.DataFrame, fold: int, n_folds: int = 5) -> tuple[list[str], list[str], list[str]]:
    """(inner_train, inner_val, outer_val) dev patients for outer fold ``fold``.

    Inner-val is the next fold ((fold + 1) % 5): patient-grouped and
    stratified by folds.py, and never the outer fold.
    """
    dev = m[(m["split"] == "dev") & (m["fold_i"] >= 0)]
    by_fold = {k: sorted(set(dev.loc[dev["fold_i"] == k, "patient_id"])) for k in range(n_folds)}
    inner = (fold + 1) % n_folds
    train = sorted({p for k, ps in by_fold.items() if k not in (fold, inner) for p in ps})
    return train, by_fold[inner], by_fold[fold]


def dev_patients(m: pd.DataFrame) -> list[str]:
    return sorted(set(m.loc[(m["split"] == "dev") & (m["fold_i"] >= 0), "patient_id"]))


def training_clips(m: pd.DataFrame, patients: Iterable[str], cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Usable new-cohort clips of ``patients`` (soft-labelled contested clips included)."""
    ps = set(patients)
    c = m[m["patient_id"].isin(ps) & (m["exclude_reason"] == "") & (m["cohort"] == "new_1920")
          & m["split"].isin(["dev"])].copy()
    if cfg.exclude_stations:
        t = clip_targets(c)
        drop = t[:, [STATION_INDEX[s] for s in cfg.exclude_stations]].sum(1) > 0
        c = c[~drop].copy()
    return c.reset_index(drop=True)


def stream_clips(m: pd.DataFrame, patients: Iterable[str]) -> pd.DataFrame:
    """Every clip that belongs in these patients' replay streams (usable + ambiguous)."""
    ps = set(patients)
    ok = (m["exclude_reason"] == "") | m["exclude_reason"].isin(REPLAY_ONLY_REASONS)
    c = m[m["patient_id"].isin(ps) & ok & m["split"].isin(["dev", "test"])]
    return c.reset_index(drop=True)


def recompute_quality(frames: pd.DataFrame, params: QualityParams) -> np.ndarray:
    """``quality.quality_score`` from the cached raw metrics, for any params."""
    gated = ((frames["dark_frac"] >= params.dark_max) | (frames["sat_frac"] >= params.sat_max)
             | (frames["redout"] >= params.redout_max)).to_numpy()
    q = np.minimum(1.0, frames["sharpness"].to_numpy(np.float64) / max(params.sharp_ref, 1e-6))
    return np.where(gated, 0.0, q)


def triangular(u: np.ndarray) -> np.ndarray:
    return 0.2 + 0.8 * (1.0 - np.abs(2.0 * np.asarray(u, np.float64) - 1.0))


def train_frame_table(frames: pd.DataFrame, clips: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Eligible training frames of ``clips`` (one row per frame, with ``clip_i``)."""
    ci = pd.Series(np.arange(len(clips)), index=clips["clip_uid"].to_numpy())
    f = frames[frames["clip_uid"].isin(ci.index)].copy()
    f["clip_i"] = ci.loc[f["clip_uid"]].to_numpy()
    lo, hi = cfg.u_range
    keep = (f["u"] >= lo) & (f["u"] <= hi) & f["layout_ok"]
    keep &= f["sharp_ratio_to_clip_median"] >= cfg.min_sharp_ratio
    if cfg.drop_dups:
        keep &= f["dup_of"].isna()
    if cfg.drop_gated:
        qp = QualityParams()
        keep &= (f["dark_frac"] < qp.dark_max) & (f["sat_frac"] < qp.sat_max) & (f["redout"] < qp.redout_max)
    if cfg.drop_overlap:
        c = clips.iloc[f["clip_i"].to_numpy()]
        head = c["overlap_head_s_f"].fillna(0).to_numpy()
        tail = c["overlap_tail_s_f"].fillna(0).to_numpy()
        dur = c["dur_s_f"].fillna(np.inf).to_numpy()
        t = f["t_s"].to_numpy()
        keep &= (t >= head) & (t <= dur - tail)
    f = f[keep].copy()
    f["tri"] = triangular(f["u"].to_numpy())
    f["nbi"] = f["mode_frame"].isin(NBI_MODES).to_numpy()
    return f.reset_index(drop=True)


# -- hierarchical sampler -------------------------------------------------------------------------

@dataclass
class ModeBalance:
    natural_share: float
    achieved_share: float
    w_wl: float
    w_nbi: float
    n_frames: int


def _mode_weights(s: float, lo: float, hi: float, cap: float) -> tuple[float, float]:
    """(w_wl, w_nbi) moving NBI share s toward [lo, hi]; each weight in [1, cap]."""
    if s > hi and s < 1.0:
        return min(cap, s * (1 - hi) / (hi * (1 - s))), 1.0
    if 0.0 < s < lo:
        return 1.0, min(cap, lo * (1 - s) / (s * (1 - lo)))
    return 1.0, 1.0


class ClipSampler:
    """Deterministic class -> patient -> clip -> frame sampler over training frames."""

    def __init__(self, clips: pd.DataFrame, frames: pd.DataFrame, targets: np.ndarray, cfg: DataConfig):
        self.cfg = cfg
        self.clips = clips
        self.frames = frames
        self.targets = targets
        n = len(clips)
        has_frames = np.bincount(frames["clip_i"].to_numpy(), minlength=n) > 0
        self.n_clips_without_frames = int((~has_frames).sum())

        # clip-level weight within (patient, class)
        w_clip = np.where(clips["frozen_b"].to_numpy(), cfg.frozen_weight, 1.0)
        nd = clips["near_dup_group"].replace("", np.nan)
        nd_size = nd.map(nd.value_counts()).fillna(1.0).to_numpy()
        w_clip = w_clip / nd_size * has_frames

        patients, p_idx = np.unique(clips["patient_id"].to_numpy(), return_inverse=True)
        t = targets * (has_frames[:, None])
        a = t * w_clip[:, None]  # (n, C) clip weight per class
        n_eff = np.zeros((len(patients), NUM_STATIONS))
        a_sum = np.zeros((len(patients), NUM_STATIONS))
        np.add.at(n_eff, p_idx, t)
        np.add.at(a_sum, p_idx, a)
        w_pat = np.minimum(n_eff, cfg.patient_cap)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_pat = w_pat / w_pat.sum(0, keepdims=True)  # P(p | c)
            p_clip = a / a_sum[p_idx]  # P(clip | p, c)
        p_pat = np.nan_to_num(p_pat)
        p_clip = np.nan_to_num(p_clip)
        present = w_pat.sum(0) > 0
        self.present_classes = [STATION_ORDER[i] for i in np.flatnonzero(present)]
        p_class = present / max(present.sum(), 1)
        clip_class = p_class[None, :] * p_pat[p_idx] * p_clip  # (n, C)

        # frame masses per class: clip mass spread by the triangular density
        fi = frames["clip_i"].to_numpy()
        tri = frames["tri"].to_numpy()
        tri_sum = np.bincount(fi, weights=tri, minlength=n)
        base = clip_class[fi] * (tri / tri_sum[fi])[:, None]  # (F, C)
        nbi = frames["nbi"].to_numpy()
        self.mode_report: dict[str, ModeBalance] = {}
        lo, hi = cfg.nbi_share
        for c in np.flatnonzero(present):
            col = base[:, c]
            tot = col.sum()
            if tot <= 0:
                continue
            s = float(col[nbi].sum() / tot)
            w_wl, w_nbi = _mode_weights(s, lo, hi, cfg.mode_weight_cap) if cfg.mode_balance else (1.0, 1.0)
            col = col * np.where(nbi, w_nbi, w_wl)
            col *= tot / col.sum()
            base[:, c] = col
            self.mode_report[STATION_ORDER[c]] = ModeBalance(
                s, float(col[nbi].sum() / col.sum()), w_wl, w_nbi, int((col > 0).sum()))
        self.frame_mass = base.sum(1)
        self.clip_prob = np.bincount(fi, weights=self.frame_mass, minlength=n)
        self.clip_prob /= self.clip_prob.sum()
        order = np.argsort(fi, kind="stable")
        self._frame_order = order
        self._starts = np.searchsorted(fi[order], np.arange(n + 1))

    def patient_share(self) -> dict[str, float]:
        """Sampling share per patient (for checking the per-patient cap)."""
        s = pd.Series(self.clip_prob, index=self.clips["patient_id"].to_numpy())
        return s.groupby(level=0).sum().to_dict()

    def schedule(self, n_steps: int, seed: int) -> np.ndarray:
        """(n_steps, clips_per_batch * k_frames) frame row indices, deterministic in seed."""
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        out = np.empty((n_steps, cfg.clips_per_batch * cfg.k_frames), np.int64)
        for s in range(n_steps):
            clips = rng.choice(len(self.clip_prob), cfg.clips_per_batch, replace=True, p=self.clip_prob)
            row = []
            for c in clips:
                idx = self._frame_order[self._starts[c]:self._starts[c + 1]]
                w = self.frame_mass[idx]
                w = w / w.sum()
                replace = len(idx) < cfg.k_frames
                row.append(rng.choice(idx, cfg.k_frames, replace=replace, p=w))
            out[s] = np.concatenate(row)
        return out


def permute_targets(clips: pd.DataFrame, targets: np.ndarray, seed: int) -> np.ndarray:
    """G-PIPELINE: shuffle clip targets among the training clips (fixed seed)."""
    rng = np.random.default_rng(seed + 7919)
    return targets[rng.permutation(len(targets))]


# -- images and torch datasets -----------------------------------------------------------------

def read_canvas(cache_root: Path, clip_uid: str, src_idx: int) -> np.ndarray:
    path = cache_root / "frames" / clip_uid / f"{int(src_idx):05d}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def eval_frame_table(frames: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    """Every cached frame of ``clips`` (clip order, then src_idx), with the clip's layout."""
    f = frames[frames["clip_uid"].isin(set(clips["clip_uid"]))].copy()
    lay = dict(zip(clips["clip_uid"], clips["layout"]))
    f["layout"] = f["clip_uid"].map(lay)
    return f.sort_values(["clip_uid", "src_idx"]).reset_index(drop=True)


@dataclass
class ImageSpec:
    """How a cached canvas becomes a model input."""

    input_size: int = 224
    masked: bool = True  # False only for the sanity positive control (unmasked cache)
    cache_root: Path = field(default=CACHE_ROOT)


class TrainFrames(Dataset):
    """Item i of a precomputed schedule -> (augmented CHW tensor, soft target)."""

    def __init__(self, frames: pd.DataFrame, clips: pd.DataFrame, targets: np.ndarray, schedule: np.ndarray,
                 spec: ImageSpec, aug_cfg, seed: int):
        self.uids = frames["clip_uid"].to_numpy()
        self.src = frames["src_idx"].to_numpy()
        self.clip_i = frames["clip_i"].to_numpy()
        self.layouts = clips["layout"].to_numpy()
        self.targets = targets
        self.flat = schedule.reshape(-1)
        self.spec = spec
        self.aug_cfg = aug_cfg
        self.seed = seed

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, i: int):
        import torch

        from training.landmarks.augment import augment_to_tensor

        r = self.flat[i]
        c = self.clip_i[r]
        canvas = read_canvas(self.spec.cache_root, self.uids[r], self.src[r])
        rng = np.random.default_rng([self.seed, int(i)])
        x = augment_to_tensor(canvas, LAYOUTS_BY_NAME[self.layouts[c]], self.aug_cfg, rng,
                              self.spec.input_size, masked=self.spec.masked)
        return x, torch.from_numpy(self.targets[c])


class EvalFrames(Dataset):
    """Row i of an eval frame table -> (CHW tensor, i). No augmentation."""

    def __init__(self, table: pd.DataFrame, spec: ImageSpec):
        self.uids = table["clip_uid"].to_numpy()
        self.src = table["src_idx"].to_numpy()
        self.layouts = table["layout"].to_numpy()
        self.spec = spec

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        from training.landmarks.augment import eval_to_tensor

        canvas = read_canvas(self.spec.cache_root, self.uids[i], self.src[i])
        return eval_to_tensor(canvas, LAYOUTS_BY_NAME[self.layouts[i]], self.spec.input_size,
                              masked=self.spec.masked), i
