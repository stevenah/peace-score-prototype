"""Procedure Replay Benchmark (PRB): per-patient 2 Hz streams from held-out predictions.

No full-length recording is used: a patient's "procedure" is rebuilt from
their own station clips.

- Clips are ordered by ``rel_t_s`` (OSD time); clips without a time are
  inserted at seeded random positions.
- Each clip is replayed at 2 Hz (every 5th frame of the 10 fps cache) with a
  random phase, INCLUDING the non-diagnostic clip ends; ``drop_p`` of the
  frames are dropped (client backpressure: they never reach the server).
- The real gap between consecutive clips (capped at 30 s; ``unknown_gap_s``
  when a time is missing) is filled per scenario:

    S0   nothing (clips abut)
    S2   synthetic transit frames made from the patient's OWN frames: motion
         blur, defocus, red-out, bile, bubbles, glare, occlusion, close-up,
         exposure extremes. A transit frame is made only from a clip that was
         ALREADY REPLAYED in this stream (so it can never preview a station
         before that station's clip, and in S4 it never comes from a removed
         clip). They must be classified by the fold's model
         (``fill_transit(..., hook=model_hook(net))``); OOF-only runs use a
         placeholder that is always "uncertain" and are labelled as such
         (``transit_source = "placeholder_uniform"``).
    S4   S2 with every clip of station k removed (its time becomes transit):
         any observation of k is a false discovery (per-station FDR). The
         removed clips are not transit sources either.
    S5   S2 with the clip order shuffled (no hidden order prior).
    hz=1 the ``landmark_every_n=2`` fallback.

- Contested, combo and quarantined clips are AMBIGUOUS segments: observing any
  of their stations is not a false positive, and they never create a
  "present" station (excluded from recall denominators).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd

from app.ml.landmarks.layouts import LAYOUTS_BY_NAME, Layout, aperture_mask
from app.ml.landmarks.preprocess import normalize, to_model_input
from app.ml.landmarks.quality import QUALITY_SIZE, QualityParams, quality_metrics, quality_score
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX
from training.landmarks import dataset as D
from training.landmarks.calibrate import PredSet, quality_of
from training.landmarks.common import CACHE_ROOT
from training.landmarks.run_utils import fp32_inference, softmax

SCENARIOS = ("S0", "S2", "S4", "S5")
TRANSIT_KINDS = ("motion_blur", "defocus", "redout", "bile", "bubbles", "glare", "occlusion",
                 "closeup", "under_exposed", "over_exposed", "near_black")
PLACEHOLDER = "placeholder_uniform"


@dataclass(frozen=True)
class StreamConfig:
    scenario: str = "S2"
    hz: float = 2.0
    cache_fps: float = 10.0
    gap_cap_s: float = 30.0
    unknown_gap_s: float = 5.0
    drop_p: float = 0.05
    seed: int = 0
    remove_station: str | None = None  # S4


@dataclass(frozen=True)
class ClipSeg:
    clip_uid: str
    stations: frozenset[str]
    ambiguous: bool
    t0: float
    t1: float


@dataclass
class Stream:
    patient_id: str
    fold: int
    cfg: StreamConfig
    t: np.ndarray  # (n,) seconds since stream start
    kind: np.ndarray  # (n,) "clip" | "transit"
    clip_pos: np.ndarray  # (n,) index into ``clips``; -1 for transit
    frame_row: np.ndarray  # (n,) row of the PredSet frames; -1 for transit
    logits: np.ndarray  # (n, 10) uncalibrated (transit rows: 0 until classified)
    probs: np.ndarray  # (n, 10) calibrated: softmax(logits / T)
    quality: np.ndarray  # (n,)
    layout_ok: np.ndarray  # (n,) bool
    clips: list[ClipSeg]
    present: frozenset[str]  # stations with >= 1 non-ambiguous clip in this stream
    allowed: frozenset[str]  # present plus the stations of ambiguous clips
    transit_src: list[tuple[str, int, str, int]] = field(default_factory=list)  # (uid, src, kind, seed)
    transit_source: str = "none"

    def __len__(self) -> int:
        return len(self.t)

    @property
    def transit_idx(self) -> np.ndarray:
        return np.flatnonzero(self.kind == "transit")

    def with_temperature(self, T: float) -> "Stream":
        """The same stream with probabilities recalibrated at temperature ``T`` (shared arrays otherwise)."""
        return replace(self, probs=softmax(self.logits, T))


def clip_table(ps: PredSet) -> pd.DataFrame:
    """One row per clip of a PredSet with its station set and ambiguity."""
    f = ps.frames
    c = f.drop_duplicates("clip_uid").set_index("clip_uid")
    c = c[["patient_id", "label", "soft_label", "exclude_reason", "rel_t_s_f", "dur_s_f", "rel_path",
           "folder_class"]].copy()
    rows = c.reset_index()
    c["stations"] = [D.station_set(r) for _, r in rows.iterrows()]
    c["ambiguous"] = [D.is_ambiguous(r) for _, r in rows.iterrows()]
    return c


def _stable_int(text: str) -> int:
    """Process-independent seed component (``hash(str)`` is salted per process)."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _order(clips: pd.DataFrame, rng: np.random.Generator, shuffle: bool) -> list[str]:
    known = clips[clips["rel_t_s_f"].notna()].sort_values("rel_t_s_f").index.tolist()
    for uid in sorted(clips[clips["rel_t_s_f"].isna()].index):
        known.insert(int(rng.integers(0, len(known) + 1)), uid)
    if shuffle:
        known = [known[i] for i in rng.permutation(len(known))]
    return known


def _gaps(order: list[str], clips: pd.DataFrame, cfg: StreamConfig) -> list[float]:
    """Gap before clip i+1 (s), real where both times are known, capped."""
    out = []
    for a, b in zip(order[:-1], order[1:]):
        ta, tb = clips.at[a, "rel_t_s_f"], clips.at[b, "rel_t_s_f"]
        da = clips.at[a, "dur_s_f"]
        g = tb - (ta + da) if np.isfinite(ta) and np.isfinite(tb) and np.isfinite(da) else cfg.unknown_gap_s
        out.append(float(np.clip(g, 0.0, cfg.gap_cap_s)))
    return out


def build_stream(ps: PredSet, patient_id: str, cfg: StreamConfig, T: float, params: QualityParams,
                 clips: pd.DataFrame | None = None) -> Stream:
    """Skeleton + clip-frame predictions for one patient; transit rows still empty."""
    rng = np.random.default_rng([cfg.seed, _stable_int(patient_id), 17])
    clips = clip_table(ps) if clips is None else clips
    pc = clips[clips["patient_id"] == patient_id]
    chrono = _order(pc, rng, shuffle=False)
    gaps_chrono = _gaps(chrono, pc, cfg)
    if cfg.scenario == "S4" and cfg.remove_station:
        keep = [u for u in chrono if cfg.remove_station not in pc.at[u, "stations"]]
        order, gaps = keep, _gaps(keep, pc, cfg)
    elif cfg.scenario == "S5":
        order = [chrono[i] for i in rng.permutation(len(chrono))]
        gaps = gaps_chrono[:max(len(order) - 1, 0)]
    else:
        order, gaps = chrono, gaps_chrono

    f = ps.frames
    rows_by_clip = {u: g.sort_values("src_idx").index.to_numpy() for u, g in f[f["patient_id"] == patient_id]
                    .groupby("clip_uid")}
    step = max(1, int(round(cfg.cache_fps / cfg.hz)))
    dt = 1.0 / cfg.hz
    fill = cfg.scenario != "S0"
    replayed: list[np.ndarray] = []  # frames of the clips replayed so far: the only transit sources

    t_list, kind, pos, frow, tsrc = [], [], [], [], []
    segs: list[ClipSeg] = []
    t = 0.0
    for i, uid in enumerate(order):
        rows = rows_by_clip.get(uid, np.zeros(0, int))
        if len(rows):
            replayed.append(rows)
            phase = int(rng.integers(0, step))
            sub = rows[phase::step]
            ts = f.loc[sub, "t_s"].to_numpy(float)
            t0 = t
            for r, tt in zip(sub, ts):
                if rng.random() >= cfg.drop_p:
                    t_list.append(t0 + tt)
                    kind.append("clip")
                    pos.append(len(segs))
                    frow.append(r)
            dur = pc.at[uid, "dur_s_f"]
            t = t0 + (float(dur) if np.isfinite(dur) else (ts.max() if len(ts) else 0.0))
            segs.append(ClipSeg(uid, pc.at[uid, "stations"], bool(pc.at[uid, "ambiguous"]), t0, t))
        if i < len(order) - 1 and fill:
            g = gaps[i] if i < len(gaps) else cfg.unknown_gap_s
            n = int(np.floor(g * cfg.hz))
            sources = np.concatenate(replayed) if replayed else np.zeros(0, int)
            for j in range(n):
                if rng.random() >= cfg.drop_p and len(sources):
                    src = int(sources[rng.integers(0, len(sources))])
                    t_list.append(t + (j + 1) * dt)
                    kind.append("transit")
                    pos.append(-1)
                    frow.append(-1)
                    tsrc.append((f.at[src, "clip_uid"], int(f.at[src, "src_idx"]),
                                 TRANSIT_KINDS[int(rng.integers(0, len(TRANSIT_KINDS)))],
                                 int(rng.integers(0, 2**31))))
            t += g

    frow_a = np.asarray(frow, int)
    n = len(frow_a)
    logits = np.zeros((n, NUM_STATIONS))
    quality = np.zeros(n)
    layout_ok = np.ones(n, bool)
    is_clip = frow_a >= 0
    if is_clip.any():
        sel = frow_a[is_clip]
        pos_in_ps = f.index.get_indexer(sel)
        logits[is_clip] = ps.logits[pos_in_ps]
        quality[is_clip] = quality_of(f.loc[sel], params)
        layout_ok[is_clip] = f.loc[sel, "layout_ok"].to_numpy(bool)
    present = frozenset(s for c in segs if not c.ambiguous for s in c.stations)
    allowed = present | frozenset(s for c in segs if c.ambiguous for s in c.stations)
    return Stream(patient_id, ps.fold, cfg, np.asarray(t_list, float), np.asarray(kind, object),
                  np.asarray(pos, int), frow_a, logits, softmax(logits, T), quality, layout_ok, segs, present,
                  allowed, tsrc, "none" if not tsrc else PLACEHOLDER)


# -- synthetic transit frames ----------------------------------------------------------------------------

def _motion_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    return k / max(k.sum(), 1e-6)


def synth_transit(canvas: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """A degraded, non-diagnostic view derived from a real canvas (uint8 320x320)."""
    img = canvas.astype(np.float32)
    h, w = img.shape[:2]
    if kind == "motion_blur":
        img = cv2.filter2D(img, -1, _motion_kernel(int(rng.integers(4, 11)) * 2 + 1, rng.uniform(0, 180)))
    elif kind == "defocus":
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(3, 8))
    elif kind == "redout":
        a = rng.uniform(0.7, 0.95)
        red = np.array([rng.uniform(180, 240), rng.uniform(10, 40), rng.uniform(10, 40)], np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 4) * (1 - a) + red * a
    elif kind == "bile":
        a = rng.uniform(0.5, 0.8)
        bile = np.array([rng.uniform(130, 170), rng.uniform(150, 190), rng.uniform(20, 60)], np.float32)
        img = img * (1 - a) + bile * a
    elif kind == "bubbles":
        for _ in range(int(rng.integers(15, 60))):
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            r = int(rng.integers(4, 30))
            cv2.circle(img, c, r, (235, 235, 235), 1 + int(r > 15))
            cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(1, r // 5), (255, 255, 255), -1)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
    elif kind == "glare":
        for _ in range(int(rng.integers(1, 4))):
            c = (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4)))
            axes = (int(rng.integers(20, 80)), int(rng.integers(10, 50)))
            cv2.ellipse(img, c, axes, rng.uniform(0, 180), 0, 360, (255, 255, 255), -1)
        img = cv2.GaussianBlur(img, (0, 0), 3)
    elif kind == "occlusion":
        m = np.zeros((h, w), np.float32)
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(m, c, int(rng.uniform(0.35, 0.6) * w), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 12)[..., None]
        tone = np.array([rng.uniform(60, 120), rng.uniform(40, 80), rng.uniform(20, 50)], np.float32)
        img = img * (1 - m) + tone * m
    elif kind == "closeup":
        s = rng.uniform(0.2, 0.35)
        cw, ch = int(w * s), int(h * s)
        x0 = int(rng.integers(w // 4, 3 * w // 4 - cw))
        y0 = int(rng.integers(h // 4, 3 * h // 4 - ch))
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (0, 0), 2.5)
    elif kind == "under_exposed":
        img = img * 0.15
    elif kind == "over_exposed":
        img = img * 2.5
    elif kind == "near_black":
        img = img * 0.03
    else:
        raise ValueError(kind)
    return np.clip(img, 0, 255).astype(np.uint8)


def model_hook(net, device: str = "cpu", size: int = 224, batch: int = 64) -> Callable:
    """hook(canvases, layouts) -> fp32 logits (n, 10) through a fold model (no TF32)."""
    import torch

    def hook(canvases: Sequence[np.ndarray], layouts: Sequence[Layout]) -> np.ndarray:
        net.eval()
        out = []
        with torch.inference_mode(), fp32_inference():
            for s in range(0, len(canvases), batch):
                x = torch.stack([normalize(to_model_input(c, lay, size))
                                 for c, lay in zip(canvases[s:s + batch], layouts[s:s + batch])])
                out.append(net(x.to(device).float())[0].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, NUM_STATIONS), np.float32)

    return hook


def fill_transit(streams: Sequence[Stream], T: float | dict[int, float], params: QualityParams,
                 hook: Callable | dict[int, Callable] | None = None, cache_root=CACHE_ROOT,
                 layout_by_clip: dict[str, str] | None = None) -> None:
    """Classify the synthetic transit frames in place.

    With ``hook`` (a model hook, or {fold: hook}) the degraded frames are
    generated, run through the model and quality-scored for real. Without
    it, transit frames stay uniform (always "uncertain") with quality 0 and
    the streams are labelled ``placeholder_uniform``.
    """
    for s in streams:
        idx = s.transit_idx
        if not len(idx):
            continue
        if hook is None:
            s.logits[idx] = 0.0
            s.probs[idx] = 1.0 / NUM_STATIONS
            s.quality[idx] = 0.0
            s.transit_source = PLACEHOLDER
            continue
        h = hook[s.fold] if isinstance(hook, dict) else hook
        t = T[s.fold] if isinstance(T, dict) else T
        canvases, layouts, qual = [], [], []
        for uid, src, kind, seed in s.transit_src:
            lay = LAYOUTS_BY_NAME[(layout_by_clip or {}).get(uid, "olympus_1920x1080")]
            c = synth_transit(D.read_canvas(cache_root, uid, src), kind, np.random.default_rng(seed))
            c[~aperture_mask(lay)] = 0
            canvases.append(c)
            layouts.append(lay)
            q_img = to_model_input(c, lay, QUALITY_SIZE)  # quality at its calibration size, as serving
            qual.append(quality_score(quality_metrics(q_img, lay, params), params))
        s.logits[idx] = h(canvases, layouts)
        s.probs[idx] = softmax(s.logits[idx], t)
        s.quality[idx] = np.asarray(qual)
        s.transit_source = "model"


def build_streams(ps: PredSet, cfg: StreamConfig, T: float, params: QualityParams,
                  patients: Sequence[str] | None = None) -> list[Stream]:
    """S0/S2/S5 streams (one per patient), or S4 streams (one per patient x present station)."""
    clips = clip_table(ps)
    patients = sorted(set(clips["patient_id"])) if patients is None else patients
    out = []
    for p in patients:
        if cfg.scenario == "S4":
            base = build_stream(ps, p, replace(cfg, scenario="S2"), T, params, clips)
            for k in sorted(base.allowed, key=STATION_INDEX.get):
                out.append(build_stream(ps, p, replace(cfg, remove_station=k), T, params, clips))
        else:
            out.append(build_stream(ps, p, cfg, T, params, clips))
    return out
