"""Stage B fine-tuning of a landmark candidate (M2/M4). Config-driven, GPU box.

    python -m training.landmarks.train --config stageB_regy32_in1k --fold 0      # outer fold 0
    python -m training.landmarks.train --config stageB_regy32_in1k --all-dev      # final model
    python -m training.landmarks.train --config stageB_effb0_in1k --smoke         # laptop, MPS/CPU

Modes
    --fold k    train on the dev folds other than k and (k+1)%5, early-stop on
                inner-val (fold (k+1)%5) NLL, then write fp32 OOF predictions
                for EVERY cached frame of fold k's patients (oof_fold{k}.npz)
                and of the inner-val patients (inner_fold{k}.npz, for T).
    --all-dev   all dev patients, the same schedule, stopped at a fixed step:
                ``optim.stop_step`` or the median best step of an experiment's
                fold runs (``--steps-from runs/<exp>``). No validation.
    --holdout F patients with period "early" in CSV F train, "late" ones are
                the OOF set (temporal hold-out; see evaluate.py temporal-split).
    --smoke     20 clips, a few steps, tiny batches, MPS/CPU; everything else
                identical (checks the whole path end to end).

Recipe (configs/*.json override ``DEFAULTS``): hierarchical sampler
(dataset.py), augmentation with the mask last (augment.py), head = dropout
0.2 + the model's Linear(10), soft-target cross-entropy with label smoothing
0.1, AdamW (CNN: backbone 1e-4 / head 1e-3, wd 0.05; ViT: 5e-5 with layer
decay 0.75), linear warmup + cosine over OPTIMIZER STEPS, bf16 autocast on
CUDA only, EMA 0.995 with warmup over parameters AND buffers (BN statistics).
Raw and EMA weights are both evaluated; the variant/step with the lowest
inner-val NLL is kept. OOF logits are computed in fp32 without autocast and
without TF32 (``run_utils.fp32_inference``), like CPU serving.

Outputs ``runs/<exp>/<fold{k}|all_dev|holdout|smoke>/``: config.json,
provenance.json (git sha, manifest + cache sha256, torch), metrics.json,
best_raw.pt / best_ema.pt (or final_*.pt), oof_fold{k}.npz, inner_fold{k}.npz.

Prediction npz format (``write_predictions``): logits f32 (n,10), emb f16
(n,D), clip_uid, src_idx i32, u f32, t_s f32, sharpness/dark_frac/sat_frac/
redout/sharp_ratio f32, dup bool, mode_frame, layout_ok bool, plus scalars
variant, fold, input_size, candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.landmarks.layouts import LAYOUTS_BY_NAME
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX
from training.landmarks import dataset as D
from training.landmarks.augment import AugmentConfig, assert_batch_masked
from training.landmarks.common import CACHE_ROOT, MANIFEST_PATH, ML_BACKEND
from training.landmarks.run_utils import (
    RUNS_ROOT,
    deep_update,
    fold_dirs,
    fp32_inference,
    load_config,
    log_softmax,
    macro_f1,
    provenance,
    read_json,
    resolve_device,
    seed_everything,
    write_json,
)

DEFAULTS: dict = {
    "name": "unnamed",
    "candidate": "regy32_in1k",
    "seed": 0,
    "input": {"size": 224, "masked": True, "cache_root": None},
    "data": D.DataConfig().to_dict(),
    "augment": AugmentConfig().to_dict(),
    "optim": {
        "steps": 2000, "warmup_steps": 100, "stop_step": None,
        "lr_backbone": 1e-4, "lr_head": 1e-3, "weight_decay": 0.05, "layer_decay": None,
        "label_smoothing": 0.1, "dropout": 0.2, "grad_clip": 1.0, "min_lr_ratio": 0.0,
    },
    "ema": {"decay": 0.995, "warmup": True},
    "eval": {"every_steps": 100, "patience": 6, "frame_stride": 2, "alt_variant_oof": True, "batch": 128},
    "loader": {"workers": 8},
    "sanity": False,
}

SMOKE = {
    "data": {"k_frames": 2, "clips_per_batch": 4, "max_clips": 20},
    "optim": {"steps": 6, "warmup_steps": 2},
    "eval": {"every_steps": 3, "patience": 10, "frame_stride": 4, "batch": 32},
    "loader": {"workers": 0},
}


def _load_with_base(name: str, depth: int = 0) -> dict:
    """A config may name ``"base": "<other config>"`` (ablations); it is applied first."""
    cfg = load_config(name)
    base = cfg.pop("base", None)
    if base and depth < 4:
        cfg = deep_update(_load_with_base(base, depth + 1), cfg)
    return cfg


def resolve_config(path_or_name: str | None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path_or_name:
        cfg = deep_update(cfg, _load_with_base(path_or_name))
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


# -- model, optimiser, schedule, EMA -------------------------------------------------------------

def build_model(candidate: str):
    """Allowlisted pretrained weights, or ``scratch:<arch>`` (random init; tests)."""
    if candidate.startswith("scratch:"):
        from app.ml.landmarks.architectures import build

        return build(candidate.split(":", 1)[1]).train()
    from training.landmarks.backbones import load_pretrained

    return load_pretrained(candidate).train()


def _no_decay(name: str, p) -> bool:
    return p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name \
        or "reg_token" in name


def param_groups(net, optim: dict) -> list[dict]:
    """AdamW groups: backbone/head learning rates, ViT layer decay, no wd on norms/biases."""
    wd = optim["weight_decay"]
    groups: dict[tuple, dict] = {}

    def add(name, p, lr):
        key = (lr, _no_decay(name, p))
        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": 0.0 if key[1] else wd})
        g["params"].append(p)

    decay = optim.get("layer_decay")
    blocks = getattr(net.backbone, "blocks", None)
    n_layers = len(blocks) if blocks is not None else 0
    for name, p in net.backbone.named_parameters():
        if not p.requires_grad:
            continue
        lr = optim["lr_backbone"]
        if decay and n_layers:
            if name.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token")):
                layer = 0
            elif name.startswith("blocks."):
                layer = int(name.split(".")[1]) + 1
            else:
                layer = n_layers + 1
            lr = lr * decay ** (n_layers + 1 - layer)
        add(name, p, lr)
    for name, p in net.head.named_parameters():
        add("head." + name, p, optim["lr_head"])
    return list(groups.values())


def lr_factor(step: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    """Linear warmup then cosine decay, in optimizer steps."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


class Ema:
    """Exponential moving average of ALL state (params + buffers), with warmup."""

    def __init__(self, model, decay: float = 0.995, warmup: bool = True):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.updates = decay, warmup, 0

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    def update(self, model) -> None:
        import torch

        self.updates += 1
        d = self.current_decay()
        with torch.no_grad():
            src = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src[k].detach(), alpha=1 - d)
                else:
                    v.copy_(src[k])


def soft_cross_entropy(logits, targets, smoothing: float = 0.1):
    import torch

    n = logits.shape[-1]
    t = targets * (1 - smoothing) + smoothing / n
    return -(t * torch.log_softmax(logits.float(), -1)).sum(-1).mean()


def state_fp32(model) -> dict:
    return {k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu().clone()
            for k, v in model.state_dict().items()}


# -- prediction ------------------------------------------------------------------------------------

def predict_table(net, table: pd.DataFrame, spec: D.ImageSpec, device: str, batch: int = 128,
                  workers: int = 4, with_emb: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """fp32 logits (and embeddings) for every row of an eval frame table. No autocast, no TF32."""
    import torch
    from torch.utils.data import DataLoader

    net.eval()
    loader = DataLoader(D.EvalFrames(table, spec), batch_size=batch, num_workers=workers)
    logits = np.zeros((len(table), NUM_STATIONS), np.float32)
    emb = np.zeros((len(table), net.embedding_dim), np.float16) if with_emb else None
    with torch.inference_mode(), fp32_inference():
        for x, idx in loader:
            lo, e = net(x.to(device).float())
            logits[idx.numpy()] = lo.float().cpu().numpy()
            if with_emb:
                emb[idx.numpy()] = e.float().cpu().numpy().astype(np.float16)
    return logits, emb


def write_predictions(path: Path, table: pd.DataFrame, logits: np.ndarray, emb: np.ndarray | None,
                      **scalars) -> None:
    arrays = {
        "logits": logits.astype(np.float32),
        "clip_uid": table["clip_uid"].to_numpy().astype("U32"),
        "src_idx": table["src_idx"].to_numpy(np.int32),
        "u": table["u"].to_numpy(np.float32),
        "t_s": table["t_s"].to_numpy(np.float32),
        "sharpness": table["sharpness"].to_numpy(np.float32),
        "dark_frac": table["dark_frac"].to_numpy(np.float32),
        "sat_frac": table["sat_frac"].to_numpy(np.float32),
        "redout": table["redout"].to_numpy(np.float32),
        "sharp_ratio": table["sharp_ratio_to_clip_median"].to_numpy(np.float32),
        "dup": table["dup_of"].notna().to_numpy(),
        "mode_frame": table["mode_frame"].fillna("").to_numpy().astype("U4"),
        "layout_ok": table["layout_ok"].to_numpy(bool),
    }
    if emb is not None:
        arrays["emb"] = emb.astype(np.float16)
    arrays.update({k: np.asarray(v) for k, v in scalars.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_predictions(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def predictions_frame(pred: dict) -> pd.DataFrame:
    """Prediction npz -> per-frame DataFrame (without logits/emb)."""
    cols = ["clip_uid", "src_idx", "u", "t_s", "sharpness", "dark_frac", "sat_frac", "redout",
            "sharp_ratio", "dup", "mode_frame", "layout_ok"]
    return pd.DataFrame({c: pred[c] for c in cols})


# -- validation metrics ------------------------------------------------------------------------------

def validation_config(run: D.DataConfig) -> D.DataConfig:
    """Inner-val (early stopping) selection: the default frame filters, the run's station set.

    Frame filters stay at their defaults so every ablation is selected on the
    same inner-val population; ``exclude_stations`` is inherited so a
    leave-station-out run never early-stops on its held-out (OOD) station.
    Label permutation never applies to validation.
    """
    return D.DataConfig(exclude_stations=run.exclude_stations)


def early_stop_table(frames: pd.DataFrame, clips: pd.DataFrame, cfg: D.DataConfig, stride: int) -> pd.DataFrame:
    """Inner-val frames filtered like training frames, every ``stride``-th per clip."""
    t = D.train_frame_table(frames, clips, cfg)
    t = t[t.groupby("clip_i").cumcount() % stride == 0].copy()
    t["layout"] = clips["layout"].to_numpy()[t["clip_i"].to_numpy()]
    return t.reset_index(drop=True)


def val_metrics(logits: np.ndarray, table: pd.DataFrame, clips: pd.DataFrame, targets: np.ndarray) -> dict:
    """Frame NLL (soft targets), frame accuracy and clip macro-F1 on primary clips."""
    lp = log_softmax(logits)
    t = targets[table["clip_i"].to_numpy()]
    nll = float(-(t * lp).sum(1).mean())
    y = t.argmax(1)
    acc = float((lp.argmax(1) == y).mean())
    df = pd.DataFrame(lp)
    df["clip_i"] = table["clip_i"].to_numpy()
    clip_lp = df.groupby("clip_i").mean()
    primary = ~clips["eval_exclude_b"].to_numpy()[clip_lp.index]
    yc = targets[clip_lp.index].argmax(1)[primary]
    pc = clip_lp.to_numpy().argmax(1)[primary]
    return {"nll": nll, "frame_acc": acc, "clip_macro_f1": macro_f1(yc, pc) if len(yc) else float("nan"),
            "n_frames": int(len(table)), "n_clips": int(primary.sum())}


# -- the training loop ------------------------------------------------------------------------------

def _smoke_clips(clips: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per = max(1, n // NUM_STATIONS)
    pick = []
    for _, g in clips[clips["soft_label"] == ""].groupby("label"):
        pick.extend(rng.choice(g.index.to_numpy(), min(per, len(g)), replace=False))
    return clips.loc[sorted(pick)[:n]].reset_index(drop=True)


def _read_holdout(path: Path) -> dict[str, str]:
    h = pd.read_csv(path, dtype=str)
    return dict(zip(h["patient_id"], h["period"]))


def split_patients(m: pd.DataFrame, mode: str, fold: int | None, holdout: Path | None, seed: int):
    """(train, inner_val, outer_val) patient lists for a mode."""
    if mode in ("fold", "smoke"):
        return D.fold_patients(m, 0 if fold is None else fold)
    if mode == "all_dev":
        return D.dev_patients(m), [], []
    if mode == "holdout":
        period = _read_holdout(holdout)
        dev = D.dev_patients(m)
        early = sorted(p for p in dev if period.get(p) == "early")
        late = sorted(p for p in dev if period.get(p) == "late")
        rng = np.random.default_rng(seed)
        inner = sorted(rng.choice(early, max(1, len(early) // 5), replace=False))
        return sorted(set(early) - set(inner)), inner, late
    raise ValueError(mode)


def fold_selection(exp_dir: Path) -> tuple[int, str]:
    """(median best step, majority selected variant) of an experiment's fold runs."""
    best = [read_json(d / "metrics.json")["best"] for d in fold_dirs(exp_dir).values()]
    if not best:
        raise SystemExit(f"no completed fold runs under {exp_dir}")
    variants = [b["variant"] or "ema" for b in best]
    return int(np.median([b["step"] for b in best])), max(sorted(set(variants)), key=variants.count)


def train(cfg: dict, mode: str, out_dir: Path, fold: int | None = None, device: str = "auto",
          manifest_path: Path = MANIFEST_PATH, holdout: Path | None = None, log=print) -> dict:
    import torch

    device = resolve_device(device)
    seed = int(cfg["seed"])
    seed_everything(seed)
    dcfg = D.DataConfig.from_dict(cfg["data"])
    acfg = AugmentConfig.from_dict(cfg["augment"])
    optim_cfg = cfg["optim"]
    inp = cfg["input"]
    if not inp.get("masked", True) and not cfg.get("sanity"):
        raise ValueError("unmasked inputs are only allowed in sanity runs (positive control)")
    cache_root = Path(inp["cache_root"]) if inp.get("cache_root") else CACHE_ROOT
    if not cache_root.is_absolute():
        cache_root = ML_BACKEND / cache_root
    spec = D.ImageSpec(int(inp["size"]), bool(inp.get("masked", True)), cache_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", {**cfg, "mode": mode, "fold": fold})
    write_json(out_dir / "provenance.json", provenance(manifest_path, cache_root, device))

    m = D.load_manifest(manifest_path)
    frames = D.load_frames(cache_root)
    train_p, inner_p, outer_p = split_patients(m, mode, fold, holdout, seed)
    clips = D.training_clips(m, train_p, dcfg)
    if mode == "smoke":
        clips = _smoke_clips(clips, dcfg.max_clips or 20, seed)
    targets = D.clip_targets(clips)
    if dcfg.permute_labels:
        targets = D.permute_targets(clips, targets, seed)
    tframes = D.train_frame_table(frames, clips, dcfg)
    sampler = D.ClipSampler(clips, tframes, targets, dcfg)

    total = int(optim_cfg["steps"])
    stop = total
    if mode == "all_dev":
        stop = int(optim_cfg.get("stop_step") or total)
    schedule = sampler.schedule(stop, seed)
    ds = D.TrainFrames(tframes, clips, targets, schedule, spec, acfg, seed)
    from torch.utils.data import DataLoader

    workers = int(cfg["loader"]["workers"])
    loader = DataLoader(ds, batch_size=schedule.shape[1], shuffle=False, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=device == "cuda",
                        prefetch_factor=4 if workers > 0 else None)

    net = build_model(cfg["candidate"]).to(device)
    ema = Ema(net, cfg["ema"]["decay"], cfg["ema"]["warmup"])
    opt = torch.optim.AdamW(param_groups(net, optim_cfg), betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _null()

    has_val = mode in ("fold", "smoke", "holdout") and bool(inner_p)
    if has_val:
        vcfg = validation_config(dcfg)
        val_clips = D.training_clips(m, inner_p, vcfg)
        if mode == "smoke":
            val_clips = _smoke_clips(val_clips, 10, seed + 1)
        val_targets = D.clip_targets(val_clips)
        val_table = early_stop_table(frames, val_clips, vcfg, int(cfg["eval"]["frame_stride"]))

    layout = LAYOUTS_BY_NAME[clips["layout"].iloc[0]]
    history, best = [], {"nll": float("inf"), "variant": None, "step": 0}
    best_nll = {"raw": float("inf"), "ema": float("inf")}
    since_best, run_loss, run_n = 0, 0.0, 0
    t0 = time.time()
    log(f"[{mode}] {cfg['name']} fold={fold} device={device} clips={len(clips)} frames={len(tframes)} "
        f"steps={stop}/{total} batch={schedule.shape[1]}")
    step = 0
    for step, (x, t) in enumerate(loader):
        if step < 2 and spec.masked:
            assert_batch_masked(x, layout, acfg.mask)
        f = lr_factor(step, int(optim_cfg["warmup_steps"]), total, float(optim_cfg.get("min_lr_ratio", 0)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        net.train()
        x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
        with autocast:
            emb = net.backbone(x)
            logits = net.head(torch.nn.functional.dropout(emb, float(optim_cfg["dropout"]), training=True))
        loss = soft_cross_entropy(logits, t, float(optim_cfg["label_smoothing"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if optim_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(optim_cfg["grad_clip"]))
        opt.step()
        ema.update(net)
        run_loss += float(loss.detach())
        run_n += 1

        last = step + 1 == stop
        if has_val and ((step + 1) % int(cfg["eval"]["every_steps"]) == 0 or last):
            rec = {"step": step + 1, "lr_factor": f, "train_loss": run_loss / max(run_n, 1),
                   "ema_decay": ema.current_decay(), "seconds": round(time.time() - t0, 1)}
            run_loss, run_n = 0.0, 0
            improved = False
            for variant, model in (("raw", net), ("ema", ema.module)):
                lo, _ = predict_table(model, val_table, spec, device, int(cfg["eval"]["batch"]), workers,
                                      with_emb=False)
                vm = val_metrics(lo, val_table, val_clips, val_targets)
                rec[variant] = vm
                if vm["nll"] < best_nll[variant]:  # each variant keeps its own best checkpoint
                    best_nll[variant] = vm["nll"]
                    torch.save(state_fp32(model), out_dir / f"best_{variant}.pt")
                if vm["nll"] < best["nll"] - 1e-4:  # the selection: lowest inner-val NLL overall
                    best = {"nll": vm["nll"], "variant": variant, "step": step + 1, **vm}
                    improved = True
            history.append(rec)
            since_best = 0 if improved else since_best + 1
            log(f"  step {step + 1:5d} loss {rec['train_loss']:.3f} raw nll {rec['raw']['nll']:.3f} "
                f"f1 {rec['raw']['clip_macro_f1']:.3f} | ema nll {rec['ema']['nll']:.3f} "
                f"f1 {rec['ema']['clip_macro_f1']:.3f} | best {best['variant']}@{best['step']}")
            if since_best >= int(cfg["eval"]["patience"]):
                log(f"  early stop at step {step + 1}")
                break
        if last:
            break

    metrics = {"mode": mode, "fold": fold, "candidate": cfg["candidate"], "steps_run": step + 1,
               "total_steps": total, "n_train_clips": int(len(clips)), "n_train_frames": int(len(tframes)),
               "n_train_patients": int(clips["patient_id"].nunique()),
               "clips_without_frames": sampler.n_clips_without_frames,
               "mode_balance": {k: vars(v) for k, v in sampler.mode_report.items()},
               "history": history, "seconds": round(time.time() - t0, 1)}
    if not has_val:
        torch.save(state_fp32(net), out_dir / "final_raw.pt")
        torch.save(state_fp32(ema.module), out_dir / "final_ema.pt")
        metrics["best"] = {"variant": cfg.get("final_variant", "ema"), "step": step + 1}
        write_json(out_dir / "metrics.json", metrics)
        return metrics

    metrics["best"] = best
    sel = best["variant"] or "ema"
    other = "raw" if sel == "ema" else "ema"
    net.load_state_dict(torch.load(out_dir / f"best_{sel}.pt", map_location=device, weights_only=True))
    k = 0 if fold is None else fold
    eval_batch, scal = int(cfg["eval"]["batch"]), {"fold": k, "input_size": spec.input_size,
                                                  "candidate": cfg["candidate"]}
    for kind, patients in (("oof", outer_p), ("inner", inner_p)):
        sc = D.stream_clips(m, patients)
        if mode == "smoke":
            sc = sc[sc["clip_uid"].isin(set(_smoke_clips(D.training_clips(m, patients), 10, seed + 2)["clip_uid"]))]
        table = D.eval_frame_table(frames, sc)
        lo, em = predict_table(net, table, spec, device, eval_batch, workers)
        write_predictions(out_dir / f"{kind}_fold{k}.npz", table, lo, em, variant=sel, **scal)
        if kind == "oof":
            metrics["oof"] = oof_summary(lo, table, m)
            if cfg["eval"].get("alt_variant_oof"):
                alt = copy.deepcopy(net)
                alt.load_state_dict(torch.load(out_dir / f"best_{other}.pt", map_location=device,
                                               weights_only=True))
                lo2, _ = predict_table(alt, table, spec, device, eval_batch, workers, with_emb=False)
                write_predictions(out_dir / f"{kind}_fold{k}_alt.npz", table, lo2, None, variant=other, **scal)
                metrics["oof_alt"] = oof_summary(lo2, table, m)
    metrics["seconds"] = round(time.time() - t0, 1)
    write_json(out_dir / "metrics.json", metrics)
    log(f"  OOF clip macro-F1 {metrics['oof']['clip_macro_f1']:.3f} ({sel}); "
        f"frame acc {metrics['oof']['frame_acc_centre']:.3f}; {metrics['seconds']}s")
    return metrics


def oof_summary(logits: np.ndarray, table: pd.DataFrame, m: pd.DataFrame) -> dict:
    """Quick uncalibrated OOF numbers (evaluate.py does the real report)."""
    info = m.set_index("clip_uid")
    t = table.join(info[["label", "exclude_reason", "eval_exclude_b"]], on="clip_uid")
    primary = (t["exclude_reason"] == "") & ~t["eval_exclude_b"]
    centre = primary & t["u"].between(0.2, 0.8) & t["layout_ok"]
    y = t["label"].map(STATION_INDEX).to_numpy()
    pred = logits.argmax(1)
    lp = pd.DataFrame(log_softmax(logits)[centre.to_numpy()])
    lp["clip_uid"] = t.loc[centre, "clip_uid"].to_numpy()
    clip = lp.groupby("clip_uid").mean()
    yc = info.loc[clip.index, "label"].map(STATION_INDEX).to_numpy()
    return {"frame_acc_centre": float((pred[centre] == y[centre]).mean()) if centre.any() else float("nan"),
            "clip_macro_f1": macro_f1(yc, clip.to_numpy().argmax(1)) if len(clip) else float("nan"),
            "n_clips": int(len(clip)), "n_frames": int(len(table))}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="configs/<name>.json (or a path)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int)
    g.add_argument("--all-dev", action="store_true")
    g.add_argument("--holdout", type=Path, help="CSV patient_id,period (early|late)")
    g.add_argument("--smoke", action="store_true")
    ap.add_argument("--candidate", help="override config.candidate")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int, help="override optim.steps")
    ap.add_argument("--steps-from", type=Path, help="all-dev: stop at the median best step of runs/<exp>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exp", help="experiment id (default: <config name>_s<seed>)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a config value, e.g. --set optim.lr_backbone=3e-4 (LR sweep)")
    a = ap.parse_args(argv)

    over: dict = {}
    if a.smoke:
        over = deep_update(over, SMOKE)
    if a.candidate:
        over["candidate"] = a.candidate
    if a.seed is not None:
        over["seed"] = a.seed
    if a.steps:
        over = deep_update(over, {"optim": {"steps": a.steps}})
    if a.workers is not None:
        over = deep_update(over, {"loader": {"workers": a.workers}})
    for kv in a.set:
        key, _, val = kv.partition("=")
        node: dict = {}
        cur = node
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = json.loads(val)
        over = deep_update(over, node)
    cfg = resolve_config(a.config, over)
    mode = "smoke" if a.smoke else "all_dev" if a.all_dev else "holdout" if a.holdout else "fold"
    if mode == "all_dev" and a.steps_from:
        cfg["optim"]["stop_step"], cfg["final_variant"] = fold_selection(a.steps_from)
    exp = a.exp or (f"smoke_{time.strftime('%Y%m%d-%H%M%S')}" if a.smoke else f"{cfg['name']}_s{cfg['seed']}")
    sub = {"fold": f"fold{a.fold}", "all_dev": "all_dev", "holdout": "holdout", "smoke": "smoke"}[mode]
    out = RUNS_ROOT / exp / sub
    train(cfg, mode, out, fold=a.fold, device=a.device, manifest_path=a.manifest, holdout=a.holdout)
    print(f"run dir: {out.relative_to(ML_BACKEND)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
