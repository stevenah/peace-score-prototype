"""Training script for the PEACE model.

Supports single split and K-fold cross-validation.

Usage:
    python training/train.py --config training/config.json
    python training/train.py --config training/config.json --kfold 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.model import PEACEModel
from training.dataset import (
    PEACEFrameDataset,
    VideoSubsampledDataset,
    create_kfold_splits,
    create_splits,
)


def train_epoch(
    model: PEACEModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: dict,
    label_smoothing: float = 0.0,
    max_grad_norm: float | None = None,
    score_class_weights: torch.Tensor | None = None,
) -> dict:
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    total_losses = {"total": 0.0, "region": 0.0, "score": 0.0}
    total_correct = {"region": 0, "score": 0}
    n_samples = 0

    ce_region = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ce_score = nn.CrossEntropyLoss(weight=score_class_weights, label_smoothing=label_smoothing)

    for batch in loader:
        images = batch["image"].to(device)
        region_targets = batch["region"].to(device)
        score_targets = batch["score"].to(device)

        region_logits, score_logits, _ = model(images)

        region_loss = ce_region(region_logits, region_targets)
        score_loss = ce_score(score_logits, score_targets)

        loss = (
            loss_weights.get("region", 1.0) * region_loss
            + loss_weights.get("score", 1.0) * score_loss
        )

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        bs = images.size(0)
        n_samples += bs
        total_losses["total"] += loss.item() * bs
        total_losses["region"] += region_loss.item() * bs
        total_losses["score"] += score_loss.item() * bs

        total_correct["region"] += (region_logits.argmax(1) == region_targets).sum().item()
        total_correct["score"] += (score_logits.argmax(1) == score_targets).sum().item()

    avg_losses = {k: v / n_samples for k, v in total_losses.items()}
    accuracies = {k: v / n_samples for k, v in total_correct.items()}
    return {**{f"loss_{k}": v for k, v in avg_losses.items()}, **{f"acc_{k}": v for k, v in accuracies.items()}}


@torch.no_grad()
def validate(
    model: PEACEModel,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict,
) -> dict:
    """Validate the model. Returns dict of average losses and accuracies."""
    model.eval()
    total_losses = {"total": 0.0, "region": 0.0, "score": 0.0}
    total_correct = {"region": 0, "score": 0}
    n_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device)
        region_targets = batch["region"].to(device)
        score_targets = batch["score"].to(device)

        region_logits, score_logits, _ = model(images)

        region_loss = ce_loss(region_logits, region_targets)
        score_loss = ce_loss(score_logits, score_targets)

        loss = (
            loss_weights.get("region", 1.0) * region_loss
            + loss_weights.get("score", 1.0) * score_loss
        )

        bs = images.size(0)
        n_samples += bs
        total_losses["total"] += loss.item() * bs
        total_losses["region"] += region_loss.item() * bs
        total_losses["score"] += score_loss.item() * bs

        total_correct["region"] += (region_logits.argmax(1) == region_targets).sum().item()
        total_correct["score"] += (score_logits.argmax(1) == score_targets).sum().item()

    avg_losses = {k: v / n_samples for k, v in total_losses.items()}
    accuracies = {k: v / n_samples for k, v in total_correct.items()}
    return {**{f"loss_{k}": v for k, v in avg_losses.items()}, **{f"acc_{k}": v for k, v in accuracies.items()}}


@torch.no_grad()
def evaluate_with_tta(
    model: PEACEModel,
    dataset: PEACEFrameDataset,
    device: torch.device,
) -> dict:
    """Evaluate with test-time augmentation.

    Region: average logits across all TTA views (flips help region).
    Score: use original view only (flips hurt score predictions).
    """
    model.eval()
    correct_region = 0
    correct_score = 0
    n = len(dataset)

    for idx in range(n):
        sample = dataset.samples[idx]
        tta_images = dataset.get_tta_images(idx)
        images = torch.stack(tta_images).to(device)  # (num_tta, 3, H, W)

        region_logits, score_logits, _ = model(images)

        # Average region logits across TTA views
        avg_region = region_logits.mean(dim=0)
        region_pred = avg_region.argmax()

        # Score: use only original (first) view
        score_pred = score_logits[0].argmax()

        if region_pred.item() == sample["region"]:
            correct_region += 1
        if score_pred.item() == sample["score"]:
            correct_score += 1

    return {
        "acc_region": correct_region / n,
        "acc_score": correct_score / n,
        "n_samples": n,
    }


def train_single(
    config: dict,
    dataset_path: str,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Train on a single train/val/test split. Returns test metrics."""
    split_cfg = config.get("splits", {})
    aug_cfg = config.get("augmentation", {})
    train_ds, val_ds, test_ds = create_splits(
        dataset_path,
        test_size=split_cfg.get("test_size", 0.2),
        val_size=split_cfg.get("val_size", 0.2),
        seed=split_cfg.get("seed", 42),
        augmentation_config=aug_cfg,
        max_frames_per_video=config.get("max_frames_per_video"),
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    model, history = _train_loop(config, train_ds, val_ds, device, output_dir)

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))

    loss_weights = config.get("loss_weights", {"region": 1.0, "score": 1.0})
    test_loader = DataLoader(
        test_ds, batch_size=config.get("batch_size", 32), shuffle=False,
        num_workers=config.get("num_workers", 4), pin_memory=True,
    )
    test_metrics = validate(model, test_loader, device, loss_weights)
    print(f"Test accuracy - Region: {test_metrics['acc_region']:.4f}, Score: {test_metrics['acc_score']:.4f}")

    # TTA evaluation
    print("\n--- Test Set Evaluation (with TTA) ---")
    tta_metrics = evaluate_with_tta(model, test_ds, device)
    print(f"TTA accuracy - Region: {tta_metrics['acc_region']:.4f}, Score: {tta_metrics['acc_score']:.4f}")

    results = {**test_metrics, "tta_acc_region": tta_metrics["acc_region"], "tta_acc_score": tta_metrics["acc_score"]}

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")
    return results


def train_kfold(
    config: dict,
    dataset_path: str,
    output_dir: Path,
    device: torch.device,
    n_folds: int = 5,
) -> dict:
    """Train with K-fold cross-validation. Returns averaged metrics."""
    aug_cfg = config.get("augmentation", {})
    folds = create_kfold_splits(
        dataset_path,
        n_folds=n_folds,
        seed=config.get("splits", {}).get("seed", 42),
        augmentation_config=aug_cfg,
        max_frames_per_video=config.get("max_frames_per_video"),
    )

    all_metrics = []
    all_tta_metrics = []

    for fold_idx, (train_ds, val_ds) in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        model, history = _train_loop(config, train_ds, val_ds, device, fold_dir)

        # Evaluate best model
        model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True))
        loss_weights = config.get("loss_weights", {"region": 1.0, "score": 1.0})
        val_loader = DataLoader(
            val_ds, batch_size=config.get("batch_size", 32), shuffle=False,
            num_workers=config.get("num_workers", 4), pin_memory=True,
        )
        val_metrics = validate(model, val_loader, device, loss_weights)
        all_metrics.append(val_metrics)

        # TTA
        tta_metrics = evaluate_with_tta(model, val_ds, device)
        all_tta_metrics.append(tta_metrics)

        print(f"Fold {fold_idx + 1} - Region: {val_metrics['acc_region']:.4f}, Score: {val_metrics['acc_score']:.4f}")
        print(f"Fold {fold_idx + 1} TTA - Region: {tta_metrics['acc_region']:.4f}, Score: {tta_metrics['acc_score']:.4f}")

        with open(fold_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(fold_dir / "val_results.json", "w") as f:
            json.dump({**val_metrics, **{f"tta_{k}": v for k, v in tta_metrics.items()}}, f, indent=2)

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"K-FOLD RESULTS ({n_folds} folds)")
    print(f"{'='*60}")

    avg_region = sum(m["acc_region"] for m in all_metrics) / n_folds
    avg_score = sum(m["acc_score"] for m in all_metrics) / n_folds
    avg_tta_region = sum(m["acc_region"] for m in all_tta_metrics) / n_folds
    avg_tta_score = sum(m["acc_score"] for m in all_tta_metrics) / n_folds

    print(f"Avg Region: {avg_region:.4f}, Avg Score: {avg_score:.4f}")
    print(f"Avg TTA Region: {avg_tta_region:.4f}, Avg TTA Score: {avg_tta_score:.4f}")

    for i, (m, t) in enumerate(zip(all_metrics, all_tta_metrics)):
        print(f"  Fold {i+1}: R={m['acc_region']:.4f} S={m['acc_score']:.4f} | TTA R={t['acc_region']:.4f} S={t['acc_score']:.4f}")

    summary = {
        "n_folds": n_folds,
        "avg_acc_region": avg_region,
        "avg_acc_score": avg_score,
        "avg_tta_acc_region": avg_tta_region,
        "avg_tta_acc_score": avg_tta_score,
        "per_fold": [
            {"acc_region": m["acc_region"], "acc_score": m["acc_score"],
             "tta_acc_region": t["acc_region"], "tta_acc_score": t["acc_score"]}
            for m, t in zip(all_metrics, all_tta_metrics)
        ],
    }
    with open(output_dir / "kfold_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")
    return summary


def _train_loop(
    config: dict,
    train_ds,
    val_ds: PEACEFrameDataset,
    device: torch.device,
    output_dir: Path,
) -> tuple[PEACEModel, list[dict]]:
    """Core training loop shared by single and k-fold modes."""
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Create model
    model_cfg = config.get("model", {})
    model = PEACEModel(
        pretrained_backbone=model_cfg.get("pretrained_backbone", True),
        neck_dim=model_cfg.get("neck_dim", 128),
    )
    if model_cfg.get("freeze_backbone", True):
        model.freeze_backbone()
        print("Backbone frozen for warmup")
    model = model.to(device)

    # Optimizer
    opt_cfg = config.get("optimizer", {})
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg.get("lr", 0.0003),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    # Training config
    train_cfg = config.get("training", {})
    num_epochs = train_cfg.get("num_epochs", 200)
    patience = train_cfg.get("patience", 30)
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    save_interval = train_cfg.get("save_interval", 10)
    label_smoothing = config.get("label_smoothing", 0.1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    loss_weights = config.get("loss_weights", {"region": 1.0, "score": 1.0})

    # Compute class weights for score task (inverse frequency)
    base_ds = train_ds.base if isinstance(train_ds, VideoSubsampledDataset) else train_ds
    num_scores = model.num_scores
    score_counts = [0] * num_scores
    for sample in base_ds.samples:
        score_counts[sample["score"]] += 1
    total_score_samples = sum(score_counts)
    score_class_weights = torch.tensor(
        [total_score_samples / (num_scores * c) for c in score_counts],
        dtype=torch.float32,
    ).to(device)
    print(f"Score class weights: {score_class_weights.tolist()}")

    # Scheduler
    sched_cfg = config.get("scheduler", {})
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg.get("T_max", num_epochs),
        eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    print(f"Training for up to {num_epochs} epochs (patience={patience})")

    for epoch in range(1, num_epochs + 1):
        # Unfreeze backbone after warmup
        if warmup_epochs > 0 and epoch == warmup_epochs + 1:
            model.unfreeze_backbone(last_stage_only=True)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt_cfg.get("lr", 0.0003) * 0.1,
                weight_decay=opt_cfg.get("weight_decay", 0.01),
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=sched_cfg.get("eta_min", 1e-6),
            )
            print(f"Epoch {epoch}: Backbone last stage unfrozen")

        # Resample frames per video each epoch
        if isinstance(train_ds, VideoSubsampledDataset):
            train_ds.resample()

        start = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, loss_weights,
            label_smoothing=label_smoothing,
            max_grad_norm=max_grad_norm,
            score_class_weights=score_class_weights,
        )
        val_metrics = validate(model, val_loader, device, loss_weights)
        elapsed = time.time() - start

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        entry = {"epoch": epoch, "lr": current_lr, "time": elapsed, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(entry)

        print(
            f"Epoch {epoch:4d} | "
            f"train: {train_metrics['loss_total']:.4f} | "
            f"val: {val_metrics['loss_total']:.4f} | "
            f"R:{val_metrics['acc_region']:.3f} S:{val_metrics['acc_score']:.3f} | "
            f"lr: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch}.pt")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train PEACE model")
    parser.add_argument("--config", default="training/config.json")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kfold", type=int, default=0, help="Number of folds (0=single split)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}

    dataset_path = args.dataset_path or config.get("dataset_path", "dataset/data-12-classes")
    output_dir = Path(args.output_dir or config.get("output_dir", "experiments/multitask"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading dataset from: {dataset_path}")

    if args.kfold > 1:
        train_kfold(config, dataset_path, output_dir, device, n_folds=args.kfold)
    else:
        train_single(config, dataset_path, output_dir, device)


if __name__ == "__main__":
    main()
