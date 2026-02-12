"""Training script for the multitask PEACE model.

Usage:
    python training/train.py
    python training/train.py --config training/config.json
    python training/train.py --dataset_path /path/to/data --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.model import PEACEMultitaskModel, REGIONS, MOTIONS
from training.dataset import create_splits


def ordinal_mse_loss(score_logits: torch.Tensor, score_targets: torch.Tensor) -> torch.Tensor:
    """Auxiliary ordinal loss: MSE between expected score and true score.

    Computes expected score as sum(p_i * i) from softmax probabilities,
    then MSE against the true score. This enforces score ordering.
    """
    probs = torch.softmax(score_logits, dim=-1)
    indices = torch.arange(score_logits.size(1), device=score_logits.device, dtype=torch.float)
    expected_score = (probs * indices).sum(dim=-1)
    return nn.functional.mse_loss(expected_score, score_targets.float())


def train_epoch(
    model: PEACEMultitaskModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: dict,
) -> dict:
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    total_losses = {
        "total": 0.0, "region": 0.0, "score": 0.0,
        "score_ordinal": 0.0, "motion": 0.0,
    }
    total_correct = {"region": 0, "score": 0, "motion": 0}
    n_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device)
        prev_images = batch["prev_image"].to(device)
        region_targets = batch["region"].to(device)
        score_targets = batch["score"].to(device)
        motion_targets = batch["motion"].to(device)

        # Extract previous features for motion head
        with torch.no_grad():
            prev_features = model.extract_features(prev_images)

        region_logits, score_logits, motion_logits, _ = model(images, prev_features)

        # Compute losses
        region_loss = ce_loss(region_logits, region_targets)
        score_loss = ce_loss(score_logits, score_targets)
        score_ord_loss = ordinal_mse_loss(score_logits, score_targets)
        motion_loss = ce_loss(motion_logits, motion_targets)

        loss = (
            loss_weights["region"] * region_loss
            + loss_weights["score"] * score_loss
            + loss_weights["score_ordinal"] * score_ord_loss
            + loss_weights["motion"] * motion_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        n_samples += bs
        total_losses["total"] += loss.item() * bs
        total_losses["region"] += region_loss.item() * bs
        total_losses["score"] += score_loss.item() * bs
        total_losses["score_ordinal"] += score_ord_loss.item() * bs
        total_losses["motion"] += motion_loss.item() * bs

        total_correct["region"] += (region_logits.argmax(1) == region_targets).sum().item()
        total_correct["score"] += (score_logits.argmax(1) == score_targets).sum().item()
        total_correct["motion"] += (motion_logits.argmax(1) == motion_targets).sum().item()

    avg_losses = {k: v / n_samples for k, v in total_losses.items()}
    accuracies = {k: v / n_samples for k, v in total_correct.items()}
    return {**{f"loss_{k}": v for k, v in avg_losses.items()}, **{f"acc_{k}": v for k, v in accuracies.items()}}


@torch.no_grad()
def validate(
    model: PEACEMultitaskModel,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict,
) -> dict:
    """Validate the model. Returns dict of average losses and accuracies."""
    model.eval()
    total_losses = {
        "total": 0.0, "region": 0.0, "score": 0.0,
        "score_ordinal": 0.0, "motion": 0.0,
    }
    total_correct = {"region": 0, "score": 0, "motion": 0}
    n_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device)
        prev_images = batch["prev_image"].to(device)
        region_targets = batch["region"].to(device)
        score_targets = batch["score"].to(device)
        motion_targets = batch["motion"].to(device)

        prev_features = model.extract_features(prev_images)
        region_logits, score_logits, motion_logits, _ = model(images, prev_features)

        region_loss = ce_loss(region_logits, region_targets)
        score_loss = ce_loss(score_logits, score_targets)
        score_ord_loss = ordinal_mse_loss(score_logits, score_targets)
        motion_loss = ce_loss(motion_logits, motion_targets)

        loss = (
            loss_weights["region"] * region_loss
            + loss_weights["score"] * score_loss
            + loss_weights["score_ordinal"] * score_ord_loss
            + loss_weights["motion"] * motion_loss
        )

        bs = images.size(0)
        n_samples += bs
        total_losses["total"] += loss.item() * bs
        total_losses["region"] += region_loss.item() * bs
        total_losses["score"] += score_loss.item() * bs
        total_losses["score_ordinal"] += score_ord_loss.item() * bs
        total_losses["motion"] += motion_loss.item() * bs

        total_correct["region"] += (region_logits.argmax(1) == region_targets).sum().item()
        total_correct["score"] += (score_logits.argmax(1) == score_targets).sum().item()
        total_correct["motion"] += (motion_logits.argmax(1) == motion_targets).sum().item()

    avg_losses = {k: v / n_samples for k, v in total_losses.items()}
    accuracies = {k: v / n_samples for k, v in total_correct.items()}
    return {**{f"loss_{k}": v for k, v in avg_losses.items()}, **{f"acc_{k}": v for k, v in accuracies.items()}}


def main():
    parser = argparse.ArgumentParser(description="Train multitask PEACE model")
    parser.add_argument("--config", default="training/config.json", help="Config file path")
    parser.add_argument("--dataset_path", default=None, help="Override dataset path")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    # Load config
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

    # Device selection
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

    # Create datasets
    print(f"Loading dataset from: {dataset_path}")
    split_cfg = config.get("splits", {})
    aug_cfg = config.get("augmentation", {})
    train_ds, val_ds, test_ds = create_splits(
        dataset_path,
        test_size=split_cfg.get("test_size", 0.2),
        val_size=split_cfg.get("val_size", 0.2),
        seed=split_cfg.get("seed", 42),
        augmentation_config=aug_cfg,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

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
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Create model
    model_cfg = config.get("model", {})
    model = PEACEMultitaskModel(
        pretrained_backbone=model_cfg.get("pretrained_backbone", True),
    )
    if model_cfg.get("freeze_backbone", True):
        model.freeze_backbone()
        print("Backbone frozen for warmup")
    model = model.to(device)

    # Optimizer and scheduler
    opt_cfg = config.get("optimizer", {})
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg.get("lr", 0.001),
        weight_decay=opt_cfg.get("weight_decay", 0.0003),
    )
    sched_cfg = config.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        patience=sched_cfg.get("patience", 5),
        factor=sched_cfg.get("factor", 0.2),
    )

    # Training config
    train_cfg = config.get("training", {})
    num_epochs = train_cfg.get("num_epochs", 1500)
    patience = train_cfg.get("patience", 25)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    save_interval = train_cfg.get("save_interval", 5)
    loss_weights = config.get("loss_weights", {
        "region": 1.0, "score": 1.0, "score_ordinal": 0.3, "motion": 0.5,
    })

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    print(f"\nStarting training for up to {num_epochs} epochs")
    print(f"Early stopping patience: {patience}")
    print(f"Backbone unfreezes after epoch {warmup_epochs}\n")

    for epoch in range(1, num_epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == warmup_epochs + 1:
            model.unfreeze_backbone(last_stage_only=True)
            # Re-create optimizer to include newly unfrozen params
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt_cfg.get("lr", 0.001) * 0.1,  # Lower LR for fine-tuning
                weight_decay=opt_cfg.get("weight_decay", 0.0003),
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=sched_cfg.get("mode", "min"),
                patience=sched_cfg.get("patience", 5),
                factor=sched_cfg.get("factor", 0.2),
            )
            print(f"Epoch {epoch}: Backbone last stage unfrozen, LR reduced to {opt_cfg.get('lr', 0.001) * 0.1}")

        start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, loss_weights)
        val_metrics = validate(model, val_loader, device, loss_weights)
        elapsed = time.time() - start

        scheduler.step(val_metrics["loss_total"])
        current_lr = optimizer.param_groups[0]["lr"]

        entry = {"epoch": epoch, "lr": current_lr, "time": elapsed, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(entry)

        # Print progress
        print(
            f"Epoch {epoch:4d} | "
            f"train_loss: {train_metrics['loss_total']:.4f} | "
            f"val_loss: {val_metrics['loss_total']:.4f} | "
            f"val_acc R:{val_metrics['acc_region']:.3f} S:{val_metrics['acc_score']:.3f} M:{val_metrics['acc_motion']:.3f} | "
            f"lr: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # Periodic save
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch}.pt")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save final model and history
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    test_metrics = validate(model, test_loader, device, loss_weights)
    print(f"Test loss: {test_metrics['loss_total']:.4f}")
    print(f"Test accuracy - Region: {test_metrics['acc_region']:.4f}, Score: {test_metrics['acc_score']:.4f}, Motion: {test_metrics['acc_motion']:.4f}")

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
