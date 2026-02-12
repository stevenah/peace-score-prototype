"""Frame-level dataset with decomposed region/score labels.

Expects directory structure:
    dataset_path/
        esophagus_0/
            video_001/
                frame_0000.jpg
                frame_0001.jpg
                ...
            video_002/
                ...
        esophagus_1/
        ...
        stomach_3/

Each class folder name is parsed as {region}_{score}.
Motion pseudo-labels are computed from consecutive frame differencing.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

REGIONS = ["esophagus", "stomach", "duodenum"]
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}

MOTIONS = ["stationary", "insertion", "retraction"]
MOTION_STATIONARY = 0
MOTION_INSERTION = 1
MOTION_RETRACTION = 2


def parse_class_name(class_name: str) -> tuple[int, int]:
    """Parse '{region}_{score}' into (region_idx, score_idx)."""
    parts = class_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse class name: {class_name}")
    region_str, score_str = parts
    if region_str not in REGION_TO_IDX:
        raise ValueError(f"Unknown region: {region_str}")
    return REGION_TO_IDX[region_str], int(score_str)


def compute_motion_label(
    current_frame: np.ndarray,
    prev_frame: np.ndarray | None,
    stationary_threshold: float = 10.0,
    motion_threshold: float = 30.0,
) -> int:
    """Compute motion pseudo-label from frame differencing.

    Returns:
        0 = stationary, 1 = insertion, 2 = retraction
    """
    if prev_frame is None:
        return MOTION_STATIONARY

    # Convert to grayscale for differencing
    if current_frame.ndim == 3:
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY).astype(float)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY).astype(float)
    else:
        curr_gray = current_frame.astype(float)
        prev_gray = prev_frame.astype(float)

    diff = np.mean(np.abs(curr_gray - prev_gray))

    if diff < stationary_threshold:
        return MOTION_STATIONARY

    # Use vertical flow direction as proxy for insertion/retraction:
    # positive mean diff in bottom half suggests insertion (moving forward)
    h = curr_gray.shape[0]
    top_diff = np.mean(np.abs(curr_gray[: h // 2] - prev_gray[: h // 2]))
    bottom_diff = np.mean(np.abs(curr_gray[h // 2 :] - prev_gray[h // 2 :]))

    if bottom_diff > top_diff:
        return MOTION_INSERTION
    else:
        return MOTION_RETRACTION


class PEACEFrameDataset(Dataset):
    """Frame-level dataset for multitask PEACE model training.

    Each sample returns:
        image: (3, 224, 224) tensor
        prev_image: (3, 224, 224) tensor (previous frame or zeros)
        region_label: int (0-2)
        score_label: int (0-3)
        motion_label: int (0-2)
    """

    def __init__(
        self,
        dataset_path: str,
        split_indices: list[int] | None = None,
        augment: bool = False,
        image_size: int = 224,
        augmentation_config: dict | None = None,
    ):
        self.image_size = image_size
        self.samples: list[dict] = []

        aug_cfg = augmentation_config or {}

        # Build transform
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]
        if augment:
            if aug_cfg.get("horizontal_flip", True):
                transform_list.append(transforms.RandomHorizontalFlip())
            rotation = aug_cfg.get("random_rotation", 10)
            if rotation > 0:
                transform_list.append(transforms.RandomRotation(rotation))
            jitter = aug_cfg.get("color_jitter", 0.1)
            if jitter > 0:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=jitter,
                        contrast=jitter,
                        saturation=jitter,
                        hue=jitter / 2,
                    )
                )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.transform = transforms.Compose(transform_list)

        # Scan dataset directory
        self._scan_dataset(dataset_path)

        # Apply split filter
        if split_indices is not None:
            self.samples = [self.samples[i] for i in split_indices]

    def _scan_dataset(self, dataset_path: str) -> None:
        """Walk dataset directory, collect all frame paths with labels."""
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        for class_dir in sorted(dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            try:
                region_idx, score_idx = parse_class_name(class_dir.name)
            except ValueError:
                continue

            # Each class dir may contain video subdirs or direct frames
            frame_paths = self._collect_frames(class_dir)

            # Group frames by video/sequence for motion label computation
            for group_key, group_frames in frame_paths.items():
                sorted_frames = sorted(group_frames)
                for i, frame_path in enumerate(sorted_frames):
                    prev_path = sorted_frames[i - 1] if i > 0 else None
                    self.samples.append({
                        "path": str(frame_path),
                        "prev_path": str(prev_path) if prev_path else None,
                        "region": region_idx,
                        "score": score_idx,
                    })

    def _collect_frames(
        self, class_dir: Path
    ) -> dict[str, list[Path]]:
        """Collect frame paths grouped by video/sequence."""
        groups: dict[str, list[Path]] = {}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # Check for video subdirectories
        subdirs = [d for d in class_dir.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                frames = [
                    f for f in subdir.iterdir()
                    if f.suffix.lower() in image_exts
                ]
                if frames:
                    groups[subdir.name] = frames
        else:
            # Frames directly in class dir
            frames = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in image_exts
            ]
            if frames:
                groups["default"] = frames

        return groups

    def _load_frame(self, path: str) -> np.ndarray:
        """Load and resize a frame as RGB numpy array."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.image_size, self.image_size))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        frame = self._load_frame(sample["path"])
        image = self.transform(frame)

        # Load previous frame for motion label
        if sample["prev_path"]:
            prev_frame = self._load_frame(sample["prev_path"])
            prev_image = self.transform(prev_frame)
            motion_label = compute_motion_label(frame, prev_frame)
        else:
            prev_image = torch.zeros_like(image)
            motion_label = MOTION_STATIONARY

        return {
            "image": image,
            "prev_image": prev_image,
            "region": sample["region"],
            "score": sample["score"],
            "motion": motion_label,
        }


def create_splits(
    dataset_path: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    augmentation_config: dict | None = None,
) -> tuple[PEACEFrameDataset, PEACEFrameDataset, PEACEFrameDataset]:
    """Create train/val/test splits of the dataset.

    Split is done at the sample level with stratification by region+score.
    """
    # First scan to get all samples
    full = PEACEFrameDataset(dataset_path, augmentation_config=augmentation_config)
    n = len(full.samples)

    # Group indices by (region, score) for stratified split
    strata: dict[tuple[int, int], list[int]] = {}
    for i, s in enumerate(full.samples):
        key = (s["region"], s["score"])
        strata.setdefault(key, []).append(i)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for key, indices in strata.items():
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        n_val = max(1, int(len(indices) * val_size))

        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test : n_test + n_val])
        train_indices.extend(indices[n_test + n_val :])

    train_ds = PEACEFrameDataset(
        dataset_path,
        split_indices=train_indices,
        augment=True,
        augmentation_config=augmentation_config,
    )
    val_ds = PEACEFrameDataset(
        dataset_path,
        split_indices=val_indices,
        augment=False,
    )
    test_ds = PEACEFrameDataset(
        dataset_path,
        split_indices=test_indices,
        augment=False,
    )

    return train_ds, val_ds, test_ds
