"""Frame-level dataset with decomposed region/score labels.

Expects directory structure:
    dataset_path/
        esophagus_0/
            esophagus_0_0010_frame0.jpg
            esophagus_0_0010_frame5.jpg
            ...
        esophagus_1/
        ...
        stomach_3/

Each class folder name is parsed as {region}_{score}.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

REGIONS = ["esophagus", "stomach", "duodenum"]
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}


def parse_class_name(class_name: str) -> tuple[int, int]:
    """Parse '{region}_{score}' into (region_idx, score_idx)."""
    parts = class_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse class name: {class_name}")
    region_str, score_str = parts
    if region_str not in REGION_TO_IDX:
        raise ValueError(f"Unknown region: {region_str}")
    return REGION_TO_IDX[region_str], int(score_str)


def _extract_video_id(path: str) -> str:
    """Extract video ID from frame filename.

    Expected format: {region}_{score}_{videoID}_frame{N}.jpg
    Returns the videoID portion.
    """
    fname = path.rsplit("/", 1)[-1]
    m = re.match(r"[a-z]+_\d+_(.+)_frame\d+\.jpg", fname)
    return m.group(1) if m else fname


def _extract_frame_number(filename: str) -> int:
    """Extract numeric frame number from filename for proper sorting."""
    m = re.search(r"frame(\d+)", filename)
    return int(m.group(1)) if m else 0


class PEACEFrameDataset(Dataset):
    """Frame-level dataset for PEACE model training.

    Each sample returns:
        image: (3, H, W) tensor
        region: int (0-2)
        score: int (0-3)
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
            if aug_cfg.get("vertical_flip", False):
                transform_list.append(transforms.RandomVerticalFlip())
            rotation = aug_cfg.get("random_rotation", 30)
            if rotation > 0:
                transform_list.append(transforms.RandomRotation(rotation))
            if aug_cfg.get("random_affine", False):
                transform_list.append(transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1),
                    scale=(0.85, 1.15), shear=10,
                ))
            jitter = aug_cfg.get("color_jitter", 0.3)
            if jitter > 0:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=jitter,
                        contrast=jitter,
                        saturation=jitter,
                        hue=jitter / 2,
                    )
                )
            if aug_cfg.get("gaussian_blur", False):
                transform_list.append(
                    transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
                )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # RandomErasing must be after ToTensor
        if augment and aug_cfg.get("random_erasing", False):
            transform_list.append(
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            )

        self.transform = transforms.Compose(transform_list)

        # TTA transforms (no random augmentation, deterministic flips)
        self.tta_transforms = [
            self.transform,  # original
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]),
        ]

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

            frame_paths = self._collect_frames(class_dir)
            for group_key, group_frames in frame_paths.items():
                sorted_frames = sorted(
                    group_frames, key=lambda p: _extract_frame_number(p.name)
                )
                for frame_path in sorted_frames:
                    self.samples.append({
                        "path": str(frame_path),
                        "region": region_idx,
                        "score": score_idx,
                    })

    def _collect_frames(
        self, class_dir: Path
    ) -> dict[str, list[Path]]:
        """Collect frame paths grouped by video/sequence."""
        groups: dict[str, list[Path]] = {}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

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
            # Group by video ID from filename
            frames = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in image_exts
            ]
            for frame in frames:
                vid_id = _extract_video_id(str(frame))
                groups.setdefault(vid_id, []).append(frame)

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
        return {
            "image": image,
            "region": sample["region"],
            "score": sample["score"],
        }

    def get_tta_images(self, idx: int) -> list[torch.Tensor]:
        """Return list of TTA-augmented versions of a frame."""
        sample = self.samples[idx]
        frame = self._load_frame(sample["path"])
        return [t(frame) for t in self.tta_transforms]


class VideoSubsampledDataset(Dataset):
    """Wrapper that randomly subsamples K frames per video each epoch.

    Call resample() at the start of each epoch to get a new random subset.
    """

    def __init__(
        self,
        base_dataset: PEACEFrameDataset,
        vid_to_sample_indices: dict[str, list[int]],
        max_frames_per_video: int = 8,
    ):
        self.base = base_dataset
        self.vid_to_sample_indices = vid_to_sample_indices
        self.max_k = max_frames_per_video
        self.active_indices: list[int] = []
        self.resample()

    def resample(self) -> None:
        """Resample K frames per video. Call at the start of each epoch."""
        self.active_indices = []
        for indices in self.vid_to_sample_indices.values():
            if len(indices) <= self.max_k:
                self.active_indices.extend(indices)
            else:
                self.active_indices.extend(random.sample(indices, self.max_k))

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> dict:
        return self.base[self.active_indices[idx]]


def create_splits(
    dataset_path: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    augmentation_config: dict | None = None,
    max_frames_per_video: int | None = None,
) -> tuple[Dataset, PEACEFrameDataset, PEACEFrameDataset]:
    """Create train/val/test splits of the dataset.

    Split is done at the VIDEO level globally to prevent data leakage.
    """
    full = PEACEFrameDataset(dataset_path, augmentation_config=augmentation_config)

    sample_vids = [_extract_video_id(s["path"]) for s in full.samples]

    vid_to_indices: dict[str, list[int]] = {}
    for i, vid in enumerate(sample_vids):
        vid_to_indices.setdefault(vid, []).append(i)

    all_vids = sorted(vid_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(all_vids)

    n_vids = len(all_vids)
    n_test = max(1, int(n_vids * test_size))
    n_val = max(1, int(n_vids * val_size))

    test_vids = set(all_vids[:n_test])
    val_vids = set(all_vids[n_test : n_test + n_val])
    train_vids = set(all_vids[n_test + n_val :])

    train_indices = [i for vid in train_vids for i in vid_to_indices[vid]]
    val_indices = [i for vid in val_vids for i in vid_to_indices[vid]]
    test_indices = [i for vid in test_vids for i in vid_to_indices[vid]]

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

    if max_frames_per_video is not None:
        train_vid_indices: dict[str, list[int]] = {}
        for i, sample in enumerate(train_ds.samples):
            vid = _extract_video_id(sample["path"])
            train_vid_indices.setdefault(vid, []).append(i)

        train_ds = VideoSubsampledDataset(
            train_ds, train_vid_indices, max_frames_per_video
        )

    return train_ds, val_ds, test_ds


def create_kfold_splits(
    dataset_path: str,
    n_folds: int = 5,
    seed: int = 42,
    augmentation_config: dict | None = None,
    max_frames_per_video: int | None = None,
) -> list[tuple[Dataset, PEACEFrameDataset]]:
    """Create K-fold cross-validation splits at the video level.

    Returns list of (train_ds, val_ds) tuples, one per fold.
    """
    full = PEACEFrameDataset(dataset_path, augmentation_config=augmentation_config)

    sample_vids = [_extract_video_id(s["path"]) for s in full.samples]
    vid_to_indices: dict[str, list[int]] = {}
    for i, vid in enumerate(sample_vids):
        vid_to_indices.setdefault(vid, []).append(i)

    all_vids = sorted(vid_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(all_vids)

    # Assign each video to a fold
    fold_assignments = [i % n_folds for i in range(len(all_vids))]

    folds = []
    for fold_idx in range(n_folds):
        val_vids = {all_vids[i] for i, f in enumerate(fold_assignments) if f == fold_idx}
        train_vids = {all_vids[i] for i, f in enumerate(fold_assignments) if f != fold_idx}

        train_indices = [i for vid in train_vids for i in vid_to_indices[vid]]
        val_indices = [i for vid in val_vids for i in vid_to_indices[vid]]

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

        if max_frames_per_video is not None:
            train_vid_indices: dict[str, list[int]] = {}
            for i, sample in enumerate(train_ds.samples):
                vid = _extract_video_id(sample["path"])
                train_vid_indices.setdefault(vid, []).append(i)
            train_ds = VideoSubsampledDataset(
                train_ds, train_vid_indices, max_frames_per_video
            )

        folds.append((train_ds, val_ds))

    return folds
