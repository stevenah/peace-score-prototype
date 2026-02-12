"""Multitask frame-level PEACE model.

Architecture:
  RegNet Y 3.2GF backbone (1512-dim) → Shared Neck (256-dim)
    ├── Region Head  → 3 classes (esophagus, stomach, duodenum)
    ├── Score Head   → 4 classes (PEACE 0-3)
    └── Motion Head  → 3 classes (insertion, retraction, stationary)
                       uses concat of current + previous features
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights

REGIONS = ["esophagus", "stomach", "duodenum"]
SCORES = [0, 1, 2, 3]
MOTIONS = ["insertion", "retraction", "stationary"]

# RegNet Y 3.2GF final stage output channels
BACKBONE_DIM = 1512
NECK_DIM = 256


class PEACEMultitaskModel(nn.Module):
    """Multitask model with shared backbone and three classification heads."""

    def __init__(
        self,
        backbone_dim: int = BACKBONE_DIM,
        neck_dim: int = NECK_DIM,
        num_regions: int = 3,
        num_scores: int = 4,
        num_motions: int = 3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.neck_dim = neck_dim

        # Backbone: RegNet Y 3.2GF without classification head
        weights = RegNet_Y_3_2GF_Weights.DEFAULT if pretrained_backbone else None
        backbone_full = regnet_y_3_2gf(weights=weights)
        self.backbone = nn.Sequential(*list(backbone_full.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Shared feature neck
        self.neck = nn.Sequential(
            nn.Linear(backbone_dim, neck_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(neck_dim, neck_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Region head (3 classes)
        self.region_head = nn.Sequential(
            nn.Linear(neck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_regions),
        )

        # PEACE score head (4 classes)
        self.score_head = nn.Sequential(
            nn.Linear(neck_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_scores),
        )

        # Motion head (3 classes) — takes current + previous features
        self.motion_head = nn.Sequential(
            nn.Linear(neck_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_motions),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract neck-level features from a batch of frames.

        Args:
            x: (B, 3, 224, 224) image tensor

        Returns:
            (B, neck_dim) feature tensor
        """
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)  # (B, backbone_dim)
        return self.neck(feat)  # (B, neck_dim)

    def forward(
        self,
        x: torch.Tensor,
        prev_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for all heads plus features.

        Args:
            x: (B, 3, 224, 224) image tensor
            prev_features: (B, neck_dim) features from previous frame, or None

        Returns:
            region_logits: (B, 3)
            score_logits: (B, 4)
            motion_logits: (B, 3)
            features: (B, neck_dim) — for caching as next frame's prev_features
        """
        features = self.extract_features(x)

        region_logits = self.region_head(features)
        score_logits = self.score_head(features)

        # Motion head uses concatenated current + previous features
        if prev_features is None:
            prev_features = torch.zeros_like(features)
        motion_input = torch.cat([features, prev_features], dim=-1)
        motion_logits = self.motion_head(motion_input)

        return region_logits, score_logits, motion_logits, features

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for warmup training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_stage_only: bool = True) -> None:
        """Unfreeze backbone parameters.

        Args:
            last_stage_only: If True, only unfreeze the last stage of RegNet.
        """
        if last_stage_only:
            # RegNet backbone children: stem, s1, s2, s3, s4
            # Unfreeze s4 (last stage)
            children = list(self.backbone[0].children())
            if len(children) >= 5:
                for param in children[-1].parameters():
                    param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
