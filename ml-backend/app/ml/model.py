"""PEACE frame classification model.

Architecture:
  RegNet Y 3.2GF backbone (1512-dim) → Shared Neck (128-dim)
    ├── Region Head  → 3 classes (esophagus, stomach, duodenum)
    └── Score Head   → 4 classes (PEACE 0-3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights

REGIONS = ["esophagus", "stomach", "duodenum"]
SCORES = [0, 1, 2, 3]

# RegNet Y 3.2GF final stage output channels
BACKBONE_DIM = 1512
NECK_DIM = 128


class PEACEModel(nn.Module):
    """Model with shared backbone and two heads: region + score."""

    def __init__(
        self,
        backbone_dim: int = BACKBONE_DIM,
        neck_dim: int = NECK_DIM,
        num_regions: int = 3,
        num_scores: int = 4,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.neck_dim = neck_dim
        self.num_scores = num_scores

        # Backbone: RegNet Y 3.2GF without classification head
        weights = RegNet_Y_3_2GF_Weights.DEFAULT if pretrained_backbone else None
        backbone_full = regnet_y_3_2gf(weights=weights)
        self.backbone = nn.Sequential(*list(backbone_full.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Shared feature neck
        self.neck = nn.Sequential(
            nn.Linear(backbone_dim, neck_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Region head (classification)
        self.region_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(neck_dim, num_regions),
        )

        # Score head (classification) — deeper than region head since
        # score grading requires more complex feature combinations
        self.score_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(neck_dim, neck_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(neck_dim, num_scores),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract neck-level features from a batch of frames."""
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)  # (B, backbone_dim)
        return self.neck(feat)  # (B, neck_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for both heads plus features.

        Returns:
            region_logits: (B, num_regions)
            score_logits: (B, num_scores)
            features: (B, neck_dim)
        """
        features = self.extract_features(x)
        region_logits = self.region_head(features)
        score_logits = self.score_head(features)
        return region_logits, score_logits, features

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_stage_only: bool = True) -> None:
        """Unfreeze backbone parameters."""
        if last_stage_only:
            # backbone[1] is Sequential containing AnyStage modules
            stages = list(self.backbone[1].children())
            for param in stages[-1].parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
