"""Landmark model architectures: permissive backbones plus a linear head.

Every model returns ``(logits, embedding)``; the embedding is the pooled
backbone feature the head sees, used for probes, kNN rejection and the
cross-split leakage audit.

Weights are NEVER downloaded here (``weights=None`` / ``pretrained=False``).
Training loads pretrained backbone weights separately from an allowlist of
weight tags with sha256s, e.g. ``net.backbone.load_state_dict(sd, strict=False)``
where the only tolerated unexpected keys are the dropped ImageNet head.

    arch          backbone                                  embedding  licence
    effb0         torchvision efficientnet_b0               1280       BSD-3
    r50           torchvision resnet50                      2048       BSD-3
    regy32        torchvision regnet_y_3_2gf                1512       BSD-3
    cnxt_t        torchvision convnext_tiny                 768        BSD-3
    dinov2_s14    timm vit_small_patch14_dinov2.lvd142m     384        Apache-2.0
    tiny_test_cnn 2 conv layers (tests and fixtures only)   16         -
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from app.ml.landmarks.stations import NUM_STATIONS


class LandmarkNet(nn.Module):
    """Backbone -> pooled embedding -> Linear(num_classes)."""

    def __init__(self, backbone: nn.Module, embedding_dim: int, num_classes: int, arch: str):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.arch = arch

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x)
        return self.head(emb), emb


def _effb0() -> tuple[nn.Module, int]:
    from torchvision.models import efficientnet_b0

    m = efficientnet_b0(weights=None)
    dim = m.classifier[1].in_features
    m.classifier = nn.Identity()
    return m, dim


def _r50() -> tuple[nn.Module, int]:
    from torchvision.models import resnet50

    m = resnet50(weights=None)
    dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, dim


def _regy32() -> tuple[nn.Module, int]:
    from torchvision.models import regnet_y_3_2gf

    m = regnet_y_3_2gf(weights=None)
    dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, dim


def _cnxt_t() -> tuple[nn.Module, int]:
    from torchvision.models import convnext_tiny

    m = convnext_tiny(weights=None)
    # classifier = [LayerNorm2d, Flatten, Linear]; keep the norm + flatten.
    dim = m.classifier[2].in_features
    m.classifier[2] = nn.Identity()
    return m, dim


def _dinov2_s14() -> tuple[nn.Module, int]:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("dinov2_s14 needs timm (pip install '.[train]')") from exc
    # Dynamic size + padding so non-multiples of 14 (e.g. 288) still run.
    m = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=224,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    return m, int(m.num_features)


def _tiny_test_cnn() -> tuple[nn.Module, int]:
    m = nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return m, 16


_FACTORIES: dict[str, Callable[[], tuple[nn.Module, int]]] = {
    "effb0": _effb0,
    "r50": _r50,
    "regy32": _regy32,
    "cnxt_t": _cnxt_t,
    "dinov2_s14": _dinov2_s14,
    "tiny_test_cnn": _tiny_test_cnn,
}

ARCHS: tuple[str, ...] = tuple(_FACTORIES)

# Convnets: may run NHWC (channels_last); see LandmarkModelManager and bench.py.
CONV_ARCHS = frozenset({"effb0", "r50", "regy32", "cnxt_t", "tiny_test_cnn"})


def available_archs() -> tuple[str, ...]:
    """Architectures buildable in this environment (timm is optional)."""
    try:
        import timm  # noqa: F401
    except ImportError:
        return tuple(a for a in ARCHS if a != "dinov2_s14")
    return ARCHS


def build(arch: str, num_classes: int = NUM_STATIONS) -> LandmarkNet:
    """Randomly initialised model for ``arch`` in eval mode."""
    try:
        factory = _FACTORIES[arch]
    except KeyError:
        raise ValueError(f"unknown landmark arch {arch!r}; expected one of {ARCHS}") from None
    backbone, dim = factory()
    return LandmarkNet(backbone, dim, num_classes, arch).eval()
