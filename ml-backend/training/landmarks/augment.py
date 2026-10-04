"""Training augmentation for the landmark model (cv2/numpy, deterministic per rng).

Order (the mask is applied LAST, so every model input is exactly 0 outside
the aperture, whatever the geometric and photometric steps did):

    rotation U(-90, 90) deg, bilinear, p=0.5        (scope torque; no flips)
    random-resized crop back to the 320 canvas      (scale 0.7-1.0, ratio 0.9-1.1)
    colour jitter (0.2, 0.2, 0.2, 0.02), random order
    grayscale p=0.1
    Gaussian blur p=0.2, sigma 0.1-2
    JPEG q75-95 p=0.5                               (the browser's q0.8 happens at full res)
    preprocess.to_model_input(canvas, layout, size)  INTER_AREA + re-mask
    preprocess.normalize

Flips, pseudo-NBI channel remap and a circle mask are fold-0 ablations only
(off by default). ``masked=False`` skips the mask entirely and exists only for
the sanity positive control (a model that is ALLOWED to see the badge must
trip the shortcut checks).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import cv2
import numpy as np

from app.ml.landmarks.layouts import CANON_SIZE, Layout, aperture_mask
from app.ml.landmarks.preprocess import IMAGENET_MEAN, IMAGENET_STD, normalize, to_model_input


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True
    rotate_p: float = 0.5
    rotate_deg: float = 90.0
    rrc_p: float = 1.0
    rrc_scale: tuple[float, float] = (0.7, 1.0)
    rrc_ratio: tuple[float, float] = (0.9, 1.1)
    jitter: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.02)  # brightness, contrast, saturation, hue
    gray_p: float = 0.1
    blur_p: float = 0.2
    blur_sigma: tuple[float, float] = (0.1, 2.0)
    jpeg_p: float = 0.5
    jpeg_q: tuple[int, int] = (75, 95)
    hflip_p: float = 0.0  # ablation
    vflip_p: float = 0.0  # ablation
    pseudo_nbi_p: float = 0.0  # ablation: WL -> NBI-like channel remap
    mask: str = "octagon"  # octagon | circle (ablation)

    @classmethod
    def from_dict(cls, d: dict | None) -> "AugmentConfig":
        d = dict(d or {})
        for k in ("rrc_scale", "rrc_ratio", "jitter", "blur_sigma", "jpeg_q"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


# -- single steps (uint8 HxWx3 RGB in, uint8 out) -----------------------------------------------

def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def random_resized_crop(img: np.ndarray, rng: np.random.Generator, scale, ratio, out: int = CANON_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    area = h * w * rng.uniform(*scale)
    r = float(np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1]))))
    cw = int(min(w, max(8, round(np.sqrt(area * r)))))
    ch = int(min(h, max(8, round(np.sqrt(area / r)))))
    x0 = int(rng.integers(0, w - cw + 1))
    y0 = int(rng.integers(0, h - ch + 1))
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (out, out), interpolation=cv2.INTER_LINEAR)


def _luma(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def color_jitter(img: np.ndarray, rng: np.random.Generator, jitter) -> np.ndarray:
    b, c, s, hue = jitter
    x = img.astype(np.float32)
    lit = x.max(-1) > 0  # statistics over the aperture, not the black surround
    for op in rng.permutation(4):
        if op == 0 and b > 0:
            x = x * rng.uniform(1 - b, 1 + b)
        elif op == 1 and c > 0:
            m = float(_luma(x)[lit].mean()) if lit.any() else 0.0
            x = (x - m) * rng.uniform(1 - c, 1 + c) + m
        elif op == 2 and s > 0:
            g = _luma(x)[..., None]
            x = (x - g) * rng.uniform(1 - s, 1 + s) + g
        elif op == 3 and hue > 0:
            hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            shift = int(round(rng.uniform(-hue, hue) * 180))
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
            x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def grayscale(img: np.ndarray) -> np.ndarray:
    g = np.clip(_luma(img.astype(np.float32)), 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:  # pragma: no cover
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def pseudo_nbi(img: np.ndarray) -> np.ndarray:
    """NBI displays 540 nm in red and 415 nm in green+blue: R<-G, G<-B, B<-B."""
    return np.stack([img[..., 1], img[..., 2], img[..., 2]], axis=-1)


@lru_cache(maxsize=8)
def _circle(size: int) -> np.ndarray:
    """Disc inscribed in the canvas's lit band (the 1080-px ROI height)."""
    m = np.zeros((size, size), np.uint8)
    r = int(np.floor(size * 0.4))  # 256/320 of the canvas height is lit
    cv2.circle(m, (size // 2, size // 2), r, 1, -1)
    m = m.astype(bool)
    m.setflags(write=False)
    return m


def input_mask(layout: Layout, size: int, mask: str = "octagon") -> np.ndarray:
    """Valid-pixel mask of a model input (aperture, or aperture AND circle)."""
    m = aperture_mask(layout, size)
    return m & _circle(size) if mask == "circle" else m


def finish(canvas: np.ndarray, layout: Layout, size: int, masked: bool = True, mask: str = "octagon") -> np.ndarray:
    """Canvas -> uint8 model input. The ONLY place the training mask is applied."""
    if not masked:
        return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    img = to_model_input(canvas, layout, size)
    if mask == "circle":
        img[~_circle(size)] = 0
    return img


def augment(canvas: np.ndarray, layout: Layout, cfg: AugmentConfig, rng: np.random.Generator,
            size: int = 224, masked: bool = True) -> np.ndarray:
    """One augmented, masked uint8 model input from a 320 px cached canvas."""
    img = canvas
    if cfg.enabled:
        if cfg.hflip_p and rng.random() < cfg.hflip_p:
            img = img[:, ::-1]
        if cfg.vflip_p and rng.random() < cfg.vflip_p:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        if rng.random() < cfg.rotate_p:
            img = rotate(img, rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
        if rng.random() < cfg.rrc_p:
            img = random_resized_crop(img, rng, cfg.rrc_scale, cfg.rrc_ratio)
        if cfg.pseudo_nbi_p and rng.random() < cfg.pseudo_nbi_p:
            img = pseudo_nbi(img)
        img = color_jitter(img, rng, cfg.jitter)
        if rng.random() < cfg.gray_p:
            img = grayscale(img)
        if rng.random() < cfg.blur_p:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*cfg.blur_sigma))
        if rng.random() < cfg.jpeg_p:
            img = jpeg(img, int(rng.integers(cfg.jpeg_q[0], cfg.jpeg_q[1] + 1)))
    return finish(img, layout, size, masked, cfg.mask)


def augment_to_tensor(canvas, layout, cfg, rng, size=224, masked=True):
    return normalize(augment(canvas, layout, cfg, rng, size, masked))


def eval_input(canvas: np.ndarray, layout: Layout, size: int = 224, masked: bool = True) -> np.ndarray:
    return finish(canvas, layout, size, masked)


def eval_to_tensor(canvas, layout, size=224, masked=True):
    return normalize(eval_input(canvas, layout, size, masked))


# -- the exact-zero guarantee ---------------------------------------------------------------------

def assert_zero_outside(img: np.ndarray, layout: Layout, mask: str = "octagon") -> None:
    """uint8 model input: every pixel outside the input mask is exactly 0."""
    m = input_mask(layout, img.shape[0], mask)
    bad = int(np.count_nonzero(img[~m]))
    if bad:
        raise AssertionError(f"{bad} non-zero values outside the aperture mask")


def assert_batch_masked(x, layout: Layout, mask: str = "octagon") -> None:
    """Normalised (B, 3, S, S) batch: outside the mask equals normalize(0) exactly."""
    import torch

    m = torch.from_numpy(~input_mask(layout, x.shape[-1], mask))
    zero = normalize(np.zeros((1, 1, 3), np.uint8)).reshape(1, 3, 1, 1).to(x.dtype)
    outside = x.detach().cpu()[:, :, m]
    if not torch.equal(outside, zero.reshape(1, 3, 1).expand_as(outside)):
        raise AssertionError("model input is not exactly zero outside the aperture mask")


# Normalised value of a black pixel, for reference in tests and sanity checks.
NORMALIZED_ZERO = tuple(-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD))
