"""Screen layouts of the endoscopy video source, and the canonical aperture mask.

The processor output is a 16:9 frame with the endoscopic image in an octagonal
aperture on the right and an on-screen display (OSD) panel on the left. The
OSD carries the clock, photo counter, scope serial and mode-dependent text,
and the NBI/RDI badge sits in the aperture's top-right corner cut. All of it
must be invisible to the landmark model, so every model input is:

    ROI crop -> resize into a 320x320 canvas -> zero outside CANONICAL octagon

The canonical octagon is the INTERSECTION of the apertures of all scope
families seen in the data (measured over 300 clips; the GIF-EZ1500/HQ190
aperture is contained in the GIF-H190/1100 one), eroded by a few pixels. Using
the intersection removes the corner-cut shape as a scope-identifying cue.

Constants were derived from the landmark CLIPS only (never the full-length
recordings). Bump LAYOUTS_VERSION whenever any geometry changes: bundles
record it and refuse to load against a different version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

LAYOUTS_VERSION = "lay-3"
CANON_SIZE = 320

# Luma threshold below which a pixel counts as "black" (outside the aperture).
_DARK_Y = 25.0


@dataclass(frozen=True)
class Layout:
    name: str
    ref_width: int
    ref_height: int
    # Endoscopic-image ROI in reference pixels, [x0, x1) x [y0, y1).
    roi: tuple[int, int, int, int]
    # Aperture polygon in reference pixels (clockwise from top-left).
    polygon: tuple[tuple[float, float], ...]
    # Boxes (x0, y0, x1, y1) in reference pixels that must be zero in the
    # model input even if they overlap the aperture (mode badges, OSD text).
    blank_boxes: tuple[tuple[int, int, int, int], ...] = ()
    # Where the mode badge is read from (reference pixels), before masking.
    badge_box: tuple[int, int, int, int] | None = None
    # Served live? Layouts that are only used for shift evaluation are False.
    served: bool = True
    # Mask erosion at CANON_SIZE, in canvas pixels.
    erode_px: int = 4
    notes: str = field(default="", compare=False)

    def scale(self, width: int) -> float:
        return width / self.ref_width


# Olympus EVIS processor, 1920x1080 (all 91 newer-cohort patients).
OLYMPUS_1920 = Layout(
    name="olympus_1920x1080",
    ref_width=1920,
    ref_height=1080,
    roi=(550, 0, 1900, 1080),
    polygon=(
        (750.0, 0.0), (1699.0, 0.0), (1899.0, 250.0), (1899.0, 829.0),
        (1699.0, 1079.0), (750.0, 1079.0), (550.0, 829.0), (550.0, 250.0),
    ),
    # NBI/RDI badge. It lies outside the canonical octagon already; blanking
    # it (dilated) anyway keeps the guarantee if the polygon is ever widened.
    blank_boxes=((1829, 39, 1908, 82),),
    badge_box=(1833, 43, 1882, 78),
    notes="measured over 300 clips, 2 scope families; EZ1500/HQ190 aperture",
)

# Legacy recorder, 1350x1080 (7 patients, 2024). Shift evaluation only: the
# OSD text is drawn INSIDE the aperture, so it is never served. Measured by
# training/landmarks/derive_layouts.py over 2,134 frames of 171 legacy clips:
# aperture = centre-outward edge scans of the lit map, OSD = pixels brighter
# than their surroundings in a clip's temporal median in >= 12% of clips.
LEGACY_1350 = Layout(
    name="legacy_1350x1080",
    ref_width=1350,
    ref_height=1080,
    roi=(260, 88, 1296, 992),
    polygon=(
        (416.0, 88.0), (1138.0, 88.0), (1295.0, 300.0), (1295.0, 781.0),
        (1140.0, 991.0), (414.0, 991.0), (260.0, 782.0), (260.0, 298.0),
    ),
    # OSD text lines reaching into the aperture (patient-ID/name labels,
    # name/birth-date/age/sex labels, comment field) and the NBI badge with
    # the icon below it; each padded 8 px.
    blank_boxes=(
        (260, 233, 546, 324), (260, 346, 479, 510), (260, 884, 378, 936),
        (1195, 189, 1250, 239),
    ),
    badge_box=(1205, 200, 1240, 225),
    served=False,
    notes="measured over 171 clips; OSD text overlaps the left of the aperture",
)

LAYOUTS: tuple[Layout, ...] = (OLYMPUS_1920, LEGACY_1350)
LAYOUTS_BY_NAME: dict[str, Layout] = {l.name: l for l in LAYOUTS}

# Live inputs narrower than this are refused (unsupported_layout): the model
# and the quality gate are calibrated on full-HD source frames.
MIN_SERVED_WIDTH = 1280


def _luma(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _box(frame: np.ndarray, box: tuple[int, int, int, int], s: float) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v * s)) for v in box)
    return frame[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]


def _verify_olympus(frame: np.ndarray, layout: Layout) -> bool:
    """Dark corner cuts inside the ROI and a dark OSD panel on the left."""
    s = layout.scale(frame.shape[1])
    # Small boxes deep inside each of the four corner cuts.
    corners = (
        (560, 10, 640, 60), (1810, 10, 1890, 30),
        (560, 1020, 640, 1070), (1810, 1020, 1890, 1070),
    )
    for box in corners:
        y = _luma(_box(frame, box, s))
        if y.size == 0 or float((y < _DARK_Y).mean()) < 0.9:
            return False
    panel = _luma(_box(frame, (0, 0, 540, layout.ref_height), s))
    return float(panel.mean()) < 60.0


def detect_layout(frame: np.ndarray) -> Layout | None:
    """Identify the screen layout of a full RGB frame, or None if unknown.

    Returns the legacy layout for 1350x1080-shaped frames; callers serving
    live video must check ``layout.served``.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    h, w = frame.shape[:2]
    aspect = w / h
    if abs(aspect - 16 / 9) < 0.02:
        if w < MIN_SERVED_WIDTH:
            return None
        return OLYMPUS_1920 if _verify_olympus(frame, OLYMPUS_1920) else None
    if abs(aspect - 1350 / 1080) < 0.02:
        return LEGACY_1350
    return None


def detect_mode(frame: np.ndarray, layout: Layout) -> str:
    """Imaging mode from the processor badge: "nbi", "rdi" or "wl".

    Reads the badge BEFORE masking. For metadata, sampling weights, metrics
    and monitoring only; it is never a model input.
    """
    if layout.badge_box is None:
        return "wl"
    s = layout.scale(frame.shape[1])
    badge = _box(frame, layout.badge_box, s).astype(np.float32).mean(axis=-1)
    if badge.size == 0 or float((badge > 150).mean()) < 0.08:
        return "wl"
    # The RDI badge ("RDI 1") is wider and runs past the NBI badge's right edge.
    x0, y0, x1, y1 = layout.badge_box
    tail = _box(frame, (x1 + 8, y0, x1 + 22, y1), s).astype(np.float32).mean(axis=-1)
    if tail.size and float((tail > 150).mean()) > 0.05:
        return "rdi"
    return "nbi"


def canvas_geometry(layout: Layout) -> tuple[float, int, int]:
    """(scale, resized_height, pad_top) mapping the ROI into the square canvas."""
    x0, y0, x1, y1 = layout.roi
    scale = CANON_SIZE / (x1 - x0)
    h = int(round((y1 - y0) * scale))
    return scale, h, (CANON_SIZE - h) // 2


def _to_canvas(layout: Layout, pts, size: int) -> np.ndarray:
    scale, _, pad_top = canvas_geometry(layout)
    k = size / CANON_SIZE
    x0, y0 = layout.roi[0], layout.roi[1]
    out = [((x - x0) * scale * k, ((y - y0) * scale + pad_top) * k) for x, y in pts]
    return np.asarray(out, dtype=np.float64)


@lru_cache(maxsize=16)
def _mask_cached(layout_name: str, size: int) -> np.ndarray:
    layout = LAYOUTS_BY_NAME[layout_name]
    mask = np.zeros((size, size), np.uint8)
    # Rasterise with 4-bit sub-pixel precision so every size is consistent.
    poly = np.round(_to_canvas(layout, layout.polygon, size) * 16).astype(np.int32)
    cv2.fillPoly(mask, [poly], 1, lineType=cv2.LINE_8, shift=4)
    erode = max(1, int(round(layout.erode_px * size / CANON_SIZE)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    # Outside the canvas counts as background: the octagon's vertical sides lie
    # on the canvas border, and OpenCV's default border would leave that rim in.
    mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    for bx0, by0, bx1, by1 in layout.blank_boxes:
        (cx0, cy0), (cx1, cy1) = _to_canvas(layout, ((bx0, by0), (bx1, by1)), size)
        pad = max(2, int(round(2 * size / CANON_SIZE)))
        mask[max(int(cy0) - pad, 0):int(np.ceil(cy1)) + pad,
             max(int(cx0) - pad, 0):int(np.ceil(cx1)) + pad] = 0
    mask = mask.astype(bool)
    mask.setflags(write=False)
    return mask


def aperture_mask(layout: Layout, size: int = CANON_SIZE) -> np.ndarray:
    """Read-only boolean (size, size) mask of valid pixels in the canvas.

    The legacy layout's mask is intersected with the served (canonical)
    octagon, so every layout exposes at most the canonical aperture.
    """
    mask = _mask_cached(layout.name, size)
    if layout.name == OLYMPUS_1920.name:
        return mask
    both = np.logical_and(mask, _mask_cached(OLYMPUS_1920.name, size))
    both.setflags(write=False)
    return both
