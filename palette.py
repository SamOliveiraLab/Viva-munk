"""
palette.py
----------
Deterministic color for any integer ID and matching Partaker's labeled
segmentation look (vivid colors + cell-id text on every cell).
"""
import colorsys
from typing import Tuple

GOLDEN = (1 + 5 ** 0.5) / 2


def color_for_id(cell_id, sat=0.85, val=0.95):
    """Return an (R, G, B) 0-255 color for the given integer-ish ID.
    Same id -> same color across calls."""
    try:
        idx = int(cell_id)
    except (TypeError, ValueError):
        idx = hash(str(cell_id)) & 0xFFFF
    hue = (idx * GOLDEN) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def labels_to_rgb(labels):
    """Convert a 2D integer label image into an (H, W, 3) uint8 RGB array,
    coloring every nonzero label with `color_for_id(label)`.
    Background (label 0) stays black.

    Uses a lookup table so cost is O(H*W + N_unique), not O(N_unique * H * W).
    """
    import numpy as np
    h, w = labels.shape
    if labels.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    max_label = int(labels.max())
    if max_label == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        lut[int(lbl)] = color_for_id(int(lbl))
    return lut[labels.astype(np.int64)]


def compute_label_centroids(labels):
    """Return a dict {label_id: (cx, cy)} of pixel-space centroids.
    Background label 0 is skipped.  O(H*W).
    """
    import numpy as np
    h, w = labels.shape
    flat = labels.ravel().astype(np.int64)
    max_label = int(flat.max()) if flat.size else 0
    if max_label == 0:
        return {}
    ys, xs = np.indices((h, w))
    counts = np.bincount(flat, minlength=max_label + 1)
    sum_x = np.bincount(flat, weights=xs.ravel(), minlength=max_label + 1)
    sum_y = np.bincount(flat, weights=ys.ravel(), minlength=max_label + 1)
    out = {}
    for lbl in range(1, max_label + 1):
        c = counts[lbl]
        if c > 0:
            out[lbl] = (float(sum_x[lbl] / c), float(sum_y[lbl] / c))
    return out


_FONT_CACHE = {}


def _get_font(size=10):
    """Best-effort small font for cell ID labels."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        try:
            f = ImageFont.truetype(c, size)
            _FONT_CACHE[size] = f
            return f
        except (OSError, IOError):
            continue
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def draw_id_labels(pil_img, centroids, font_size=10, max_chars=6):
    """Draw white text with a black outline at every centroid.
    `centroids` is {id: (cx, cy)}.  Mutates pil_img and returns it.
    """
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    W, H = pil_img.size
    for cid, (cx, cy) in centroids.items():
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        text = str(cid)
        if len(text) > max_chars:
            text = text[-max_chars:]
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
        tx = int(cx - tw / 2)
        ty = int(cy - th / 2)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return pil_img

