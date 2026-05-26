"""
palette.py
----------
Deterministic color for any integer ID, used to paint cells consistently
across frames in tracking.gif and across cells within a frame in real.gif.

Golden-ratio hue spacing gives good visual separation for any number of
IDs without needing matplotlib.
"""
import colorsys
from typing import Tuple

GOLDEN = (1 + 5 ** 0.5) / 2


def color_for_id(cell_id, sat=0.75, val=0.95):
    """Return an (R, G, B) 0-255 color for the given integer-ish ID.

    Same id -> same color across calls.
    """
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
    # ensure labels fit into LUT index dtype
    return lut[labels.astype(np.int64)]
