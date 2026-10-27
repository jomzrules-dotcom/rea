"""RGBA image primitives: loading, background keying and connected components.

Everything works on ``uint8`` arrays of shape (H, W, 4). Only numpy and Pillow
are required.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MAGENTA = (255, 0, 255)


def load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"))


def save_rgba(arr: np.ndarray, path: str | Path) -> None:
    Image.fromarray(arr.astype(np.uint8), "RGBA").save(path)


def color_distance(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    diff = rgb[..., :3].astype(np.int32) - np.array(color, dtype=np.int32)
    return np.sqrt((diff * diff).sum(axis=-1))


def border_color(arr: np.ndarray) -> tuple[int, int, int]:
    """Most common colour on the outer 1 px border (the usual flat background)."""
    edge = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]])
    (color, _), = Counter(map(tuple, edge.tolist())).most_common(1)
    return tuple(int(c) for c in color)


def foreground_mask(arr: np.ndarray, tolerance: float = 24.0) -> tuple[np.ndarray, dict]:
    """Return (mask, info). Uses real transparency when present, else the border colour."""
    alpha = arr[..., 3]
    if (alpha < 255).any() and (alpha == 0).any():
        return alpha > 0, {"background": "alpha"}
    bg = border_color(arr)
    return color_distance(arr, bg) > tolerance, {"background": "color", "color": list(bg)}


def key_magenta(arr: np.ndarray, distance: float = 110.0, spill: int = 90) -> np.ndarray:
    """Foreground mask for art on a flat #FF00FF background, dropping magenta spill."""
    rgb = arr[..., :3].astype(np.int32)
    spill_px = (np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]) > spill
    return (color_distance(arr, MAGENTA) >= distance) & ~spill_px & (arr[..., 3] > 0)


def green_dot_mask(arr: np.ndarray) -> np.ndarray:
    r, g, b = (arr[..., i].astype(np.int32) for i in range(3))
    return (g > 150) & (r < 140) & (b < 140) & (g - r > 60)


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Connected-component labelling (labels 1..n, 0 = background).

    Vectorised min-label propagation with pointer jumping; converges in a few
    dozen iterations for sprite-sized blobs and needs no scipy.
    """
    h, w = mask.shape
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    big = np.int64(h * w)
    lab = np.where(mask, idx, big)
    shifts = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if connectivity == 8 else [])
    while True:
        prev = lab
        cur = lab.copy()
        for dy, dx in shifts:
            for sy, sx in ((dy, dx), (-dy, -dx)):
                ys = slice(max(sy, 0), h + min(sy, 0))
                yd = slice(max(-sy, 0), h + min(-sy, 0))
                xs = slice(max(sx, 0), w + min(sx, 0))
                xd = slice(max(-sx, 0), w + min(-sx, 0))
                np.minimum(cur[yd, xd], lab[ys, xs], out=cur[yd, xd])
        cur = np.where(mask, cur, big)
        flat = cur.ravel()
        fg = flat < big
        # pointer jumping: follow each label to its own label until stable
        for _ in range(8):
            nxt = flat.copy()
            nxt[fg] = flat[flat[fg]]
            if np.array_equal(nxt, flat):
                break
            flat = nxt
        cur = flat.reshape(h, w)
        lab = cur
        if np.array_equal(lab, prev):
            break
    out = np.zeros((h, w), dtype=np.int32)
    roots = np.unique(lab[mask])
    out[mask] = np.searchsorted(roots, lab[mask]) + 1
    return out, len(roots)


@dataclass
class Blob:
    label: int
    area: int
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    cx: float
    cy: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def blobs(labels: np.ndarray, count: int) -> list[Blob]:
    if count == 0:
        return []
    ys, xs = np.nonzero(labels)
    ls = labels[ys, xs]
    area = np.bincount(ls, minlength=count + 1)
    sx = np.bincount(ls, weights=xs, minlength=count + 1)
    sy = np.bincount(ls, weights=ys, minlength=count + 1)
    x0 = np.full(count + 1, np.iinfo(np.int32).max)
    y0 = x0.copy()
    x1 = np.full(count + 1, -1)
    y1 = x1.copy()
    np.minimum.at(x0, ls, xs)
    np.minimum.at(y0, ls, ys)
    np.maximum.at(x1, ls, xs)
    np.maximum.at(y1, ls, ys)
    return [
        Blob(i, int(area[i]), int(x0[i]), int(y0[i]), int(x1[i]) + 1, int(y1[i]) + 1,
             float(sx[i] / area[i]), float(sy[i] / area[i]))
        for i in range(1, count + 1)
    ]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def edge_pixels(mask: np.ndarray) -> np.ndarray:
    """Opaque pixels with at least one transparent 4-neighbour (image border counts as empty)."""
    p = np.pad(mask, 1, constant_values=False)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~interior


def outline_color(arr: np.ndarray, mask: np.ndarray) -> tuple[int, int, int] | None:
    """Dominant dark colour on the silhouette edge, or None when the edge is not dark."""
    edge = edge_pixels(mask)
    if not edge.any():
        return None
    cols = arr[edge][:, :3]
    (color, n), = Counter(map(tuple, cols.tolist())).most_common(1)
    luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    if n / len(cols) < 0.35 or luma > 90:
        return None
    return tuple(int(c) for c in color)
