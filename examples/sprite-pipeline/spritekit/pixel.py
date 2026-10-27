"""Hi-res render -> pixel-art frame.

* a fixed fractional source window, so the character's feet anchor lands on the
  same output pixel in every frame;
* alpha from area coverage with a hard threshold (no semi-transparent fringe);
* colour either by premultiplied BOX averaging (``downscale``) or by the
  dominant palette colour of each block (``downscale_dominant``). Averaging
  turns small features such as eyes into grey mush; the dominant colour keeps
  them crisp, and dark "ink" colours win once they cover ``ink_share`` of a block
  so outlines and pupils survive;
* one median-cut palette shared by every frame of every animation/direction
  (per-frame palettes make colours flicker);
* a crisp 1 px silhouette outline.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import image as im


def downscale(hires: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int],
              alpha_threshold: int = 110) -> np.ndarray:
    """Area-average ``box`` of ``hires`` into ``size`` with premultiplied alpha."""
    a = hires[..., 3].astype(np.float32) / 255.0
    chans = [hires[..., i].astype(np.float32) * a for i in range(3)] + [a * 255.0]
    out = [np.array(Image.fromarray(c, "F").resize(size, Image.BOX, box=box)) for c in chans]
    alpha = out[3]
    opaque = alpha >= alpha_threshold
    res = np.zeros((size[1], size[0], 4), np.uint8)
    safe = np.maximum(alpha / 255.0, 1e-6)
    for i in range(3):
        res[..., i] = np.clip(np.rint(out[i] / safe), 0, 255)
    res[..., 3] = np.where(opaque, 255, 0)
    res[~opaque, :3] = 0
    return res


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def palette_indices(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Nearest-palette index per pixel (-1 where transparent), via unique colours."""
    idx = np.full(frame.shape[:2], -1, np.int64)
    mask = frame[..., 3] > 0
    if not mask.any():
        return idx
    keys, inverse = np.unique(_pack(frame[mask][:, :3]), return_inverse=True)
    cols = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], 1)
    d = ((cols[:, None, :] - palette[None, :, :].astype(np.int64)) ** 2).sum(-1)
    idx[mask] = d.argmin(1)[inverse.ravel()]
    return idx


def downscale_dominant(hires: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int],
                       palette: np.ndarray, alpha_threshold: int = 110, ink_share: float = 0.3,
                       ink_luma: float = 60.0) -> np.ndarray:
    """Per output pixel: coverage decides alpha, the most common palette colour decides RGB."""
    fw, fh = size
    x0, y0, x1, y1 = box
    h, w = hires.shape[:2]
    cols = np.floor((np.arange(w) + 0.5 - x0) * fw / (x1 - x0)).astype(np.int64)
    rows = np.floor((np.arange(h) + 0.5 - y0) * fh / (y1 - y0)).astype(np.int64)
    out_idx = rows[:, None] * fw + cols[None, :]
    inside = (rows[:, None] >= 0) & (rows[:, None] < fh) & (cols[None, :] >= 0) & (cols[None, :] < fw)
    pal_idx = palette_indices(hires, palette)
    k = len(palette)
    sel = inside & (pal_idx >= 0)
    hist = np.bincount(out_idx[sel] * k + pal_idx[sel], minlength=fw * fh * k).reshape(fh * fw, k).astype(float)
    area = np.bincount(out_idx[inside], minlength=fw * fh).astype(float)
    coverage = hist.sum(1) / np.maximum(area, 1)
    opaque = coverage * 255 >= alpha_threshold
    luma = palette.astype(float) @ np.array([0.299, 0.587, 0.114])
    ink = luma < ink_luma
    choice = hist.argmax(1)
    if ink.any():
        ink_hist = np.where(ink[None, :], hist, 0)
        ink_best = ink_hist.argmax(1)
        ink_cov = ink_hist.max(1) / np.maximum(hist.sum(1), 1)
        choice = np.where(ink_cov >= ink_share, ink_best, choice)
    res = np.zeros((fh * fw, 4), np.uint8)
    res[opaque, :3] = palette[choice[opaque]]
    res[opaque, 3] = 255
    return res.reshape(fh, fw, 4)


def shared_palette(frames: list[np.ndarray], colors: int = 36, reserve: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    """Median-cut palette over the opaque pixels of all frames; ``reserve`` colours are always kept."""
    reserve = reserve or []
    px = np.concatenate([f[f[..., 3] > 0][:, :3] for f in frames]) if frames else np.zeros((0, 3), np.uint8)
    if len(px) == 0:
        return np.array(reserve or [(0, 0, 0)], np.uint8)
    strip = Image.fromarray(px.reshape(1, -1, 3).astype(np.uint8), "RGB")
    n = max(1, colors - len(reserve))
    q = strip.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[: 3 * n], np.uint8).reshape(-1, 3)
    used = np.unique(np.array(q)).astype(int)
    pal = pal[used[used < len(pal)]]
    if reserve:
        pal = np.concatenate([np.array(reserve, np.uint8), pal])
    return np.unique(pal, axis=0)


def apply_palette(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = frame.copy()
    mask = frame[..., 3] > 0
    if not mask.any():
        return out
    px = frame[mask][:, :3].astype(np.int32)
    d = ((px[:, None, :] - palette[None, :, :].astype(np.int32)) ** 2).sum(-1)
    out[mask, :3] = palette[d.argmin(1)]
    return out


def add_outline(frame: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    out[im.edge_pixels(frame[..., 3] > 0), :3] = color
    return out
