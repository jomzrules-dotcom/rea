"""Sheet packing, Aseprite-compatible metadata and GIF previews."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from . import image as im


@dataclass
class Animation:
    name: str
    frames: list[np.ndarray]  # RGBA, all the same size
    durations_ms: list[int]
    pivot: tuple[float, float] | None = None  # feet anchor in frame pixels
    loop: bool = True


def frames_to_sheet(frames: list[np.ndarray], columns: int | None = None, keep_background: bool = False,
                    padding: int = 1, align: int = 1) -> tuple[np.ndarray, tuple[int, int]]:
    """Lay equal-size frames on a grid, cropped to their *common* bounding box.

    Cropping every frame to the union box keeps relative motion (bob, sway,
    travel) intact; trimming and re-centring each frame separately is the
    classic cause of jittering sprites. ``align`` snaps the crop to multiples of
    an upscaled pixel-art grid so the art's pixel scale survives packing.
    """
    if not frames:
        raise ValueError("no frames")
    if keep_background:
        masks = [np.ones(f.shape[:2], bool) for f in frames]
    else:
        alpha_based = any((f[..., 3] == 0).any() for f in frames)
        if alpha_based:
            masks = [f[..., 3] > 0 for f in frames]
        else:
            bg = im.border_color(frames[0])
            masks = [im.color_distance(f, bg) > 24 for f in frames]
    union = np.zeros(frames[0].shape[:2], bool)
    for m in masks:
        union |= m
    box = im.bbox(union) or (0, 0, 1, 1)
    x0, y0 = max(box[0] - padding, 0), max(box[1] - padding, 0)
    x1, y1 = min(box[2] + padding, union.shape[1]), min(box[3] + padding, union.shape[0])
    if align > 1:
        x0, y0 = x0 - x0 % align, y0 - y0 % align
        x1 = min(x1 + (-x1) % align, union.shape[1])
        y1 = min(y1 + (-y1) % align, union.shape[0])
    cw, ch = x1 - x0, y1 - y0
    columns = columns or len(frames)
    rows = -(-len(frames) // columns)
    sheet = np.zeros((rows * ch, columns * cw, 4), np.uint8)
    for i, (f, m) in enumerate(zip(frames, masks)):
        cell = f[y0:y1, x0:x1].copy()
        cell[~m[y0:y1, x0:x1]] = 0
        r, c = divmod(i, columns)
        sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
    return sheet, (cw, ch)


def pack(animations: list[Animation], image_name: str = "sheet.png") -> tuple[np.ndarray, dict]:
    """One row per animation; returns the sheet and Aseprite-style JSON metadata."""
    fw, fh = animations[0].frames[0].shape[1], animations[0].frames[0].shape[0]
    cols = max(len(a.frames) for a in animations)
    sheet = np.zeros((fh * len(animations), fw * cols, 4), np.uint8)
    frames_meta, tags = [], []
    for r, anim in enumerate(animations):
        start = len(frames_meta)
        for c, (fr, d) in enumerate(zip(anim.frames, anim.durations_ms)):
            if fr.shape[:2] != (fh, fw):
                raise ValueError(f"{anim.name} frame {c} is {fr.shape[1]}x{fr.shape[0]}, expected {fw}x{fh}")
            sheet[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = fr
            entry = {
                "filename": f"{anim.name} {c}",
                "frame": {"x": c * fw, "y": r * fh, "w": fw, "h": fh},
                "rotated": False, "trimmed": False,
                "spriteSourceSize": {"x": 0, "y": 0, "w": fw, "h": fh},
                "sourceSize": {"w": fw, "h": fh},
                "duration": int(d),
            }
            if anim.pivot:
                entry["pivot"] = {"x": round(anim.pivot[0] / fw, 4), "y": round(anim.pivot[1] / fh, 4)}
            frames_meta.append(entry)
        tags.append({"name": anim.name, "from": start, "to": len(frames_meta) - 1,
                     "direction": "forward", "repeat": "0" if anim.loop else "1"})
    meta = {
        "frames": frames_meta,
        "meta": {
            "app": "spritekit", "version": __version__, "image": image_name, "format": "RGBA8888",
            "size": {"w": sheet.shape[1], "h": sheet.shape[0]}, "scale": "1", "frameTags": tags,
        },
    }
    return sheet, meta


def write_sheet(animations: list[Animation], out_png: str | Path, out_json: str | Path | None = None) -> dict:
    out_png = Path(out_png)
    sheet, meta = pack(animations, out_png.name)
    im.save_rgba(sheet, out_png)
    if out_json:
        Path(out_json).write_text(json.dumps(meta, indent=1))
    return meta


def write_gif(frames: list[np.ndarray], durations_ms: list[int], path: str | Path, scale: int = 4,
              background: tuple[int, int, int] = (44, 44, 54)) -> None:
    """Preview GIF on a flat background, nearest-neighbour upscaled."""
    out = []
    for f in frames:
        rgb = Image.new("RGB", (f.shape[1], f.shape[0]), background)
        rgb.paste(Image.fromarray(f, "RGBA"), (0, 0), Image.fromarray(f[..., 3], "L"))
        out.append(rgb.resize((f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST))
    # GIF stores centiseconds; round to 10 ms so playback matches the sheet metadata
    durs = [max(20, int(round(d / 10.0)) * 10) for d in durations_ms]
    out[0].save(path, save_all=True, append_images=out[1:], duration=durs, loop=0, disposal=2, optimize=False)
