"""Reverse-engineer an existing sprite sheet or animated GIF/APNG.

The report keeps four kinds of statement apart:

* ``observed``  - read directly from pixels or file metadata (sizes, colours, durations).
* ``derived``   - computed deterministically from observations (grid, anchors, duplicates).
* ``inferred``  - heuristics that can be wrong (row = animation, loops, suggested timing).
* ``unknown``   - facts the input cannot provide (e.g. timing of a plain PNG).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from . import image as im


@dataclass
class Frame:
    index: int
    x: int  # cell rectangle on the sheet
    y: int
    w: int
    h: int
    row: int
    col: int
    trim: tuple[int, int, int, int]  # opaque bbox relative to the cell (x0, y0, x1, y1)
    anchor: tuple[float, float]  # feet anchor relative to the cell
    duration_ms: int | None = None
    exact_hash: str = ""
    trim_hash: str = ""
    mirror_hash: str = ""
    duplicate_of: int | None = None
    mirror_of: int | None = None

    def to_json(self) -> dict:
        return {
            "index": self.index, "row": self.row, "col": self.col,
            "rect": [self.x, self.y, self.w, self.h],
            "trim": list(self.trim), "anchor": [round(self.anchor[0], 2), round(self.anchor[1], 2)],
            "duration_ms": self.duration_ms,
            "duplicate_of": self.duplicate_of, "mirror_of": self.mirror_of,
        }


@dataclass
class Analysis:
    width: int
    height: int
    layout: str  # "grid" | "packed" | "animation"
    cell: tuple[int, int] | None
    frames: list[Frame]
    observed: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    inferred: dict = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "size": [self.width, self.height], "layout": self.layout,
            "cell": list(self.cell) if self.cell else None,
            "frame_count": len(self.frames),
            "observed": self.observed, "derived": self.derived,
            "inferred": self.inferred, "unknown": self.unknown,
            "frames": [f.to_json() for f in self.frames],
        }


# ---------------------------------------------------------------------------
# pixel scale


def detect_pixel_scale(arr: np.ndarray, max_scale: int = 16, agree: float = 0.98) -> int:
    """Integer upscale factor of pixel art (1 when the art is native resolution).

    Every colour change of nearest-neighbour upscaled art falls on a multiple of
    the scale (plus one constant offset) on both axes.
    """
    a = arr.astype(np.int32)
    col_change = np.nonzero((a[:, 1:] != a[:, :-1]).any(axis=(0, 2)))[0] + 1
    row_change = np.nonzero((a[1:] != a[:-1]).any(axis=(1, 2)))[0] + 1
    if len(col_change) < 4 or len(row_change) < 4:
        return 1
    for k in range(max_scale, 1, -1):
        ok = True
        for changes in (col_change, row_change):
            _, counts = np.unique(changes % k, return_counts=True)
            if counts.max() / len(changes) < agree:
                ok = False
                break
        if ok:
            return k
    return 1


# ---------------------------------------------------------------------------
# sprites (islands merged across small gaps) and grid inference


def find_sprites(mask: np.ndarray, gap: int = 1) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of sprites: 8-connected islands merged when within ``gap`` px."""
    labels, n = im.label_components(mask)
    boxes = [(b.x0, b.y0, b.x1, b.y1) for b in im.blobs(labels, n)]
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for a_pos, i in enumerate(order):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in order[a_pos + 1:]:
            bx0, by0, bx1, by1 = boxes[j]
            if bx0 > ax1 + gap:
                break
            if by0 <= ay1 + gap and ay0 <= by1 + gap:
                parent[find(i)] = find(j)
    merged: dict[int, list[int]] = {}
    for i, b in enumerate(boxes):
        r = find(i)
        m = merged.setdefault(r, [b[0], b[1], b[2], b[3]])
        m[0], m[1] = min(m[0], b[0]), min(m[1], b[1])
        m[2], m[3] = max(m[2], b[2]), max(m[3], b[3])
    return sorted((tuple(v) for v in merged.values()), key=lambda b: (b[1], b[0]))


def _divisors(n: int, minimum: int) -> list[int]:
    return [d for d in range(max(1, minimum), n + 1) if n % d == 0]


def infer_grid(width: int, height: int, sprites: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    """Smallest uniform cell size that cuts no sprite and leaves no interior holes.

    Candidates are divisors of the sheet size. A cell boundary may never pass
    through a sprite; among valid grids, fewer empty cells before the last
    occupied cell (row-major) wins, then more cells. Returns None when only the
    trivial 1x1 grid fits (the sheet is then packed, not gridded).
    """
    if not sprites:
        return None
    max_w = max(b[2] - b[0] for b in sprites)
    max_h = max(b[3] - b[1] for b in sprites)
    best = None
    for cw in _divisors(width, max_w):
        cols = width // cw
        if any(b[0] // cw != (b[2] - 1) // cw for b in sprites):
            continue
        for ch in _divisors(height, max_h):
            rows = height // ch
            if cols * rows == 1:
                continue
            if any(b[1] // ch != (b[3] - 1) // ch for b in sprites):
                continue
            # rows may end early (animations of different lengths): only gaps
            # before a row's last occupied cell, and empty rows, count as holes
            occupied: dict[int, set[int]] = {}
            for b in sprites:
                occupied.setdefault(b[1] // ch, set()).add(b[0] // cw)
            holes = sum(max(c) + 1 - len(c) for c in occupied.values())
            holes += max(occupied) + 1 - len(occupied)
            key = (holes, -(cols * rows))
            if best is None or key < best[0]:
                best = (key, (cw, ch))
    return best[1] if best else None


# ---------------------------------------------------------------------------
# frame measurements


def feet_anchor(mask: np.ndarray) -> tuple[float, float]:
    """Bottom-centre anchor: mean x of the lowest two opaque rows, y = bottom edge."""
    ys = np.nonzero(mask.any(axis=1))[0]
    if len(ys) == 0:
        return (mask.shape[1] / 2, float(mask.shape[0]))
    bottom = ys.max()
    band = mask[max(bottom - 1, 0): bottom + 1]
    xs = np.nonzero(band)[1]
    return (float(xs.mean()) + 0.5, float(bottom + 1))


def _digest(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes() + str(a.shape).encode(), digest_size=10).hexdigest()


def _normalised(cell: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = cell.copy()
    out[~mask] = 0
    return out


def _measure(sheet: np.ndarray, mask: np.ndarray, rects: list[tuple[int, int, int, int, int, int]]) -> list[Frame]:
    frames = []
    for i, (x, y, w, h, row, col) in enumerate(rects):
        cm = mask[y:y + h, x:x + w]
        cell = _normalised(sheet[y:y + h, x:x + w], cm)
        box = im.bbox(cm) or (0, 0, 0, 0)
        trimmed = cell[box[1]:box[3], box[0]:box[2]]
        frames.append(Frame(
            index=i, x=x, y=y, w=w, h=h, row=row, col=col, trim=box, anchor=feet_anchor(cm),
            exact_hash=_digest(cell), trim_hash=_digest(trimmed), mirror_hash=_digest(trimmed[:, ::-1]),
        ))
    first_exact: dict[str, int] = {}
    first_trim: dict[str, int] = {}
    for f in frames:
        if f.exact_hash in first_exact:
            f.duplicate_of = first_exact[f.exact_hash]
        else:
            first_exact[f.exact_hash] = f.index
        first_trim.setdefault(f.trim_hash, f.index)
    for f in frames:
        src = first_trim.get(f.mirror_hash)
        if src is not None and src != f.index and f.mirror_hash != f.trim_hash:
            f.mirror_of = src
    return frames


def _frame_difference(sheet: np.ndarray, mask: np.ndarray, a: Frame, b: Frame) -> float:
    """Mean per-pixel RGBA difference of two cells aligned on their feet anchors."""
    dx = int(round(b.anchor[0] - a.anchor[0]))
    dy = int(round(b.anchor[1] - a.anchor[1]))
    ca = _normalised(sheet[a.y:a.y + a.h, a.x:a.x + a.w], mask[a.y:a.y + a.h, a.x:a.x + a.w]).astype(np.int32)
    cb = _normalised(sheet[b.y:b.y + b.h, b.x:b.x + b.w], mask[b.y:b.y + b.h, b.x:b.x + b.w]).astype(np.int32)
    h, w = min(a.h, b.h), min(a.w, b.w)
    ca = ca[:h, :w]
    cb = np.roll(cb[:h, :w], (-dy, -dx), axis=(0, 1))
    return float(np.abs(ca - cb).mean())


def _row_period(frames: list[Frame]) -> int | None:
    hashes = [f.exact_hash for f in frames]
    n = len(hashes)
    for p in range(1, n // 2 + 1):
        if all(hashes[i] == hashes[i + p] for i in range(n - p)):
            return p
    return None


def _summarise(sheet: np.ndarray, mask: np.ndarray, an: Analysis, bg: dict) -> None:
    opaque = sheet[mask]
    colours = Counter(map(tuple, opaque[:, :3].tolist()))
    an.observed.update({
        "background": bg,
        "unique_colors": len(colours),
        "top_colors": [["#%02x%02x%02x" % c, n] for c, n in colours.most_common(8)],
        "semi_transparent_pixels": int(((sheet[..., 3] > 0) & (sheet[..., 3] < 255)).sum()),
    })
    oc = im.outline_color(sheet, mask)
    an.derived["outline_color"] = "#%02x%02x%02x" % oc if oc else None
    an.derived["pixel_scale"] = detect_pixel_scale(sheet)
    an.derived["duplicates"] = [[f.index, f.duplicate_of] for f in an.frames if f.duplicate_of is not None]
    an.derived["mirrors"] = [[f.index, f.mirror_of] for f in an.frames if f.mirror_of is not None]

    rows: dict[int, list[Frame]] = {}
    for f in an.frames:
        rows.setdefault(f.row, []).append(f)
    row_reports = []
    for r, fs in sorted(rows.items()):
        ax = np.array([f.anchor for f in fs])
        diffs = [_frame_difference(sheet, mask, fs[i], fs[i + 1]) for i in range(len(fs) - 1)]
        report = {
            "row": r, "frames": [f.index for f in fs],
            "anchor_drift_px": [round(float(ax[:, 0].std()), 2), round(float(ax[:, 1].std()), 2)],
            "repeat_period": _row_period(fs),
        }
        if len(fs) >= 3 and diffs:
            closure = _frame_difference(sheet, mask, fs[-1], fs[0])
            # a loop's wrap-around step looks like any other step; a one-shot clip
            # (attack, death) ends far from its first pose, beyond its largest step
            largest = max(diffs)
            report["loop_closure_ratio"] = round(closure / largest, 2) if largest else None
            report["loops"] = bool(closure <= 1.25 * largest)
        mirrored = [f.mirror_of for f in fs]
        if all(m is not None for m in mirrored):
            src_rows = {an.frames[m].row for m in mirrored}
            if len(src_rows) == 1:
                report["mirror_of_row"] = src_rows.pop()
        row_reports.append(report)
    an.inferred["rows_as_animations"] = row_reports
    an.inferred["note"] = (
        "Each row is assumed to be one animation or direction. anchor_drift_px is the spread of the "
        "feet anchor: above ~1 px in an idle row means the sprite slides; walk rows vary naturally as "
        "the feet alternate. loop_closure_ratio (wrap-around step / largest step) <= 1.25 suggests the row loops."
    )


# ---------------------------------------------------------------------------
# public entry points


def analyze_sheet(sheet: np.ndarray, cell: tuple[int, int] | None = None, gap: int = 1,
                  tolerance: float = 24.0) -> Analysis:
    mask, bg = im.foreground_mask(sheet, tolerance)
    h, w = mask.shape
    sprites = find_sprites(mask, gap)
    grid = cell or infer_grid(w, h, sprites)
    rects = []
    if grid:
        cw, ch = grid
        cols = w // cw
        occupied = sorted({(b[1] // ch, b[0] // cw) for b in sprites}) if not cell else [
            (r, c) for r in range(h // ch) for c in range(cols) if mask[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].any()
        ]
        rects = [(c * cw, r * ch, cw, ch, r, c) for r, c in occupied]
        layout = "grid"
    else:
        # packed atlas: one frame per sprite, rows grouped by vertical overlap
        layout = "packed"
        row_id, row_bottom = -1, -1
        for b in sprites:
            if b[1] >= row_bottom:
                row_id += 1
                col = 0
            rects.append((b[0], b[1], b[2] - b[0], b[3] - b[1], row_id, col))
            row_bottom = max(row_bottom, b[3])
            col += 1
    an = Analysis(w, h, layout, grid, _measure(sheet, mask, rects))
    an.observed["sprites_found"] = len(sprites)
    an.derived["grid_source"] = "caller" if cell else ("inferred" if grid else "packed")
    if grid:
        an.derived["grid"] = {"columns": w // grid[0], "rows": h // grid[1]}
    _summarise(sheet, mask, an, bg)
    an.unknown.append("frame timing: a still image carries no durations")
    an.inferred["suggested_timing_ms"] = 100
    return an


def load_animation(path: str | Path) -> tuple[list[np.ndarray], list[int], dict]:
    """Composited RGBA frames and per-frame durations of a GIF/APNG/WebP."""
    frames, durations = [], []
    with Image.open(path) as src:
        info = {"format": src.format, "loop": src.info.get("loop")}
        for fr in ImageSequence.Iterator(src):
            frames.append(np.array(fr.convert("RGBA")))
            durations.append(int(fr.info.get("duration", 0) or 0))
    return frames, durations, info


def analyze_animation(path: str | Path) -> tuple[Analysis, np.ndarray]:
    """Turn an animated image into a sheet and analyse it, preserving real durations."""
    from .sheet import frames_to_sheet

    frames, durations, info = load_animation(path)
    scale = detect_pixel_scale(frames[0])
    sheet, cell = frames_to_sheet(frames, columns=len(frames), keep_background=False, align=scale)
    an = analyze_sheet(sheet, cell=cell)
    an.layout = "animation"
    an.unknown = [u for u in an.unknown if not u.startswith("frame timing")]
    an.inferred.pop("suggested_timing_ms", None)
    for f in an.frames:
        f.duration_ms = durations[f.col] if f.col < len(durations) else None
    an.observed.update({"source_format": info["format"], "loop": info["loop"],
                        "durations_ms": durations, "total_ms": sum(durations)})
    if durations and all(d > 0 for d in durations):
        an.derived["fps_equivalent"] = round(1000 * len(durations) / sum(durations), 2)
    return an, sheet


def to_aseprite(an: Analysis, image_name: str) -> dict:
    """Aseprite-style JSON for the analysed sheet (rows become frame tags)."""
    frames, tags = [], []
    default = an.inferred.get("suggested_timing_ms", 100)
    for r in an.inferred["rows_as_animations"]:
        start = len(frames)
        for i in r["frames"]:
            f = an.frames[i]
            frames.append({
                "filename": f"row{f.row} {f.col}",
                "frame": {"x": f.x, "y": f.y, "w": f.w, "h": f.h}, "rotated": False, "trimmed": False,
                "spriteSourceSize": {"x": 0, "y": 0, "w": f.w, "h": f.h}, "sourceSize": {"w": f.w, "h": f.h},
                "duration": f.duration_ms if f.duration_ms else default,
                "pivot": {"x": round(f.anchor[0] / f.w, 4), "y": round(f.anchor[1] / f.h, 4)},
            })
        tags.append({"name": f"row{r['row']}", "from": start, "to": len(frames) - 1, "direction": "forward"})
    return {"frames": frames, "meta": {"app": "spritekit-analyze", "image": image_name, "format": "RGBA8888",
                                       "size": {"w": an.width, "h": an.height}, "scale": "1", "frameTags": tags}}


def describe(an: Analysis) -> str:
    """Short human-readable summary."""
    lines = [f"{an.width}x{an.height} {an.layout} sheet, {len(an.frames)} frames"]
    if an.cell:
        g = an.derived.get("grid", {})
        lines.append(f"cell {an.cell[0]}x{an.cell[1]} ({g.get('columns')} cols x {g.get('rows')} rows, {an.derived['grid_source']})")
    lines.append(f"pixel scale x{an.derived['pixel_scale']}, {an.observed['unique_colors']} colours, outline {an.derived['outline_color']}")
    for r in an.inferred["rows_as_animations"]:
        bits = [f"row {r['row']}: {len(r['frames'])} frames", f"anchor drift {r['anchor_drift_px']}"]
        if "loops" in r:
            bits.append("loops" if r["loops"] else f"no loop (closure {r['loop_closure_ratio']})")
        if r.get("repeat_period"):
            bits.append(f"repeats every {r['repeat_period']}")
        if "mirror_of_row" in r:
            bits.append(f"mirror of row {r['mirror_of_row']}")
        lines.append("  " + ", ".join(bits))
    if an.derived["duplicates"]:
        lines.append(f"duplicate frames (hold frames): {len(an.derived['duplicates'])}")
    if "durations_ms" in an.observed:
        lines.append(f"durations {an.observed['durations_ms']} ms (~{an.derived.get('fps_equivalent')} fps)")
    for u in an.unknown:
        lines.append(f"unknown: {u}")
    return "\n".join(lines)

