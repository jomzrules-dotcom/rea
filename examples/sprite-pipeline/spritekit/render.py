"""Skeletal 2D renderer: affine bone chains, soft joints, inverse-mapped sampling.

A bone's world matrix is

    M_b = M_parent . T(pivot) . R(rest + anim) . S(foreshortening along axis) . T(-pivot)

where pivots and axes live in the rig's *assembled rest* coordinates. Every
output pixel is mapped back into its part image (nearest sampling), so no holes
appear when parts rotate or stretch.

Soft joints (cheap skinning): child pixels within ``soft`` px of their pivot
receive only part of the animated rotation (weight 0.5 at the joint rising to 1
at the radius); with ``parent_soft`` the parent's pixels near that joint follow
the child by 0.5 falling to 0. Rotation about the pivot preserves distance, so
the per-pixel inverse map is exact for rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def translate(x: float, y: float) -> np.ndarray:
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)


def rotate(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def axis_scale(f: float, axis: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, float)
    m = np.eye(3)
    m[:2, :2] += (f - 1.0) * np.outer(a, a)
    return m


def about(pivot: np.ndarray, m: np.ndarray) -> np.ndarray:
    return translate(*pivot) @ m @ translate(-pivot[0], -pivot[1])


@dataclass
class Pose:
    angles: dict[str, float] = field(default_factory=dict)  # animated local rotation (radians)
    scales: dict[str, float] = field(default_factory=dict)  # foreshortening along the bone axis
    offset: tuple[float, float] = (0.0, 0.0)  # whole-body translation (ground pin, bob)
    # bone -> grid (canvas px per output px): the bone's displacement from its rest
    # position is rounded to whole output pixels and inherited stretch is dropped,
    # so rigid detail (a face) moves in clean pixel steps instead of shimmering
    snap: dict[str, float] = field(default_factory=dict)


def bone_matrices(rig, pose: Pose, base: np.ndarray | None = None) -> dict[str, np.ndarray]:
    base = translate(*pose.offset) if base is None else base @ translate(*pose.offset)
    rest = bone_matrices(rig, Pose(offset=pose.offset), base) if pose.snap else {}
    out: dict[str, np.ndarray] = {}
    for name, b in rig.bones.items():
        parent = out[b.parent] if b.parent else base
        local = rotate(b.rest + pose.angles.get(name, 0.0)) @ axis_scale(pose.scales.get(name, 1.0), b.axis)
        m = parent @ about(b.pivot, local)
        grid = pose.snap.get(name)
        if grid:
            # snapped bones stay rigid: drop inherited stretch, keep only rotation
            angle = math.atan2(m[1, 0], m[0, 0])
            pivot_world = transform_points(m, b.pivot[None, :])[0]
            m = translate(*pivot_world) @ rotate(angle) @ translate(-b.pivot[0], -b.pivot[1])
            here = transform_points(m, b.pivot[None, :])[0]
            home = transform_points(rest[name], b.pivot[None, :])[0]
            target = home + np.round((here - home) / grid) * grid
            m = translate(*(target - here)) @ m
        out[name] = m
    return out


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:2, :2].T + m[:2, 2]


def _part_region(m_list: list[np.ndarray], part, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = part.image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) + part.origin
    pts = np.concatenate([transform_points(m, corners) for m in m_list])
    x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, size[0]), min(y1, size[1])
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _apply(m: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]


def _rotate_about(x, y, px, py, theta):
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = x - px, y - py
    return px + c * dx - s * dy, py + s * dx + c * dy


def render(rig, pose: Pose, size: tuple[int, int], order: list[str], base: np.ndarray | None = None) -> np.ndarray:
    """Render the rig into a (h, w, 4) uint8 canvas, drawing bones in ``order``."""
    w, h = size
    canvas = np.zeros((h, w, 4), np.float32)
    mats = bone_matrices(rig, pose, base)
    by_bone: dict[str, list] = {}
    for p in rig.parts:
        by_bone.setdefault(p.bone, []).append(p)
    children: dict[str, list[str]] = {}
    for name, b in rig.bones.items():
        if b.parent:
            children.setdefault(b.parent, []).append(name)
    for bone_name in order:
        bone = rig.bones[bone_name]
        for part in by_bone.get(bone_name, []):
            m = mats[bone_name]
            soft = bone.soft > 0 and bone.parent is not None
            mp = mats[bone.parent] if bone.parent else None
            region = _part_region([m] + ([mp] if soft else []), part, size)
            if region is None:
                continue
            x0, y0, x1, y1 = region
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            anim = pose.angles.get(bone_name, 0.0)
            if soft and anim != 0.0:
                qx, qy = _apply(np.linalg.inv(mp), gx, gy)
                px, py = bone.pivot
                d = np.hypot(qx - px, qy - py)
                wgt = np.clip(0.5 + 0.5 * d / bone.soft, 0.5, 1.0)
                theta = bone.rest + wgt * anim
                # inverse of T(p) R(theta) S T(-p): rotate back, then undo the axis scale
                lx, ly = _rotate_about(qx, qy, px, py, -theta)
                inv_s = np.linalg.inv(axis_scale(pose.scales.get(bone_name, 1.0), bone.axis))
                lx, ly = _apply(about(bone.pivot, inv_s), lx, ly)
            else:
                lx, ly = _apply(np.linalg.inv(m), gx, gy)
            for child in children.get(bone_name, []):
                cb = rig.bones[child]
                ca = pose.angles.get(child, 0.0)
                if not cb.parent_soft or cb.soft <= 0 or ca == 0.0:
                    continue
                d = np.hypot(lx - cb.pivot[0], ly - cb.pivot[1])
                wgt = np.clip(0.5 * (1.0 - d / cb.soft), 0.0, 0.5)
                lx, ly = _rotate_about(lx, ly, cb.pivot[0], cb.pivot[1], -wgt * ca)
            ix = np.floor(lx - part.origin[0]).astype(np.int64)
            iy = np.floor(ly - part.origin[1]).astype(np.int64)
            ph, pw = part.image.shape[:2]
            ok = (ix >= 0) & (iy >= 0) & (ix < pw) & (iy < ph)
            src = np.zeros(ix.shape + (4,), np.float32)
            src[ok] = part.image[iy[ok], ix[ok]]
            a = src[..., 3:4] / 255.0
            dst = canvas[y0:y1, x0:x1]
            out_a = a + dst[..., 3:4] / 255.0 * (1 - a)
            rgb = (src[..., :3] * a + dst[..., :3] * (dst[..., 3:4] / 255.0) * (1 - a)) / np.maximum(out_a, 1e-6)
            canvas[y0:y1, x0:x1, :3] = rgb
            canvas[y0:y1, x0:x1, 3:4] = out_a * 255.0
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
