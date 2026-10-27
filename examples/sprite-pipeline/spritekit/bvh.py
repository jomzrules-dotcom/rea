"""Minimal BVH (Biovision hierarchy) parser with vectorised forward kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Joint:
    name: str
    parent: int  # -1 for the root
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    channel_start: int = 0


@dataclass
class Motion:
    joints: list[Joint]
    frame_time: float
    data: np.ndarray  # (frames, channels)

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    def index(self, name: str) -> int:
        return self.names.index(name)


def parse(text: str) -> Motion:
    tokens = text.split()
    pos = 0
    joints: list[Joint] = []
    stack: list[int] = []
    n_channels = 0

    def take() -> str:
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    if take() != "HIERARCHY":
        raise ValueError("not a BVH file: missing HIERARCHY")
    while pos < len(tokens):
        t = take()
        if t in ("ROOT", "JOINT"):
            name = take()
            joints.append(Joint(name, stack[-1] if stack else -1, np.zeros(3)))
        elif t == "End":
            take()  # "Site"
            parent = stack[-1]
            joints.append(Joint(joints[parent].name + "_End", parent, np.zeros(3)))
        elif t == "{":
            stack.append(len(joints) - 1)
        elif t == "}":
            stack.pop()
        elif t == "OFFSET":
            joints[stack[-1]].offset = np.array([float(take()) for _ in range(3)])
        elif t == "CHANNELS":
            j = joints[stack[-1]]
            j.channel_start = n_channels
            j.channels = [take() for _ in range(int(take()))]
            n_channels += len(j.channels)
        elif t == "MOTION":
            break
        else:
            raise ValueError(f"unexpected BVH token {t!r}")
    if take() != "Frames:":
        raise ValueError("missing Frames:")
    frames = int(take())
    if take() != "Frame" or take() != "Time:":
        raise ValueError("missing Frame Time:")
    frame_time = float(take())
    values = np.array(tokens[pos:pos + frames * n_channels], dtype=float)
    if values.size != frames * n_channels:
        raise ValueError(f"expected {frames * n_channels} motion values, found {values.size}")
    return Motion(joints, frame_time, values.reshape(frames, n_channels))


def load(path: str | Path) -> Motion:
    return parse(Path(path).read_text())


def _axis_rotation(axis: str, deg: np.ndarray) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    o, z = np.ones_like(r), np.zeros_like(r)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def forward_kinematics(motion: Motion) -> np.ndarray:
    """Global joint positions, shape (frames, joints, 3).

    Local rotation is the product of channel rotations in file order
    (``Zrotation Yrotation Xrotation`` gives Rz @ Ry @ Rx).
    """
    f = motion.data.shape[0]
    nj = len(motion.joints)
    pos = np.zeros((f, nj, 3))
    rot = np.zeros((f, nj, 3, 3))
    for i, j in enumerate(motion.joints):
        local_r = np.broadcast_to(np.eye(3), (f, 3, 3)).copy()
        local_t = np.broadcast_to(j.offset, (f, 3)).copy()
        for k, ch in enumerate(j.channels):
            col = motion.data[:, j.channel_start + k]
            if ch.endswith("rotation"):
                local_r = local_r @ _axis_rotation(ch[0].upper(), col)
            elif ch.endswith("position"):
                local_t[:, "XYZ".index(ch[0].upper())] += col
        if j.parent < 0:
            rot[:, i] = local_r
            pos[:, i] = local_t
        else:
            pr = rot[:, j.parent]
            rot[:, i] = pr @ local_r
            pos[:, i] = pos[:, j.parent] + np.einsum("fij,fj->fi", pr, local_t)
    return pos
