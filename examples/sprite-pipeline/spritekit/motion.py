"""Motion capture -> per-view 2D bone motion.

Pipeline (walk):
  1. Forward kinematics on the BVH.
  2. Face the pelvis towards +Z and remove forward travel with a linear fit
     (sway and bob are kept).
  3. One gait cycle = two consecutive upward zero crossings of
     (left ankle z - right ankle z) near the middle of the take.
  4. Remove end drift, resample to N frames.
  5. Symmetrize: average every joint with its mirrored partner half a cycle later.
  6. Orthographic camera (yaw per view, fixed pitch); per bone: screen angle,
     delta from its cycle mean, foreshortening and depth.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import bvh

# logical joint -> candidate BVH names (CMU first, then common alternatives)
JOINT_NAMES = {
    "pelvis": ["Hips", "Hip", "pelvis"],
    "chest": ["Spine1", "Spine2", "Chest", "Spine"],
    "neck": ["Neck", "Neck1"],
    "head": ["Head"],
    "shoulder_L": ["LeftArm", "LeftUpperArm", "LShoulder"],
    "elbow_L": ["LeftForeArm", "LeftLowerArm", "LElbow"],
    "hand_L": ["LeftHand", "LWrist"],
    "hip_L": ["LeftUpLeg", "LeftUpperLeg", "LHip"],
    "knee_L": ["LeftLeg", "LeftLowerLeg", "LKnee"],
    "ankle_L": ["LeftFoot", "LAnkle"],
}
for _k in list(JOINT_NAMES):
    if _k.endswith("_L"):
        JOINT_NAMES[_k[:-2] + "_R"] = [re.sub(r"^Left", "Right", n) if n.startswith("Left") else "R" + n[1:]
                                       for n in JOINT_NAMES[_k]]

# rig bone -> (from joint, to joint); "_S" is a side placeholder
BONES = {
    "torso": ("pelvis", "neck"),
    "neck": ("neck", "head"),
    "upper_arm_S": ("shoulder_S", "elbow_S"),
    "forearm_S": ("elbow_S", "hand_S"),
    "thigh_S": ("hip_S", "knee_S"),
    "shin_S": ("knee_S", "ankle_S"),
}

VIEW_YAW = {"front": 0.0, "front_left": -45.0, "side_left": -90.0, "back_left": -135.0, "back": 180.0}


def _norm(name: str) -> str:
    return re.sub(r"^.*[:|]", "", name).lower()


def resolve_joints(motion: bvh.Motion) -> dict[str, int]:
    names = {_norm(n): i for i, n in enumerate(motion.names)}
    out = {}
    for key, candidates in JOINT_NAMES.items():
        for c in candidates:
            if c.lower() in names:
                out[key] = names[c.lower()]
                break
        else:
            raise ValueError(f"BVH has no joint for {key} (tried {candidates})")
    return out


def _rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _circular_mean(a: np.ndarray) -> float:
    return float(np.angle(np.exp(1j * a).mean()))


@dataclass
class Cycle:
    """One looping cycle of 3D joint positions (Y up, facing +Z)."""
    positions: dict[str, np.ndarray]  # logical joint -> (N, 3)
    frame_ms: float
    source_frames: tuple[float, float]  # cycle start/end in source frame units

    @property
    def n(self) -> int:
        return next(iter(self.positions.values())).shape[0]


def _upward_crossings(s: np.ndarray) -> list[float]:
    out = []
    for i in range(1, len(s)):
        if s[i - 1] < 0 <= s[i]:
            out.append(i - 1 + (-s[i - 1]) / (s[i] - s[i - 1]))
    return out


def _sample(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    i0 = np.clip(np.floor(t).astype(int), 0, len(arr) - 2)
    w = (t - i0)[:, None]
    return arr[i0] * (1 - w) + arr[i0 + 1] * w


def extract_cycle(motion: bvh.Motion, frames: int = 12, symmetrize: bool = True) -> Cycle:
    """Extract one symmetric, drift-free walk cycle resampled to ``frames`` frames."""
    if frames % 2:
        raise ValueError("frames must be even so the half-cycle mirror lands on a frame")
    pos = bvh.forward_kinematics(motion)
    j = resolve_joints(motion)
    p = {k: pos[:, i] for k, i in j.items()}

    # face +Z: forward = cross(right->left hip vector, up)
    lr = (p["hip_L"] - p["hip_R"]).mean(axis=0)
    fwd = np.cross(lr, [0, 1, 0])
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    rot = _rot_y(-yaw)
    p = {k: v @ rot.T for k, v in p.items()}

    # remove forward travel (linear fit of pelvis x/z), keep sway and bob
    t = np.arange(len(p["pelvis"]), dtype=float)
    trend = np.zeros((len(t), 3))
    for axis in (0, 2):
        a, b = np.polyfit(t, p["pelvis"][:, axis], 1)
        trend[:, axis] = a * t + b
    p = {k: v - trend for k, v in p.items()}

    s = p["ankle_L"][:, 2] - p["ankle_R"][:, 2]
    cross = _upward_crossings(s)
    if len(cross) < 2:
        raise ValueError("could not find a full gait cycle (need two upward crossings of left-right ankle z)")
    m = max(len(cross) // 2 - 1, 0)
    t0, t1 = cross[m], cross[m + 1]
    ts = t0 + (t1 - t0) * np.arange(frames) / frames
    out = {}
    for k, v in p.items():
        samples = _sample(v, ts)
        drift = _sample(v, np.array([t1]))[0] - _sample(v, np.array([t0]))[0]
        out[k] = samples - drift * (np.arange(frames) / frames)[:, None]
    centre_x = out["pelvis"][:, 0].mean()
    for v in out.values():
        v[:, 0] -= centre_x
    if symmetrize:
        half = frames // 2
        sym = {}
        for k, v in out.items():
            partner = k[:-2] + ("_R" if k.endswith("_L") else "_L") if k[-2:] in ("_L", "_R") else k
            mirrored = np.roll(out[partner], -half, axis=0) * np.array([-1.0, 1.0, 1.0])
            sym[k] = 0.5 * (v + mirrored)
        out = sym
    return Cycle(out, (t1 - t0) * motion.frame_time * 1000.0 / frames, (t0, t1))


@dataclass
class BoneTrack:
    delta: np.ndarray  # screen-angle delta from cycle mean (radians, y-down screen coords)
    mean: float  # circular mean screen angle
    foreshortening: np.ndarray
    depth: np.ndarray  # mean depth of the bone (bigger = farther from camera)
    length3d: float


@dataclass
class ViewMotion:
    view: str
    yaw: float
    pitch: float
    frame_ms: float
    bones: dict[str, BoneTrack] = field(default_factory=dict)  # rig names, sides "a"/"b" = screen left/right
    ankle_depth: dict[str, np.ndarray] = field(default_factory=dict)  # side -> depth relative to pelvis
    leg_length3d: float = 1.0


def project(cycle: Cycle, view: str, pitch: float = 15.0) -> ViewMotion:
    """Project a cycle through an orthographic camera and measure 2D bone tracks.

    Mocap sides are mapped to screen sides: the side whose limb projects further
    left becomes ``a`` (screen-left), matching how rig dots are labelled. In a
    profile view, where both sides project to the same x, ``a`` is the far side
    (the prompts ask for far limbs drawn to the left of the near ones).
    """
    yaw = VIEW_YAW[view]
    cam = _rot_x(pitch) @ _rot_y(yaw)
    proj = {}
    for k, v in cycle.positions.items():
        q = v @ cam.T
        proj[k] = np.stack([q[:, 0], -q[:, 1], -q[:, 2]], axis=1)  # screen x, screen y (down), depth
    side_map = {}
    for limb, joint in (("arm", "shoulder"), ("leg", "hip")):
        lx, rx = proj[f"{joint}_L"][:, 0].mean(), proj[f"{joint}_R"][:, 0].mean()
        ld, rd = proj[f"{joint}_L"][:, 2].mean(), proj[f"{joint}_R"][:, 2].mean()
        if abs(lx - rx) < 0.25 * abs(ld - rd):
            # profile view: both sides project to the same x, so the drawing
            # convention decides: the far limb is drawn on the screen-left
            left_is_a = ld > rd
        else:
            left_is_a = lx < rx
        side_map[limb] = {"a": "L", "b": "R"} if left_is_a else {"a": "R", "b": "L"}
    vm = ViewMotion(view, yaw, pitch, cycle.frame_ms)
    cos_p = math.cos(math.radians(pitch))
    for bone, (j0, j1) in BONES.items():
        sides = ["a", "b"] if "_S" in bone else [None]
        for side in sides:
            if side:
                mside = side_map["leg" if bone.startswith(("thigh", "shin")) else "arm"][side]
                a0, a1, name = j0.replace("_S", "_" + mside), j1.replace("_S", "_" + mside), bone.replace("_S", "_" + side)
            else:
                a0, a1, name = j0, j1, bone
            v2 = proj[a1][:, :2] - proj[a0][:, :2]
            ang = np.arctan2(v2[:, 1], v2[:, 0])
            mean = _circular_mean(ang)
            len3 = np.linalg.norm(cycle.positions[a1] - cycle.positions[a0], axis=1)
            fs = np.clip(np.linalg.norm(v2, axis=1) / np.maximum(len3, 1e-9) / cos_p, 0.3, 1.05)
            depth = 0.5 * (proj[a0][:, 2] + proj[a1][:, 2])
            vm.bones[name] = BoneTrack(wrap_angle(ang - mean), mean, fs, depth, float(len3.mean()))
    for side, mside in side_map["leg"].items():
        vm.ankle_depth[side] = proj[f"ankle_{mside}"][:, 2] - proj["pelvis"][:, 2]
    legs = [np.linalg.norm(cycle.positions[f"hip_{s}"] - cycle.positions[f"knee_{s}"], axis=1).mean()
            + np.linalg.norm(cycle.positions[f"knee_{s}"] - cycle.positions[f"ankle_{s}"], axis=1).mean() for s in "LR"]
    vm.leg_length3d = float(np.mean(legs))
    return vm
