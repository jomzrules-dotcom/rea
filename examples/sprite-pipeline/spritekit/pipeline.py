"""Picture -> animated sprite sheet.

exploded-view drawing --auto_rig--> rigs --mocap/keys--> poses --render--> hi-res frames
--downscale + shared palette + outline--> pixel frames --flip--> 4/8 directions --pack--> sheet + JSON
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import bvh, motion
from . import pixel as px
from . import render
from .rig import Rig, RigParams, auto_rig
from .sheet import Animation

DIRECTION = {"front": "s", "front_left": "sw", "side_left": "w", "back_left": "nw", "back": "n"}
FLIP = {"sw": "se", "w": "e", "nw": "ne"}
COMPASS = ["s", "sw", "w", "nw", "n", "ne", "e", "se"]  # sheet row order within each animation


@dataclass
class BuildParams:
    frame_size: tuple[int, int] = (64, 64)
    char_height: int = 44  # rest-pose height in output pixels, shared by every view
    foot_margin: int = 5  # px between the feet and the bottom of the frame
    colors: int = 36
    outline: tuple[int, int, int] | str | None = "auto"  # "auto", None or an RGB tuple
    alpha_threshold: int = 110
    downscale: str = "dominant"  # "dominant" (crisp features) or "box" (area average)
    pitch: float = 15.0
    walk_frames: int = 12
    idle_frames: int = 8
    idle_ms: int = 150
    amplitude: dict = field(default_factory=lambda: {"leg": 0.85, "arm": 1.0, "spine": 0.7, "neck": 0.6})
    elbow_bend_keep: float = 0.5
    ground_depth: float = 0.6
    bob_cap: float = 0.03  # max ground-pin offset as a fraction of figure height
    flip: bool = True
    snap_head: bool = True  # move the head in whole output pixels
    rig: RigParams = field(default_factory=RigParams)


@dataclass
class Clip:
    name: str
    poses: list[render.Pose]
    orders: list[list[str]]
    durations: list[int]


@dataclass
class BuildResult:
    animations: list[Animation]
    rigs: list[Rig]
    hires: dict[str, list[np.ndarray]]
    palette: np.ndarray
    report: dict


# ---------------------------------------------------------------------------
# draw order


def draw_order(rig: Rig, far_arm: str, far_leg: str) -> list[str]:
    near_arm = "b" if far_arm == "a" else "a"
    near_leg = "b" if far_leg == "a" else "a"
    base = [f"upper_arm_{far_arm}", f"forearm_{far_arm}", f"shin_{far_leg}", f"thigh_{far_leg}",
            f"shin_{near_leg}", f"thigh_{near_leg}", "neck", "torso", "head",
            f"upper_arm_{near_arm}", f"forearm_{near_arm}"]
    props: dict[str, list[str]] = {}
    for n, b in rig.bones.items():
        if n.startswith("prop"):
            props.setdefault(b.parent, []).append(n)
    out = []
    for n in base:
        out.append(n)
        out.extend(props.get(n, []))
    return out


# ---------------------------------------------------------------------------
# clips


def walk_clip(rig: Rig, vm: motion.ViewMotion, params: BuildParams) -> Clip:
    amp = params.amplitude
    n = len(vm.bones["torso"].delta)
    bend = {s: params.elbow_bend_keep * float(motion.wrap_angle(vm.bones[f"forearm_{s}"].mean - vm.bones[f"upper_arm_{s}"].mean))
            for s in "ab"}
    far_arm = "a" if vm.bones["upper_arm_a"].depth.mean() > vm.bones["upper_arm_b"].depth.mean() else "b"
    poses, orders = [], []
    for i in range(n):
        d = {k: float(t.delta[i]) for k, t in vm.bones.items()}
        world = {"torso": amp["spine"] * d["torso"], "neck": amp["neck"] * d["neck"]}
        for s in "ab":
            world[f"upper_arm_{s}"] = amp["arm"] * d[f"upper_arm_{s}"]
            world[f"forearm_{s}"] = amp["arm"] * d[f"forearm_{s}"]
            world[f"thigh_{s}"] = amp["leg"] * d[f"thigh_{s}"]
            world[f"shin_{s}"] = amp["leg"] * d[f"shin_{s}"]
        angles = {"torso": world["torso"], "neck": world["neck"] - world["torso"]}
        scales = {}
        for s in "ab":
            angles[f"upper_arm_{s}"] = world[f"upper_arm_{s}"] - world["torso"]
            angles[f"forearm_{s}"] = world[f"forearm_{s}"] - world[f"upper_arm_{s}"] + bend[s]
            angles[f"thigh_{s}"] = world[f"thigh_{s}"]
            angles[f"shin_{s}"] = world[f"shin_{s}"] - world[f"thigh_{s}"]
            for limb in ("upper_arm", "forearm", "thigh", "shin"):
                scales[f"{limb}_{s}"] = float(vm.bones[f"{limb}_{s}"].foreshortening[i])
        poses.append(render.Pose(angles, scales))
        leg_depth = {s: vm.bones[f"thigh_{s}"].depth[i] + vm.bones[f"shin_{s}"].depth[i] for s in "ab"}
        orders.append(draw_order(rig, far_arm, "a" if leg_depth["a"] > leg_depth["b"] else "b"))
    _ground_pin(rig, poses, vm, params)
    durations = _durations(vm.frame_ms, n)
    return Clip("walk", poses, orders, durations)


def _durations(frame_ms: float, n: int) -> list[int]:
    """Integer milliseconds whose total matches the real cycle length (no cumulative drift)."""
    edges = np.rint(np.arange(n + 1) * frame_ms).astype(int)
    return [int(v) for v in np.diff(edges)]


def _foot_contacts(rig: Rig, pose: render.Pose, depth: dict[str, float], k: float) -> float:
    mats = render.bone_matrices(rig, pose)
    ys = []
    for s in "ab":
        b = rig.bones[f"shin_{s}"]
        foot = render.transform_points(mats[f"shin_{s}"], b.end[None, :])[0]
        ys.append(foot[1] + depth[s] * k)
    return max(ys)


def _ground_pin(rig: Rig, poses: list[render.Pose], vm: motion.ViewMotion, params: BuildParams) -> None:
    """Move the body each frame so the lower foot touches the ground."""
    leg_px = np.mean([rig.bones[f"thigh_{s}"].length + rig.bones[f"shin_{s}"].length for s in "ab"])
    k = leg_px / vm.leg_length3d * math.sin(math.radians(vm.pitch)) * params.ground_depth
    rest = _foot_contacts(rig, render.Pose(), {"a": 0.0, "b": 0.0}, 0.0)
    rel = {s: vm.ankle_depth[s] - 0.5 * (vm.ankle_depth["a"] + vm.ankle_depth["b"]) for s in "ab"}
    dy = np.array([rest - _foot_contacts(rig, p, {s: rel[s][i] for s in "ab"}, k) for i, p in enumerate(poses)])
    dy = 0.25 * np.roll(dy, 1) + 0.5 * dy + 0.25 * np.roll(dy, -1)
    cap = params.bob_cap * rig.height
    dy = np.clip(dy, -cap, cap)
    for p, v in zip(poses, dy):
        p.offset = (0.0, float(v))


def idle_clip(rig: Rig, params: BuildParams) -> Clip:
    """Keyed breathing: chest scale, slight sink, arms lagging the chest."""
    n = params.idle_frames
    poses, orders = [], []
    order = draw_order(rig, "a", "a")
    for i in range(n):
        u = 2 * math.pi * i / n
        breath = math.sin(u)
        arm = 0.035 * math.sin(u - 0.6)
        angles = {"upper_arm_a": arm, "upper_arm_b": -arm,
                  "forearm_a": 0.6 * arm, "forearm_b": -0.6 * arm}
        sink = rig.height / params.char_height  # one output pixel
        poses.append(render.Pose(angles, {"torso": 1.0 + 0.018 * breath},
                                 (0.0, sink * (0.5 - 0.5 * math.cos(u)))))
        orders.append(order)
    return Clip("idle", poses, orders, [params.idle_ms] * n)


# ---------------------------------------------------------------------------
# build


def rig_drawings(drawings: np.ndarray | list[np.ndarray], views: list[str], params: RigParams | None = None) -> list[Rig]:
    """Rig one drawing holding every view (left to right), or one drawing per view."""
    for v in views:
        if v not in DIRECTION:
            raise ValueError(f"unknown view {v!r}; choose from {sorted(DIRECTION)}")
    if isinstance(drawings, np.ndarray):
        drawings = [drawings]
    if len(drawings) == 1:
        return auto_rig(drawings[0], views, params)
    if len(drawings) != len(views):
        raise ValueError(f"{len(drawings)} drawings for {len(views)} views: pass one drawing, or one per view")
    return [auto_rig(d, [v], params)[0] for d, v in zip(drawings, views)]


def build(drawing: np.ndarray | list[np.ndarray], views: list[str], mocap: bvh.Motion | None = None,
          params: BuildParams | None = None, animations: tuple[str, ...] = ("idle", "walk")) -> BuildResult:
    params = params or BuildParams()
    if "walk" in animations and mocap is None:
        raise ValueError("walk needs motion capture (BVH); see fetch_mocap.sh")
    rigs = rig_drawings(drawing, views, params.rig)
    fw, fh = params.frame_size
    scale = params.char_height / max(r.height for r in rigs)
    win_w, win_h = fw / scale, fh / scale
    canvas = (int(math.ceil(win_w)), int(math.ceil(win_h)))
    cycle = motion.extract_cycle(mocap, params.walk_frames) if "walk" in animations else None

    hires: dict[str, list[np.ndarray]] = {}
    clips: list[tuple[str, Clip]] = []
    for rig in rigs:
        base = render.translate(win_w / 2 - rig.root[0], (fh - params.foot_margin) / scale - rig.ground_y)
        for anim in animations:
            if anim == "walk":
                clip = walk_clip(rig, motion.project(cycle, rig.view, params.pitch), params)
            elif anim == "idle":
                clip = idle_clip(rig, params)
            else:
                raise ValueError(f"unknown animation {anim!r}")
            # snap whole-body motion to whole output pixels: sub-pixel bobbing makes
            # small features shimmer after downscaling
            for pose in clip.poses:
                pose.offset = tuple(round(v * scale) / scale for v in pose.offset)
                if params.snap_head:
                    pose.snap["head"] = 1.0 / scale
            key = f"{anim}_{DIRECTION[rig.view]}"
            hires[key] = [render.render(rig, p, canvas, o, base) for p, o in zip(clip.poses, clip.orders)]
            clips.append((key, clip))

    outline = params.outline
    if outline == "auto":
        outline = next((r.outline for r in rigs if r.outline), (27, 22, 34))
    reserve = [outline] if outline else None
    window = (0.0, 0.0, win_w, win_h)
    if params.downscale == "dominant":
        # the hi-res frames only contain source art colours (nearest sampling),
        # so the palette is built from the art itself, weighted by on-screen area
        palette = px.shared_palette([f for v in hires.values() for f in v[:: max(1, len(v) // 4)]],
                                    params.colors, reserve)
        small = {k: [px.downscale_dominant(f, window, (fw, fh), palette, params.alpha_threshold) for f in v]
                 for k, v in hires.items()}
    elif params.downscale == "box":
        small = {k: [px.downscale(f, window, (fw, fh), params.alpha_threshold) for f in v] for k, v in hires.items()}
        palette = px.shared_palette([f for v in small.values() for f in v], params.colors, reserve)
        small = {k: [px.apply_palette(f, palette) for f in v] for k, v in small.items()}
    else:
        raise ValueError(f"unknown downscale mode {params.downscale!r} (dominant or box)")
    if outline:
        small = {k: [px.add_outline(f, outline) for f in v] for k, v in small.items()}

    pivot = (fw / 2, fh - params.foot_margin)
    anims: list[Animation] = []
    for key, clip in clips:
        anims.append(Animation(key, small[key], clip.durations, pivot))
    if params.flip:
        for key, clip in clips:
            anim, d = key.rsplit("_", 1)
            if d in FLIP:
                anims.append(Animation(f"{anim}_{FLIP[d]}", [f[:, ::-1].copy() for f in small[key]], clip.durations, pivot))
    anims.sort(key=lambda a: (a.name.split("_")[0], COMPASS.index(a.name.rsplit("_", 1)[1])))

    report = {
        "views": views, "scale": round(scale, 4), "hires_canvas": list(canvas),
        "frame_size": [fw, fh], "pivot": list(pivot), "palette_size": int(len(palette)),
        "outline": list(outline) if outline else None,
        "rigs": [r.summary() for r in rigs],
        "animations": {a.name: {"frames": len(a.frames), "durations_ms": a.durations_ms} for a in anims},
    }
    if cycle is not None:
        report["mocap"] = {"cycle_source_frames": [round(float(t), 2) for t in cycle.source_frames],
                           "frame_ms": round(cycle.frame_ms, 2)}
    return BuildResult(anims, rigs, hires, palette, report)
