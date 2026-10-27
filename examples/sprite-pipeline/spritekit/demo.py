"""Synthetic test inputs: an exploded-view chibi drawing and a procedural walk BVH.

The drawing follows the same conventions an artist (or an image model) is asked
for: flat #FF00FF background, every body part separated by a small gap, and a
#00FF00 dot centred in the gap at every joint. It exists so the pipeline and
its tests run without external art; real projects replace it with drawn views.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

OUTLINE = (27, 22, 34)
SKIN = (238, 190, 152)
HAIR = (122, 74, 42)
TUNIC = (64, 104, 196)
PANTS = (78, 66, 92)
BOOTS = (96, 62, 40)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)


def _shade(c: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, v * f))) for v in c)


def _capsule(draw: ImageDraw.ImageDraw, a, b, r: float, fill) -> None:
    draw.line([tuple(a), tuple(b)], fill=fill, width=int(round(2 * r)))
    for p in (a, b):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)


def _shaded_capsule(draw, a, b, r, colour, dark=1.0, outline=3):
    a, b = np.asarray(a, float), np.asarray(b, float)
    _capsule(draw, a, b, r, OUTLINE)
    _capsule(draw, a, b, r - outline, _shade(colour, 0.78 * dark))
    off = np.array([-0.3, -0.2]) * r
    _capsule(draw, a + off, b + off, max(r - outline - 3, 1), _shade(colour, dark))
    _capsule(draw, a + off * 1.8, b + off * 1.8, max(r * 0.25, 1), _shade(colour, 1.18 * dark))


def _figure(draw: ImageDraw.ImageDraw, cx: float, view: str, gap: float = 12.0) -> dict[str, tuple[float, float]]:
    """Draw one exploded figure; returns its joint dot positions."""
    back = view.startswith("back")
    side_view = view.startswith("side")
    g2 = gap / 2
    # torso (narrower in profile)
    t0, t1 = 132.0, 222.0
    tw = 28.0 if side_view else 40.0
    neck = (cx, t0 - g2)
    # head (circle) sits g/2 above the neck joint; the face turns with the view
    hr = 54.0
    face_shift = {"front": 0, "front_left": -4, "side_left": -8}.get(view, 0)
    hc = (cx + face_shift, neck[1] - g2 - hr)
    eyes = {"front": (-18, 18), "front_left": (-26, 10), "side_left": (-36,)}.get(view, ())
    # the far limbs are drawn slightly darker; which screen side is far depends on the view
    far = {"front_left": "a", "side_left": "a", "back_left": "b"}.get(view)
    toe = {"front_left": -1, "side_left": -1, "back_left": -1}.get(view, 0)
    draw.ellipse([hc[0] - hr, hc[1] - hr, hc[0] + hr, hc[1] + hr], fill=OUTLINE)
    if back:
        draw.ellipse([hc[0] - hr + 3, hc[1] - hr + 3, hc[0] + hr - 3, hc[1] + hr - 3], fill=_shade(HAIR, 0.8))
        draw.ellipse([hc[0] - hr + 10, hc[1] - hr + 6, hc[0] + hr - 18, hc[1] + hr - 22], fill=HAIR)
        draw.ellipse([hc[0] - 24, hc[1] - 44, hc[0] + 4, hc[1] - 24], fill=_shade(HAIR, 1.2))
    else:
        draw.ellipse([hc[0] - hr + 3, hc[1] - hr + 3, hc[0] + hr - 3, hc[1] + hr - 3], fill=_shade(SKIN, 0.85))
        draw.ellipse([hc[0] - hr + 8, hc[1] - hr + 14, hc[0] + hr - 16, hc[1] + hr - 6], fill=SKIN)
        draw.pieslice([hc[0] - hr + 3, hc[1] - hr + 3, hc[0] + hr - 3, hc[1] + hr - 3], 180, 360, fill=HAIR)
        draw.chord([hc[0] - hr + 3, hc[1] - hr + 3, hc[0] + hr - 3, hc[1] + 12], 180, 360, fill=_shade(HAIR, 0.8))
        for ex in (hc[0] + e for e in eyes):
            draw.ellipse([ex - 7, hc[1] + 6, ex + 7, hc[1] + 24], fill=OUTLINE)
            draw.ellipse([ex - 4, hc[1] + 9, ex + 1, hc[1] + 15], fill=(250, 250, 255))
    # torso rounded rect with belt
    draw.rounded_rectangle([cx - tw, t0, cx + tw, t1], 14, fill=OUTLINE)
    draw.rounded_rectangle([cx - tw + 3, t0 + 3, cx + tw - 3, t1 - 3], 12, fill=_shade(TUNIC, 0.8))
    draw.rounded_rectangle([cx - tw + 8, t0 + 6, cx + tw - 18, t1 - 22], 10, fill=TUNIC)
    draw.rectangle([cx - tw + 3, t1 - 28, cx + tw - 3, t1 - 18], fill=BOOTS)
    if not back and not side_view:
        bx = cx + face_shift
        draw.rectangle([bx - 7, t1 - 30, bx + 7, t1 - 16], fill=(230, 200, 80))

    joints: dict[str, tuple[float, float]] = {"neck": neck}
    # side "a" = screen left, "b" = screen right
    for side, sx in (("a", -1), ("b", 1)):
        dark = 0.82 if side == far else 1.0
        shoulder = (cx + sx * (tw + g2), t0 + 12)
        ang = math.radians(90 + sx * -42)  # A-pose: arms well away from the torso
        u = np.array([math.cos(ang), math.sin(ang)])
        ua, fa, ra = 46.0, 44.0, 12.0
        elbow = np.array(shoulder) + u * (ua + gap)
        hand = elbow + u * (fa + gap)
        _shaded_capsule(draw, np.array(shoulder) + u * (g2 + ra), elbow - u * (g2 + ra), ra, TUNIC, dark)
        _shaded_capsule(draw, elbow + u * (g2 + ra), hand - u * ra, ra - 1, SKIN, dark)
        joints[f"shoulder_{side}"] = shoulder
        joints[f"elbow_{side}"] = tuple(elbow)

        hip = (cx + sx * 20, t1 + g2)
        lu = np.array([math.cos(math.radians(90 - sx * 6)), math.sin(math.radians(90 - sx * 6))])
        tl, sl, lr = 46.0, 44.0, 15.0
        knee = np.array(hip) + lu * (tl + gap)
        ankle = knee + lu * (sl + gap)
        _shaded_capsule(draw, np.array(hip) + lu * (g2 + lr), knee - lu * (g2 + lr), lr, PANTS, dark)
        shin_end = ankle - lu * lr
        _shaded_capsule(draw, knee + lu * (g2 + lr), shin_end, lr - 1, PANTS, dark)
        boot_a = shin_end + np.array([0, -2])
        boot_b = shin_end + np.array([toe * 12, 6 if toe else 8])
        _shaded_capsule(draw, boot_a, boot_b, lr - 1, BOOTS, dark)
        joints[f"hip_{side}"] = hip
        joints[f"knee_{side}"] = tuple(knee)
    return joints


def exploded_views(views: tuple[str, ...] = ("front_left", "back_left"), dot_radius: int = 4) -> tuple[np.ndarray, list[dict]]:
    """RGBA drawing with one exploded figure per view, plus the true joint positions."""
    width = 300 * len(views)
    img = Image.new("RGBA", (width, 420), MAGENTA + (255,))
    draw = ImageDraw.Draw(img)
    truth = []
    for i, v in enumerate(views):
        joints = _figure(draw, 150 + 300 * i, v)
        for x, y in joints.values():
            draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=GREEN)
        truth.append(joints)
    return np.array(img), truth


def synthetic_walk_bvh(frames: int = 240, fps: float = 120.0, cycle_s: float = 1.0) -> str:
    """Procedural CMU-style walk BVH used only by tests and the offline demo fallback.

    Real projects should use captured motion (see fetch_mocap.sh); sine-driven
    walks look mechanical. This exists to exercise the parser and retargeting.
    """
    lines = ["HIERARCHY", "ROOT Hips", "{", "OFFSET 0 0 0",
             "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"]

    def joint(name, offset, children, depth=1):
        pad = "\t" * depth
        out = [f"{pad}JOINT {name}", f"{pad}{{", f"{pad}\tOFFSET {offset[0]} {offset[1]} {offset[2]}",
               f"{pad}\tCHANNELS 3 Zrotation Yrotation Xrotation"]
        if children:
            for c in children:
                out += joint(*c, depth=depth + 1)
        else:
            out += [f"{pad}\tEnd Site", f"{pad}\t{{", f"{pad}\t\tOFFSET 0 -1 0", f"{pad}\t}}"]
        out.append(f"{pad}}}")
        return out

    tree = []
    for side, sx in (("Left", 1), ("Right", -1)):
        tree.append((f"{side}UpLeg", (sx * 2, -1, 0), [(f"{side}Leg", (0, -7, 0), [(f"{side}Foot", (0, -7, 0), [])])]))
    tree.append(("Spine", (0, 2, 0), [("Spine1", (0, 3, 0), [
        ("Neck", (0, 3, 0), [("Head", (0, 2, 0), [])]),
        ("LeftArm", (3, 2, 0), [("LeftForeArm", (0, -5, 0), [("LeftHand", (0, -5, 0), [])])]),
        ("RightArm", (-3, 2, 0), [("RightForeArm", (0, -5, 0), [("RightHand", (0, -5, 0), [])])]),
    ])]))
    for t in tree:
        lines += joint(*t)
    lines.append("}")
    order = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
             "Spine", "Spine1", "Neck", "Head", "LeftArm", "LeftForeArm", "LeftHand",
             "RightArm", "RightForeArm", "RightHand"]
    lines += ["MOTION", f"Frames: {frames}", f"Frame Time: {1.0 / fps:.6f}"]
    speed = 12.0  # units per second of forward travel along +Z
    for f in range(frames):
        t = f / fps
        ph = 2 * math.pi * t / cycle_s
        vals = {n: [0.0, 0.0, 0.0] for n in order}
        root = [0.3 * math.sin(ph), 15 + 0.4 * math.cos(2 * ph), speed * t]
        for side, sgn in (("Left", 1), ("Right", -1)):
            swing = sgn * 25 * math.sin(ph)
            vals[f"{side}UpLeg"][2] = swing  # X rotation: forward/back
            vals[f"{side}Leg"][2] = 30 * max(0.0, math.sin(ph * sgn + 1.0))
            vals[f"{side}Arm"][2] = -sgn * 20 * math.sin(ph)
            vals[f"{side}ForeArm"][2] = -15
        row = root + [0.0, 0.0, 0.0]
        for n in order[1:]:
            row += vals[n]
        lines.append(" ".join(f"{v:.4f}" for v in row))
    return "\n".join(lines) + "\n"
