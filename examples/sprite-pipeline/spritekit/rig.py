"""Auto-rig an exploded-view drawing.

Input conventions (what the artist is asked for):
  * flat #FF00FF background, no shadows or text;
  * every body part separated by a small even gap;
  * a solid #00FF00 dot centred in the gap at every joint: neck, both shoulders,
    both elbows, both hips, both knees (shoulder dots may be omitted);
  * both arms and both legs drawn; the torso complete under the arms.

Output: one :class:`Rig` per figure with parts moved together (gaps closed with
a small overlap), pivots at the contact points, a synthesized neck, and a rest
pose with arms hanging and legs straightened and pulled together.

Bone sides: ``a`` is the limb drawn on the screen-left, ``b`` on the screen-right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import image as im
from . import render


@dataclass
class Bone:
    name: str
    parent: str | None
    pivot: np.ndarray
    axis: np.ndarray  # unit vector of the limb direction in assembled coordinates
    length: float
    end: np.ndarray  # far end (child joint or tip) in assembled coordinates
    rest: float = 0.0  # local rest rotation (radians, y-down screen convention)
    soft: float = 0.0
    parent_soft: bool = False


@dataclass
class Part:
    name: str
    bone: str
    image: np.ndarray  # RGBA crop, background transparent
    origin: np.ndarray  # top-left of the crop in assembled coordinates


@dataclass
class RigParams:
    overlap: float = 7.0  # px the child is pushed into its parent when closing gaps
    leg_overlap: float = 2.0  # thigh overlap after pulling legs in (use ~6 for front/back views)
    arm_hang_deg: float = 8.0
    leg_out_deg: float = 1.5
    neck_fraction: float = 0.08  # synthesized neck length as a fraction of head height
    min_part_area: int = 300
    min_dot_area: int = 12
    # soft joint radii as fractions of figure height
    soft: dict = field(default_factory=lambda: {
        "neck": 0.15, "upper_arm": 0.035, "forearm": 0.04, "thigh": 0.03, "shin": 0.04})


@dataclass
class Rig:
    view: str
    bones: dict[str, Bone]
    parts: list[Part]
    root: np.ndarray  # hip centre
    ground_y: float
    top_y: float
    outline: tuple[int, int, int] | None
    dots: dict[str, tuple[float, float]]
    warnings: list[str] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.ground_y - self.top_y

    def summary(self) -> dict:
        return {
            "view": self.view, "height_px": round(self.height, 1),
            "bones": {n: {"parent": b.parent, "pivot": [round(float(v), 1) for v in b.pivot],
                          "length": round(b.length, 1), "rest_deg": round(math.degrees(b.rest), 1)}
                      for n, b in self.bones.items()},
            "parts": {p.name: p.bone for p in self.parts},
            "dots": {k: [round(v[0], 1), round(v[1], 1)] for k, v in self.dots.items()},
            "warnings": self.warnings,
        }


class RigError(ValueError):
    pass


# ---------------------------------------------------------------------------
# helpers


@dataclass
class _Blob:
    label: int
    pts: np.ndarray  # (n, 2) pixel centres x, y in exploded coordinates
    centroid: np.ndarray
    box: tuple[int, int, int, int]


def _nearest(pts: np.ndarray, target) -> tuple[np.ndarray, float]:
    d = np.hypot(pts[:, 0] - target[0], pts[:, 1] - target[1])
    i = int(d.argmin())
    return pts[i], float(d[i])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 1.0])


def _closest_pair(a: np.ndarray, b: np.ndarray, sample: int = 600) -> tuple[np.ndarray, np.ndarray]:
    a = a[:: max(1, len(a) // sample)]
    b = b[:: max(1, len(b) // sample)]
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    i, j = np.unravel_index(d.argmin(), d.shape)
    return a[i], b[j]


def _split_figures(parts: list[_Blob], n: int) -> list[list[_Blob]]:
    """Group parts into ``n`` figures using the widest empty column runs."""
    if n == 1:
        return [parts]
    spans = sorted((p.box[0], p.box[2]) for p in parts)
    gaps = []
    reach = spans[0][1]
    for x0, x1 in spans[1:]:
        if x0 > reach:
            gaps.append((x0 - reach, (reach + x0) / 2))
        reach = max(reach, x1)
    if len(gaps) < n - 1:
        raise RigError(f"expected {n} figures side by side, found {len(gaps) + 1}")
    cuts = sorted(c for _, c in sorted(gaps, reverse=True)[: n - 1])
    groups: list[list[_Blob]] = [[] for _ in range(n)]
    for p in parts:
        groups[int(np.searchsorted(cuts, p.centroid[0]))].append(p)
    return groups


def _label_dots(dots: list[np.ndarray]) -> dict[str, np.ndarray]:
    ds = sorted(dots, key=lambda d: d[1])
    if len(ds) not in (7, 9):
        raise RigError(f"expected 9 joint dots (or 7 without shoulders), found {len(ds)}")
    out = {"neck": ds[0]}
    knees = sorted(ds[-2:], key=lambda d: d[0])
    hips = sorted(ds[-4:-2], key=lambda d: d[0])
    rest = ds[1:-4]
    if len(rest) == 4:
        shoulders = sorted(rest[:2], key=lambda d: d[0])
        elbows = sorted(rest[2:], key=lambda d: d[0])
        out["shoulder_a"], out["shoulder_b"] = shoulders
    else:
        elbows = sorted(rest, key=lambda d: d[0])
    out["elbow_a"], out["elbow_b"] = elbows
    out["hip_a"], out["hip_b"] = hips
    out["knee_a"], out["knee_b"] = knees
    return out


# ---------------------------------------------------------------------------
# main entry


def auto_rig(arr: np.ndarray, views: list[str], params: RigParams | None = None) -> list[Rig]:
    """Rig every figure in an exploded-view drawing (figures ordered left to right)."""
    params = params or RigParams()
    dot_mask = im.green_dot_mask(arr)
    grown = dot_mask.copy()
    grown[1:] |= dot_mask[:-1]
    grown[:-1] |= dot_mask[1:]
    grown[:, 1:] |= dot_mask[:, :-1]
    grown[:, :-1] |= dot_mask[:, 1:]
    fg = im.key_magenta(arr) & ~grown
    clean = arr.copy()
    clean[~fg] = 0

    labels, n = im.label_components(fg)
    parts = []
    for b in im.blobs(labels, n):
        if b.area < params.min_part_area:
            continue
        ys, xs = np.nonzero(labels[b.y0:b.y1, b.x0:b.x1] == b.label)
        pts = np.stack([xs + b.x0 + 0.5, ys + b.y0 + 0.5], axis=1)
        parts.append(_Blob(b.label, pts, np.array([b.cx + 0.5, b.cy + 0.5]), (b.x0, b.y0, b.x1, b.y1)))
    dl, dn = im.label_components(dot_mask)
    dots = [np.array([b.cx + 0.5, b.cy + 0.5]) for b in im.blobs(dl, dn) if b.area >= params.min_dot_area]

    rigs = []
    for view, group in zip(views, _split_figures(parts, len(views))):
        x0 = min(p.box[0] for p in group)
        x1 = max(p.box[2] for p in group)
        fig_dots = [d for d in dots if x0 - 20 <= d[0] <= x1 + 20]
        rigs.append(_rig_figure(view, clean, labels, group, fig_dots, params))
    return rigs


def _rig_figure(view: str, clean: np.ndarray, labels: np.ndarray, parts: list[_Blob],
                dots: list[np.ndarray], params: RigParams) -> Rig:
    warnings: list[str] = []
    jd = _label_dots(dots)
    unused = list(parts)

    def take_nearest(target, filt=None) -> _Blob:
        cands = [p for p in unused if filt is None or filt(p)]
        if not cands:
            raise RigError(f"no part left near {np.round(target, 1).tolist()}")
        best = min(cands, key=lambda p: _nearest(p.pts, target)[1])
        unused.remove(best)
        return best

    hip_c = (jd["hip_a"] + jd["hip_b"]) / 2
    torso = take_nearest((jd["neck"] + hip_c) / 2)
    head_cands = [p for p in unused if p.centroid[1] < jd["neck"][1]]
    merged_head = False
    if head_cands:
        head = min(head_cands, key=lambda p: _nearest(p.pts, jd["neck"])[1])
        unused.remove(head)
    else:
        # head drawn touching the torso: split the blob at the neck dot
        above = torso.pts[:, 1] < jd["neck"][1]
        head = _Blob(-1, torso.pts[above], torso.pts[above].mean(axis=0), torso.box)
        torso = _Blob(torso.label, torso.pts[~above], torso.pts[~above].mean(axis=0), torso.box)
        merged_head = True
        warnings.append("head and torso were one blob; split at the neck dot (no neck synthesized)")

    limb: dict[str, _Blob] = {}
    for s in "ab":
        if f"shoulder_{s}" in jd:
            limb[f"upper_arm_{s}"] = take_nearest((jd[f"shoulder_{s}"] + jd[f"elbow_{s}"]) / 2)
        else:
            elbow = jd[f"elbow_{s}"]
            reach = np.linalg.norm(elbow - torso.centroid)
            limb[f"upper_arm_{s}"] = take_nearest(elbow, lambda p: np.linalg.norm(p.centroid - torso.centroid) < reach)
            a, b = _closest_pair(limb[f"upper_arm_{s}"].pts, torso.pts)
            jd[f"shoulder_{s}"] = (a + b) / 2
            warnings.append(f"shoulder_{s} dot missing; inferred from the closest approach of arm and torso")
        sh, el = jd[f"shoulder_{s}"], jd[f"elbow_{s}"]
        limb[f"forearm_{s}"] = take_nearest(el + (el - sh) * 0.6)
        hp, kn = jd[f"hip_{s}"], jd[f"knee_{s}"]
        limb[f"thigh_{s}"] = take_nearest((hp + kn) / 2)
        limb[f"shin_{s}"] = take_nearest(kn + (kn - hp) * 0.6)

    # --- close gaps (exploded -> assembled). offset = translation applied to a blob.
    offsets: dict[str, np.ndarray] = {"torso": np.zeros(2)}
    blobs: dict[str, _Blob] = {"torso": torso, "head": head, **limb}
    pivots: dict[str, np.ndarray] = {}
    joint_of = {"upper_arm": "shoulder", "forearm": "elbow", "thigh": "hip", "shin": "knee"}
    parent_of = {"upper_arm": "torso", "forearm": "upper_arm", "thigh": "torso", "shin": "thigh"}

    def close(child: str, parent: str, dot: np.ndarray, overlap: float) -> None:
        qp, _ = _nearest(blobs[parent].pts, dot)
        qc, _ = _nearest(blobs[child].pts, dot)
        v = qp - qc
        direction = _unit(v) if np.linalg.norm(v) > 1e-6 else _unit(dot - blobs[child].centroid)
        offsets[child] = offsets[parent] + v + direction * overlap
        pivots[child] = qp + offsets[parent] + direction * overlap * 0.5

    for s in "ab":
        for kind in ("upper_arm", "forearm", "thigh", "shin"):
            par = "torso" if parent_of[kind] == "torso" else f"{parent_of[kind]}_{s}"
            close(f"{kind}_{s}", par, jd[f"{joint_of[kind]}_{s}"], params.overlap)

    head_h = head.pts[:, 1].max() - head.pts[:, 1].min()
    head_w = head.pts[:, 0].max() - head.pts[:, 0].min()
    neck_len = 0.0 if merged_head else max(4.0, params.neck_fraction * head_h)
    qp, _ = _nearest(torso.pts, jd["neck"])
    qc, _ = _nearest(head.pts, jd["neck"])
    up = _unit(qc - qp) if np.linalg.norm(qc - qp) > 1e-6 else np.array([0.0, -1.0])
    offsets["head"] = (qp + up * neck_len) - qc
    neck_base = qp.copy()
    chin = qp + up * neck_len

    # --- parts
    def crop(blob: _Blob, name: str, bone: str) -> Part:
        xs = np.floor(blob.pts[:, 0]).astype(int)
        ys = np.floor(blob.pts[:, 1]).astype(int)
        bx0, by0 = xs.min(), ys.min()
        img = np.zeros((ys.max() - by0 + 1, xs.max() - bx0 + 1, 4), np.uint8)
        img[ys - by0, xs - bx0] = clean[ys, xs]
        return Part(name, bone, img, np.array([bx0, by0], float) + offsets[name])

    parts = [crop(blobs[k], k, k) for k in blobs]
    outline = im.outline_color(clean, np.isin(labels, [b.label for b in blobs.values() if b.label > 0]))

    # synthesized neck strip
    if neck_len > 0:
        parts.append(_neck_part(clean, head, offsets["head"], neck_base, chin, head_w, params.overlap, outline))

    # --- props: any remaining part rides rigidly on the nearest body part
    body = list(blobs)
    hosts: dict[str, str] = {}
    for i, extra in enumerate(unused):
        host = min(body, key=lambda k: _closest_dist(extra.pts, blobs[k].pts))
        name = f"prop{i}"
        hosts[name] = host
        offsets[name] = offsets[host].copy()
        blobs[name] = extra
        parts.append(crop(extra, name, name))
        a, b = _closest_pair(extra.pts, blobs[host].pts)
        pivots[name] = (a + b) / 2 + offsets[host]
        warnings.append(f"extra part {name} ({len(extra.pts)} px) attached rigidly to {host}")

    # --- bones (assembled coordinates)
    def assembled_centroid(k: str) -> np.ndarray:
        return blobs[k].centroid + offsets[k]

    def far_end(k: str, pivot: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        axis = _unit(assembled_centroid(k) - pivot)
        proj = (blobs[k].pts + offsets[k] - pivot) @ axis
        length = float(proj.max())
        return axis, length, pivot + axis * length

    root = (pivots["thigh_a"] + pivots["thigh_b"]) / 2
    bones: dict[str, Bone] = {}
    bones["root"] = Bone("root", None, root.copy(), np.array([0.0, -1.0]), 0.0, root.copy())
    t_axis = _unit(neck_base - root)
    bones["torso"] = Bone("torso", "root", root.copy(), t_axis, float(np.linalg.norm(neck_base - root)), neck_base.copy())
    bones["neck"] = Bone("neck", "torso", neck_base.copy(), up, neck_len, chin.copy())
    h_axis, h_len, h_end = far_end("head", chin)
    bones["head"] = Bone("head", "neck", chin.copy(), h_axis, h_len, h_end)
    for s in "ab":
        ua, fa, th, sh = (f"{k}_{s}" for k in ("upper_arm", "forearm", "thigh", "shin"))
        e = pivots[fa]
        bones[ua] = Bone(ua, "torso", pivots[ua], _unit(e - pivots[ua]), float(np.linalg.norm(e - pivots[ua])), e.copy())
        ax, ln, en = far_end(fa, pivots[fa])
        bones[fa] = Bone(fa, ua, pivots[fa], ax, ln, en, parent_soft=True)
        k = pivots[sh]
        bones[th] = Bone(th, "root", pivots[th], _unit(k - pivots[th]), float(np.linalg.norm(k - pivots[th])), k.copy())
        ax, ln, en = far_end(sh, pivots[sh])
        bones[sh] = Bone(sh, th, pivots[sh], ax, ln, en, parent_soft=True)
    for name, host in hosts.items():
        bones[name] = Bone(name, host, pivots[name], np.array([0.0, 1.0]), 0.0, pivots[name].copy())

    # --- rest pose
    for s, sign in (("a", 1), ("b", -1)):
        arm_target = math.radians(90 + sign * params.arm_hang_deg)  # screen-left limb hangs down-left
        leg_target = math.radians(90 + sign * params.leg_out_deg)
        _set_world_angle(bones, f"upper_arm_{s}", arm_target)
        _set_world_angle(bones, f"forearm_{s}", arm_target)
        _set_world_angle(bones, f"thigh_{s}", leg_target)
        _set_world_angle(bones, f"shin_{s}", leg_target)

    rig = Rig(view, bones, parts, root, 0.0, 0.0, outline, {k: (float(v[0]), float(v[1])) for k, v in jd.items()}, warnings)
    leg_overlap = params.leg_overlap if "_" in view else max(params.leg_overlap, 6.0)
    _pull_legs_in(rig, leg_overlap)

    # extents in rest pose
    mats = render.bone_matrices(rig, render.Pose())
    ys = []
    for p in rig.parts:
        h, w = p.image.shape[:2]
        mask = p.image[..., 3] > 0
        yy, xx = np.nonzero(mask)
        pts = np.stack([xx + 0.5, yy + 0.5], 1) + p.origin
        ys.append(render.transform_points(mats[p.bone], pts)[:, 1])
    ys = np.concatenate(ys)
    rig.top_y, rig.ground_y = float(ys.min() - 0.5), float(ys.max() + 0.5)
    height = rig.height
    for kind, frac in params.soft.items():
        for name, b in bones.items():
            if name == kind or name.startswith(kind + "_"):
                b.soft = frac * height
    return rig


def _closest_dist(a: np.ndarray, b: np.ndarray) -> float:
    p, q = _closest_pair(a, b, sample=300)
    return float(np.linalg.norm(p - q))


def _world_rest_angle(bones: dict[str, Bone], name: str) -> float:
    """World angle of a bone's axis in the rest pose (sum of ancestor rests + drawn axis)."""
    b = bones[name]
    total = math.atan2(b.axis[1], b.axis[0])
    p = b.parent
    while p:
        total += bones[p].rest
        p = bones[p].parent
    return total + b.rest


def _set_world_angle(bones: dict[str, Bone], name: str, target: float) -> None:
    current = _world_rest_angle(bones, name)
    d = (target - current + math.pi) % (2 * math.pi) - math.pi
    bones[name].rest += d


def _descendants(bones: dict[str, Bone], name: str) -> list[str]:
    out = [name]
    for n, b in bones.items():
        if b.parent in out and n not in out:
            out.append(n)
    return out


def _pull_legs_in(rig: Rig, overlap: float) -> None:
    """Translate both legs horizontally until the thighs overlap by ``overlap`` px."""
    mats = render.bone_matrices(rig, render.Pose())
    edges = {}
    for s in "ab":
        part = next(p for p in rig.parts if p.bone == f"thigh_{s}")
        yy, xx = np.nonzero(part.image[..., 3] > 0)
        pts = render.transform_points(mats[f"thigh_{s}"], np.stack([xx + 0.5, yy + 0.5], 1) + part.origin)
        rows = np.round(pts[:, 1]).astype(int)
        edges[s] = (rows, pts[:, 0])
    ra, xa = edges["a"]
    rb, xb = edges["b"]
    common = np.intersect1d(ra, rb)
    if len(common) == 0:
        return
    gap = min(xb[rb == r].min() - xa[ra == r].max() for r in common)
    shift = max(0.0, (gap + overlap) / 2)
    if shift == 0.0:
        return
    for s, sign in (("a", 1), ("b", -1)):
        moved = _descendants(rig.bones, f"thigh_{s}")
        delta = np.array([sign * shift, 0.0])
        for n in moved:
            rig.bones[n].pivot = rig.bones[n].pivot + delta
            rig.bones[n].end = rig.bones[n].end + delta
        for p in rig.parts:
            if p.bone in moved:
                p.origin = p.origin + delta


def _neck_part(clean: np.ndarray, head: _Blob, head_offset: np.ndarray, base: np.ndarray, chin: np.ndarray,
               head_w: float, overlap: float, outline) -> Part:
    """Skin-coloured strip from inside the torso top to under the chin."""
    # sample skin: most common bright, non-outline colour in the chin band of the head
    ys = head.pts[:, 1]
    band = head.pts[(ys > ys.max() - 0.2 * (ys.max() - ys.min())) &
                    (np.abs(head.pts[:, 0] - head.centroid[0]) < 0.2 * head_w)]
    cols = clean[np.floor(band[:, 1]).astype(int), np.floor(band[:, 0]).astype(int), :3].astype(int) if len(band) else np.zeros((0, 3), int)
    luma = cols @ np.array([0.299, 0.587, 0.114]) if len(cols) else np.zeros(0)
    cols = cols[luma > 80]
    if len(cols):
        vals, counts = np.unique(cols, axis=0, return_counts=True)
        skin = vals[counts.argmax()]
    else:
        skin = np.array([230, 185, 150])
    edge = np.array(outline if outline else (27, 22, 34))
    width = max(4, int(round(0.3 * head_w)))
    top = chin[1] - overlap
    bottom = base[1] + overlap
    h = max(2, int(math.ceil(bottom - top)))
    img = np.zeros((h, width, 4), np.uint8)
    img[..., :3] = skin
    img[..., 3] = 255
    img[:, :2, :3] = edge
    img[:, -2:, :3] = edge
    shadow = max(1, int(round(overlap + 0.25 * (h - 2 * overlap))))
    img[:shadow, 2:-2, :3] = (skin * 0.75).astype(np.uint8)
    origin = np.array([(chin[0] + base[0]) / 2 - width / 2, top])
    return Part("neck", "neck", img, origin)
