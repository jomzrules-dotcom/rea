"""Command line: ``python -m spritekit {analyze,build,demo} ...``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import analyze as an
from . import bvh, demo, pipeline, render
from . import image as im
from . import sheet as sh

VIEWER = Path(__file__).with_name("viewer.html")


def _size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h or w)


def _write_viewer(out_dir: Path, meta: dict) -> Path:
    html = VIEWER.read_text().replace("/*META*/null", json.dumps(meta))
    path = out_dir / "viewer.html"
    path.write_text(html)
    return path


def cmd_analyze(args: argparse.Namespace) -> int:
    src = Path(args.input)
    with_timing = False
    try:
        from PIL import Image
        with Image.open(src) as probe:
            with_timing = getattr(probe, "n_frames", 1) > 1
    except OSError as e:
        print(f"cannot open {src}: {e}", file=sys.stderr)
        return 2
    if with_timing:
        result, sheet = an.analyze_animation(src)
    else:
        sheet = im.load_rgba(src)
        result = an.analyze_sheet(sheet, cell=_size(args.cell) if args.cell else None, gap=args.gap)
    print(an.describe(result))
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(json.dumps(result.to_json(), indent=1))
        sheet_name = src.name if not with_timing else src.stem + "_sheet.png"
        im.save_rgba(sheet, out / sheet_name)
        meta = an.to_aseprite(result, sheet_name)
        (out / (Path(sheet_name).stem + ".json")).write_text(json.dumps(meta, indent=1))
        frames_dir = out / "frames"
        frames_dir.mkdir(exist_ok=True)
        rows: dict[int, list] = {}
        for f in result.frames:
            cell = sheet[f.y:f.y + f.h, f.x:f.x + f.w]
            im.save_rgba(cell, frames_dir / f"r{f.row:02d}_c{f.col:02d}.png")
            rows.setdefault(f.row, []).append((cell, f.duration_ms or result.inferred.get("suggested_timing_ms", 100)))
        for r, items in rows.items():
            sh.write_gif([c for c, _ in items], [d for _, d in items], out / f"row{r:02d}.gif")
        _write_viewer(out, meta)
        print(f"wrote {out}/ (analysis.json, {Path(sheet_name).stem}.json, frames/, row GIFs, viewer.html)")
    return 0


def _debug_rig_image(rigs, path: Path) -> None:
    """Rest pose of every view with bone pivots marked (green) and ends (orange)."""
    tiles = []
    for rig in rigs:
        order = pipeline.draw_order(rig, "a", "a")
        x0 = rig.root[0] - rig.height * 0.7
        y0 = rig.top_y - 10
        w, h = int(rig.height * 1.4), int(rig.height + 20)
        base = render.translate(-x0, -y0)
        img = render.render(rig, render.Pose(), (w, h), order, base)
        mats = render.bone_matrices(rig, render.Pose(), base)
        for name, b in rig.bones.items():
            for pt, col in ((b.pivot, (0, 200, 0, 255)), (b.end, (255, 140, 0, 255))):
                m = mats[b.parent] if (pt is b.pivot and b.parent) else mats[name]
                x, y = render.transform_points(m, pt[None, :])[0]
                xi, yi = int(x), int(y)
                img[max(yi - 2, 0):yi + 3, max(xi - 2, 0):xi + 3] = col
        tiles.append(img)
    hh = max(t.shape[0] for t in tiles)
    out = np.concatenate([np.pad(t, ((0, hh - t.shape[0]), (0, 0), (0, 0))) for t in tiles], axis=1)
    im.save_rgba(out, path)


def run_build(drawing: np.ndarray, views: list[str], motion_path: str | None, out: Path,
              args: argparse.Namespace, synthetic: bool = False) -> int:
    anims = tuple(a.strip() for a in args.anims.split(",") if a.strip())
    mocap = None
    if "walk" in anims:
        if motion_path:
            mocap = bvh.load(motion_path)
        elif synthetic:
            print("warning: no --bvh given; using a synthetic walk (looks mechanical). "
                  "Run fetch_mocap.sh for real motion capture.", file=sys.stderr)
            mocap = bvh.parse(demo.synthetic_walk_bvh())
        else:
            print("walk needs --bvh (run fetch_mocap.sh), or pass --anims idle", file=sys.stderr)
            return 2
    params = pipeline.BuildParams(
        frame_size=_size(args.size), char_height=args.height, colors=args.colors,
        walk_frames=args.frames, outline=None if args.no_outline else "auto")
    res = pipeline.build(drawing, views, mocap, params, anims)
    out.mkdir(parents=True, exist_ok=True)
    meta = sh.write_sheet(res.animations, out / "sheet.png", out / "sheet.json")
    previews = out / "previews"
    previews.mkdir(exist_ok=True)
    for a in res.animations:
        sh.write_gif(a.frames, a.durations_ms, previews / f"{a.name}.gif")
    if args.hires:
        for k, frames in res.hires.items():
            im.save_rgba(np.concatenate(frames, axis=1), previews / f"{k}_hires.png")
    _debug_rig_image(res.rigs, out / "rig_debug.png")
    (out / "report.json").write_text(json.dumps(res.report, indent=1, default=float))
    _write_viewer(out, meta)
    for r in res.rigs:
        for w in r.warnings:
            print(f"[{r.view}] {w}", file=sys.stderr)
    print(f"{len(res.animations)} animations, {sum(len(a.frames) for a in res.animations)} frames, "
          f"{res.report['palette_size']} colours -> {out}/sheet.png (+ sheet.json, viewer.html, previews/)")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    drawing = im.load_rgba(args.drawing)
    return run_build(drawing, [v.strip() for v in args.views.split(",")], args.bvh, Path(args.out), args)


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    views = ["front_left", "back_left"]
    drawing, _ = demo.exploded_views(tuple(views))
    im.save_rgba(drawing, out / "exploded.png")
    return run_build(drawing, views, args.bvh, out, args, synthetic=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="spritekit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="reverse-engineer a sprite sheet, GIF or APNG")
    a.add_argument("input")
    a.add_argument("--cell", help="force the cell size, e.g. 64x64")
    a.add_argument("--gap", type=int, default=1, help="merge islands closer than this (px)")
    a.add_argument("--out", help="write analysis.json, Aseprite JSON, frames, GIFs and a viewer here")
    a.set_defaults(func=cmd_analyze)

    def build_opts(q):
        q.add_argument("--bvh", help="motion capture walk (BVH); see fetch_mocap.sh")
        q.add_argument("--anims", default="idle,walk")
        q.add_argument("--size", default="64x64", help="output frame size")
        q.add_argument("--height", type=int, default=44, help="character height in output pixels")
        q.add_argument("--colors", type=int, default=36, help="shared palette size")
        q.add_argument("--frames", type=int, default=12, help="walk frames per cycle (even)")
        q.add_argument("--no-outline", action="store_true")
        q.add_argument("--hires", action="store_true", help="also write hi-res strips")

    b = sub.add_parser("build", help="exploded-view drawing -> animated sprite sheet")
    b.add_argument("drawing")
    b.add_argument("--views", default="front_left,back_left",
                   help="figure views left to right: front, front_left, side_left, back_left, back")
    b.add_argument("--out", default="out")
    build_opts(b)
    b.set_defaults(func=cmd_build)

    d = sub.add_parser("demo", help="draw a synthetic character and build its sheet")
    d.add_argument("--out", default="demo-out")
    build_opts(d)
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
