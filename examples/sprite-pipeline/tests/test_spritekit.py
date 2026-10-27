"""Tests for spritekit. Run from examples/sprite-pipeline: python -m unittest discover -s tests"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spritekit import analyze, bvh, demo, motion, pipeline, render  # noqa: E402
from spritekit import prompts as pr  # noqa: E402
from spritekit import image as im  # noqa: E402
from spritekit import rig as rigmod  # noqa: E402
from spritekit import sheet as sh  # noqa: E402
from spritekit.__main__ import main  # noqa: E402


def _bfs_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, bool)
    n = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                n += 1
                stack = [(y, x)]
                seen[y, x] = True
                while stack:
                    a, b = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = a + dy, b + dx
                            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                                seen[yy, xx] = True
                                stack.append((yy, xx))
    return n


def _figure(cell: tuple[int, int], phase: float, colour=(200, 60, 60)) -> np.ndarray:
    """Asymmetric test sprite whose arm position depends on ``phase``."""
    w, h = cell
    img = Image.new("RGBA", cell, (0, 0, 0, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    cx = w // 2
    d.rectangle([cx - 5, h - 26, cx + 5, h - 8], fill=colour + (255,), outline=(20, 20, 30, 255))
    d.ellipse([cx - 6, h - 38, cx + 6, h - 26], fill=(240, 200, 160, 255), outline=(20, 20, 30, 255))
    d.rectangle([cx + 6, h - 24, cx + 9, h - 16], fill=(20, 20, 30, 255))  # asymmetric detail
    ax = int(round(6 * math.sin(phase)))
    d.line([cx - 6, h - 22, cx - 10 + ax, h - 12], fill=(20, 20, 30, 255), width=2)
    d.rectangle([cx - 4 + ax // 2, h - 8, cx - 1 + ax // 2, h - 2], fill=(20, 20, 30, 255))
    d.rectangle([cx + 1 - ax // 2, h - 8, cx + 4 - ax // 2, h - 2], fill=(20, 20, 30, 255))
    return np.array(img)


def _grid_sheet(cell=(48, 40), rows=(6, 4, 6), cols=8) -> np.ndarray:
    w, h = cell
    s = np.zeros((h * len(rows), w * cols, 4), np.uint8)
    for r, n in enumerate(rows):
        for c in range(n):
            fr = _figure(cell, 2 * math.pi * c / n)
            if r == 2:
                fr = _figure(cell, 2 * math.pi * c / n)[:, ::-1]
            s[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
    return s


class ImageTests(unittest.TestCase):
    def test_labelling_matches_bfs(self):
        rng = np.random.default_rng(1)
        for p in (0.3, 0.55, 0.7):
            m = rng.random((40, 50)) > p
            labels, n = im.label_components(m)
            self.assertEqual(n, _bfs_count(m))
            self.assertTrue((labels[m] > 0).all())
            self.assertTrue((labels[~m] == 0).all())

    def test_magenta_key_drops_spill_and_dots(self):
        a = np.zeros((4, 4, 4), np.uint8)
        a[..., 3] = 255
        a[0, 0, :3] = (255, 0, 255)
        a[0, 1, :3] = (200, 40, 210)  # spill
        a[0, 2, :3] = (0, 255, 0)
        a[0, 3, :3] = (120, 80, 60)
        k = im.key_magenta(a)
        self.assertEqual(k[0].tolist(), [False, False, True, True])
        self.assertEqual(im.green_dot_mask(a)[0].tolist(), [False, False, True, False])


class AnalyzeTests(unittest.TestCase):
    def test_grid_rows_mirrors_and_loops(self):
        s = _grid_sheet()
        an = analyze.analyze_sheet(s)
        self.assertEqual(an.layout, "grid")
        self.assertEqual(an.cell, (48, 40))
        self.assertEqual(len(an.frames), 16)
        rows = an.inferred["rows_as_animations"]
        self.assertEqual([len(r["frames"]) for r in rows], [6, 4, 6])
        self.assertEqual(rows[2].get("mirror_of_row"), 0)
        self.assertTrue(all(r["loops"] for r in rows))
        for r in rows:  # anchors measured per cell, feet do not slide much
            self.assertLess(r["anchor_drift_px"][1], 0.5)
        self.assertEqual(an.derived["outline_color"], "#14141e")
        self.assertIn("frame timing: a still image carries no durations", an.unknown)

    def test_duplicates_and_period(self):
        cell = (32, 40)
        frames = [_figure(cell, p) for p in (0, 1, 4, 0, 1, 4)]
        s = np.concatenate(frames, axis=1)
        an = analyze.analyze_sheet(s)
        self.assertEqual(an.cell, cell)
        self.assertEqual(an.derived["duplicates"], [[3, 0], [4, 1], [5, 2]])
        self.assertEqual(an.inferred["rows_as_animations"][0]["repeat_period"], 3)

    def test_opaque_background_and_cell_override(self):
        s = _grid_sheet()
        flat = s.copy()
        flat[s[..., 3] == 0] = (90, 160, 90, 255)
        an = analyze.analyze_sheet(flat)
        self.assertEqual(an.observed["background"], {"background": "color", "color": [90, 160, 90]})
        self.assertEqual(an.cell, (48, 40))
        forced = analyze.analyze_sheet(s, cell=(96, 40))
        self.assertEqual(forced.derived["grid_source"], "caller")
        self.assertEqual(len(forced.frames), 8)

    def test_packed_atlas_falls_back_to_islands(self):
        s = np.zeros((70, 101, 4), np.uint8)
        s[3:30, 2:20] = 255
        s[5:40, 30:41] = 255
        s[45:66, 7:90] = 255
        an = analyze.analyze_sheet(s)
        self.assertEqual(an.layout, "packed")
        self.assertEqual([(f.x, f.y, f.w, f.h) for f in an.frames], [(2, 3, 18, 27), (30, 5, 11, 35), (7, 45, 83, 21)])
        self.assertEqual([f.row for f in an.frames], [0, 0, 1])

    def test_one_shot_row_does_not_loop(self):
        cell = (40, 40)
        frames = []
        for i in range(6):  # a block sliding steadily right never returns to its start
            f = np.zeros((40, 40, 4), np.uint8)
            f[20:36, 2 + 4 * i:10 + 4 * i] = (90, 90, 200, 255)
            f[36:38, 0:40] = (20, 20, 30, 255)  # ground keeps the anchor fixed
            frames.append(f)
        an = analyze.analyze_sheet(np.concatenate(frames, axis=1), cell=cell)
        self.assertFalse(an.inferred["rows_as_animations"][0]["loops"])

    def test_pixel_scale(self):
        s = _grid_sheet()
        for k in (1, 3):
            big = np.repeat(np.repeat(s, k, axis=0), k, axis=1)
            self.assertEqual(analyze.detect_pixel_scale(big), k)

    def test_animation_durations_and_common_crop(self):
        frames, durs = [], [80, 120, 80, 200]
        for i in range(4):
            f = np.zeros((50, 60, 4), np.uint8)
            f[10 + 2 * i:20 + 2 * i, 5 + 5 * i:15 + 5 * i] = (255, 0, 0, 255)
            frames.append(f)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "anim.gif"
            imgs = [Image.fromarray(f, "RGBA") for f in frames]
            imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durs, loop=0, disposal=2)
            an, s = analyze.analyze_animation(path)
        self.assertEqual(an.observed["durations_ms"], durs)
        self.assertEqual([f.duration_ms for f in an.frames], durs)
        # common crop: the square keeps moving 5 px right / 2 px down per frame
        xs = [f.trim[0] for f in an.frames]
        ys = [f.trim[1] for f in an.frames]
        self.assertEqual(np.diff(xs).tolist(), [5, 5, 5])
        self.assertEqual(np.diff(ys).tolist(), [2, 2, 2])


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.motion = bvh.parse(demo.synthetic_walk_bvh())

    def test_parse_and_fk(self):
        m = self.motion
        self.assertIn("LeftFoot_End", m.names)
        pos = bvh.forward_kinematics(m)
        self.assertEqual(pos.shape, (240, len(m.joints), 3))
        hips, upleg = m.index("Hips"), m.index("LeftUpLeg")
        np.testing.assert_allclose(pos[0, upleg] - pos[0, hips], [2, -1, 0], atol=1e-9)
        np.testing.assert_allclose(pos[120, hips, 2], 12.0, atol=1e-6)  # travel along +Z

    def test_cycle_period_and_symmetry(self):
        c = motion.extract_cycle(self.motion, 12)
        self.assertAlmostEqual(c.frame_ms * 12, 1000.0, delta=15)
        self.assertLess(abs(c.positions["pelvis"][:, 2]).max(), 0.6)  # travel removed
        vm = motion.project(c, "front")
        a, b = vm.bones["thigh_a"].delta, vm.bones["thigh_b"].delta
        np.testing.assert_allclose(a, -np.roll(b, -6), atol=1e-6)  # mirrored half a cycle later

    def test_side_mapping_follows_screen(self):
        c = motion.extract_cycle(self.motion, 12)
        for view in ("front_left", "back_left"):
            vm = motion.project(c, view)
            self.assertGreater(len(vm.bones), 8)
        # in a front-left view the character's right side is screen-left and farther away
        vm = motion.project(c, "front_left")
        self.assertGreater(vm.bones["thigh_a"].depth.mean(), vm.bones["thigh_b"].depth.mean())


    def test_profile_view_maps_screen_left_to_far_side(self):
        c = motion.extract_cycle(self.motion, 12)
        for view in ("side_left", "front_left"):
            vm = motion.project(c, view)
            self.assertGreater(vm.bones["upper_arm_a"].depth.mean(), vm.bones["upper_arm_b"].depth.mean(), view)
            self.assertGreater(vm.bones["thigh_a"].depth.mean(), vm.bones["thigh_b"].depth.mean(), view)


class PromptTests(unittest.TestCase):
    def test_eight_direction_pack_covers_five_views(self):
        c = pr.Character("young knight, blue tunic", accent="red scarf", weapon="short sword", hairstyles=["bob", "spiky"])
        pack = pr.prompt_pack(c, 8)
        self.assertEqual(pack.views, ["front", "front_left", "side_left", "back_left", "back"])
        steps = [p.step for p in pack.prompts]
        self.assertEqual(steps, ["design", "turnaround"] + ["exploded"] * 5 + ["heads"])
        exploded = [p for p in pack.prompts if p.step == "exploded"]
        self.assertEqual([p.file for p in exploded], [f"view_{v}.png" for v in pack.views])
        for p, v in zip(exploded, pack.views):
            self.assertIn(pr.VIEWS[v].label, p.text)
            self.assertIn("#00FF00", p.text)
            self.assertIn("exactly 9 dots", p.text)
            self.assertIn("#FF00FF", p.text)
            self.assertIn("short sword", p.text)
            self.assertEqual(p.attach, ["design.png", "turnaround.png"])
        self.assertIn("LEFT of the near arm", exploded[2].text)  # side view convention used by the rig
        for v in pack.views:
            self.assertIn(pr.VIEWS[v].label, pack.prompts[1].text)  # turnaround lists every view
        self.assertIn("--views front,front_left,side_left,back_left,back", pack.build_command)
        self.assertIn("view_front.png view_front_left.png", pack.build_command)
        md = pr.to_markdown(pack)
        self.assertIn("Directions: 8", md)

    def test_four_direction_sheet_layout(self):
        pack = pr.prompt_pack(pr.Character("mage"), 4, "sheet")
        self.assertEqual(pack.views, ["front_left", "back_left"])
        self.assertEqual([p.file for p in pack.prompts], ["design.png", "turnaround.png", "views.png"])
        self.assertIn("left to right", pack.prompts[-1].text)
        self.assertIn("Directions: 4", pr.to_markdown(pack))

    def test_review_accepts_good_art_and_explains_bad_art(self):
        views = ["front", "front_left", "side_left", "back_left", "back"]
        art, truth = demo.exploded_views(tuple(views))
        self.assertTrue(pr.review(art, views).ok)
        bad = art.copy()
        x, y = truth[2]["knee_b"]
        bad[int(y) - 5:int(y) + 6, int(x) - 5:int(x) + 6] = (255, 0, 255, 255)
        result = pr.review(bad, views)
        self.assertFalse(result.ok)
        self.assertEqual([f.figure for f in result.findings], ["side_left"])
        self.assertIn("exactly 9 solid #00FF00 dots", result.fix_prompt)
        self.assertIn("In the side view", result.fix_prompt)
        wrong = pr.review(art, views[:4])
        self.assertIn("4 figure(s)", wrong.fix_prompt)


class RigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art, cls.truth = demo.exploded_views()
        cls.rigs = rigmod.auto_rig(cls.art, ["front_left", "back_left"])

    def test_dots_found_and_labelled(self):
        for r, truth in zip(self.rigs, self.truth):
            self.assertEqual(r.warnings, [])
            for k, (x, y) in truth.items():
                self.assertAlmostEqual(r.dots[k][0], x, delta=1.0, msg=k)
                self.assertAlmostEqual(r.dots[k][1], y, delta=1.0, msg=k)

    def test_rest_pose_is_one_connected_body(self):
        for r in self.rigs:
            order = pipeline.draw_order(r, "a", "a")
            base = render.translate(-(r.root[0] - 200), 0)
            img = render.render(r, render.Pose(), (400, 420), order, base)
            _, n = im.label_components(img[..., 3] > 0)
            self.assertEqual(n, 1, f"{r.view}: gaps not closed")

    def test_rest_pose_limbs(self):
        for r in self.rigs:
            for s, sign in (("a", 1), ("b", -1)):
                for bone, target in ((f"upper_arm_{s}", 90 + sign * 8), (f"thigh_{s}", 90 + sign * 1.5)):
                    ang = math.degrees(rigmod._world_rest_angle(r.bones, bone))
                    self.assertAlmostEqual(ang % 360, target, delta=0.01)

    def test_missing_shoulder_dots_are_inferred(self):
        art = self.art.copy()
        for side in ("a", "b"):
            x, y = self.truth[0][f"shoulder_{side}"]
            art[int(y) - 5:int(y) + 6, int(x) - 5:int(x) + 6] = (255, 0, 255, 255)
        r = rigmod.auto_rig(art[:, :300], ["front_left"])[0]
        self.assertTrue(any("shoulder_a dot missing" in w for w in r.warnings))
        self.assertAlmostEqual(r.dots["shoulder_a"][0], self.truth[0]["shoulder_a"][0], delta=4)

    def test_wrong_dot_count_is_reported(self):
        art = self.art[:, :300].copy()
        x, y = self.truth[0]["knee_a"]
        art[int(y) - 5:int(y) + 6, int(x) - 5:int(x) + 6] = (255, 0, 255, 255)
        with self.assertRaises(rigmod.RigError):
            rigmod.auto_rig(art, ["front_left"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        art, _ = demo.exploded_views()
        cls.params = pipeline.BuildParams()
        cls.res = pipeline.build(art, ["front_left", "back_left"], bvh.parse(demo.synthetic_walk_bvh()), cls.params)

    def test_animations_and_directions(self):
        names = [a.name for a in self.res.animations]
        self.assertEqual(sorted(names), sorted(f"{a}_{d}" for a in ("idle", "walk") for d in ("sw", "se", "nw", "ne")))
        walk = next(a for a in self.res.animations if a.name == "walk_sw")
        self.assertEqual(len(walk.frames), 12)
        self.assertAlmostEqual(sum(walk.durations_ms), 1000, delta=15)

    def test_pixel_frames(self):
        fw, fh = self.params.frame_size
        colours = set()
        for a in self.res.animations:
            for f in a.frames:
                self.assertEqual(f.shape, (fh, fw, 4))
                self.assertTrue(set(np.unique(f[..., 3])) <= {0, 255})
                rows = np.nonzero(f[..., 3].any(axis=1))[0]
                # a foot touches the baseline in every frame (ground pin, shared anchor)
                self.assertLessEqual(abs(rows.max() - (fh - self.params.foot_margin - 1)), 1, a.name)
                colours |= set(map(tuple, f[f[..., 3] > 0][:, :3].tolist()))
        self.assertLessEqual(len(colours), self.params.colors)

    def test_flipped_directions_mirror_exactly(self):
        by = {a.name: a for a in self.res.animations}
        for src, dst in (("walk_sw", "walk_se"), ("idle_nw", "idle_ne")):
            for a, b in zip(by[src].frames, by[dst].frames):
                np.testing.assert_array_equal(a[:, ::-1], b)

    def test_idle_face_is_pixel_stable(self):
        idle = next(a for a in self.res.animations if a.name == "idle_sw")
        heads, tops = [], []
        for f in idle.frames:
            rows = np.nonzero(f[..., 3].any(axis=1))[0]
            tops.append(int(rows.min()))
            heads.append(f[rows.min():rows.min() + 10])  # hair and eyes; the chin row touches the swaying arm
        self.assertEqual(max(tops) - min(tops), 1)  # breathing sinks the head by one whole pixel
        for h in heads[1:]:
            np.testing.assert_array_equal(h, heads[0])

    def test_round_trip_through_analyzer(self):
        sheet, meta = sh.pack(self.res.animations)
        an = analyze.analyze_sheet(sheet)
        self.assertEqual(an.cell, self.params.frame_size)
        self.assertEqual(len(an.frames), len(meta["frames"]))
        mirrored = [r for r in an.inferred["rows_as_animations"] if "mirror_of_row" in r]
        self.assertEqual(len(mirrored), 8)
        tags = {t["name"]: t for t in meta["meta"]["frameTags"]}
        self.assertEqual(tags["walk_sw"]["to"] - tags["walk_sw"]["from"] + 1, 12)


class EightDirectionTests(unittest.TestCase):
    def test_five_views_one_drawing_per_view(self):
        views = ["front", "front_left", "side_left", "back_left", "back"]
        art, _ = demo.exploded_views(tuple(views))
        per_view = [art[:, 300 * i:300 * (i + 1)] for i in range(len(views))]
        res = pipeline.build(per_view, views, bvh.parse(demo.synthetic_walk_bvh()), animations=("walk",))
        self.assertEqual([a.name for a in res.animations],
                         [f"walk_{d}" for d in ("s", "sw", "w", "nw", "n", "ne", "e", "se")])
        by = {a.name: a for a in res.animations}
        for a, b in zip(by["walk_w"].frames, by["walk_e"].frames):
            np.testing.assert_array_equal(a[:, ::-1], b)
        with self.assertRaises(ValueError):
            pipeline.build(per_view[:2], views, None, animations=("idle",))


class CliTests(unittest.TestCase):
    def test_demo_and_analyze_commands(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "demo"
            self.assertEqual(main(["demo", "--out", str(out)]), 0)
            for name in ("exploded.png", "sheet.png", "sheet.json", "report.json", "viewer.html", "rig_debug.png"):
                self.assertTrue((out / name).exists(), name)
            meta = json.loads((out / "sheet.json").read_text())
            self.assertIn(json.dumps(meta), (out / "viewer.html").read_text())
            an_out = Path(d) / "an"
            self.assertEqual(main(["analyze", str(out / "previews" / "walk_sw.gif"), "--out", str(an_out)]), 0)
            report = json.loads((an_out / "analysis.json").read_text())
            self.assertEqual(report["frame_count"], 12)
            self.assertEqual(report["derived"]["pixel_scale"], 4)
            self.assertEqual(main(["build", str(out / "exploded.png"), "--anims", "walk", "--out", str(Path(d) / "b")]), 2)
            self.assertEqual(main(["check", str(out / "exploded.png"), "--views", "front_left,back_left"]), 0)
            self.assertEqual(main(["check", str(out / "exploded.png"), "--views", "front_left"]), 1)
            pdir = Path(d) / "prompts"
            self.assertEqual(main(["prompts", "--character", "rogue", "--out", str(pdir)]), 0)
            pack = json.loads((pdir / "prompts.json").read_text())
            self.assertEqual(pack["views"], ["front", "front_left", "side_left", "back_left", "back"])
            self.assertTrue((pdir / "prompts.md").read_text().startswith("# Sprite art prompts: rogue"))


if __name__ == "__main__":
    unittest.main()
