# Sprite pipeline: pictures to animated 2D sprite sheets

`spritekit` is a small Python toolkit (numpy + Pillow only). It works in both
directions:

- **Analyze** takes an existing sprite sheet, GIF or APNG and recovers its grid,
  frames, feet anchors, palette, outline, pixel scale, hold frames, mirrored
  directions, loops and real frame timings.
- **Build** takes a drawn character and turns it into a rigged, motion-capture
  animated pixel-art sprite sheet with Aseprite-compatible JSON, GIF previews
  and an HTML viewer.
- **Prompts** writes the prompts an image model (Codex) needs to draw that
  character: the base model, a turnaround, and an exploded rig view for each of
  front, 45° front, side, 45° back and back. It also checks the returned art
  and writes corrective prompts.

[`SKILL.md`](SKILL.md) is the same workflow written as instructions for an agent.

The rest of this page explains what the process does and why each step is
needed for an accurate result.

This is a standalone example. It is not part of REA's runtime, CLI or MCP tools.

## Quick start

```bash
cd examples/sprite-pipeline
python3 -m pip install numpy pillow        # if not already installed
./fetch_mocap.sh mocap                     # CMU walk 07_01 (free for any use)

# 1. End-to-end demo with a synthetic character (4 or 8 directions)
python3 -m spritekit demo --bvh mocap/07_01.bvh --directions 8 --out demo-out
#    demo-out/sheet.png, sheet.json, viewer.html, previews/*.gif, rig_debug.png, report.json

# 2. Your own character: prompts for the image model -> check the art -> build
python3 -m spritekit prompts --character "young knight, blue tunic, short brown hair" \
    --accent "red scarf" --weapon "short sword" --out art
#    ...send each prompt in art/prompts.md to Codex, save the images it returns...
python3 -m spritekit check view_front.png view_front_left.png view_side_left.png view_back_left.png view_back.png \
    --views front,front_left,side_left,back_left,back
python3 -m spritekit build view_front.png view_front_left.png view_side_left.png view_back_left.png view_back.png \
    --views front,front_left,side_left,back_left,back --bvh mocap/07_01.bvh --out out

# 3. Reverse-engineer someone's sheet or animated GIF
python3 -m spritekit analyze hero_walk.png --out analysis
python3 -m spritekit analyze hero_walk.gif --out analysis   # keeps the real durations

# tests (about 10 s)
python3 -m unittest discover -s tests
```

Open `viewer.html` from the output folder. It has animation chips,
play/step, speed, zoom, onion skin and a pivot overlay.

## Part 1: what a sprite sheet actually encodes

Analyzing many sheets shows they all reduce to the same few facts. The
analyzer recovers each one and labels it as **observed** (read from pixels or
file metadata), **derived** (computed exactly from observations), **inferred**
(a heuristic) or **unknown** (not present in the input).

| Fact | How it is recovered | Kind |
| --- | --- | --- |
| Background | Real alpha if present, otherwise the most common border colour | observed |
| Sprites | 8-connected islands, merged when within `--gap` px (detached hands, sparks) | derived |
| Grid / cell size | The smallest divisor of the sheet size whose cell lines cut no sprite, with no empty cells inside a row. Rows may end early. Falls back to a packed atlas when no grid fits | derived |
| Feet anchor (pivot) | Mean x of the two lowest opaque rows; y = bottom edge | derived |
| Pixel scale | Every colour change in upscaled art falls on a multiple of the scale on both axes | derived |
| Hold frames | Identical cells (exact hashes) | derived |
| Mirrored directions | A trimmed frame equals the horizontal flip of another frame. A whole row of these is a flipped direction | derived |
| Outline colour | Dominant dark colour on the silhouette edge | derived |
| Row = animation/direction | Convention | inferred |
| Loops | The wrap-around step (last to first frame) is no larger than the largest step inside the row | inferred |
| Anchor drift | Spread of the feet anchor. In an idle row, more than 1 px means the sprite slides | inferred |
| Timing | A PNG has none. GIF/APNG durations are read exactly | unknown / observed |

What this reveals about well-made sheets, and what the builder therefore does:

1. **Timing lives outside the image.** It goes in JSON, not in pixel spacing.
2. **All frames share one fixed anchor.** Frames are never trimmed and
   re-centred one at a time; doing that is the classic cause of jitter. When
   frames come from a GIF, `frames_to_sheet` crops every frame to their *common*
   bounding box.
3. **Right-facing directions are usually mirrored left-facing ones.**
4. **One palette is shared by every frame.** Per-frame palettes make colours
   flicker.
5. **Each sheet uses one integer pixel scale and one outline colour.**

## Part 2: turning a picture into motion

### Why one flat picture is not enough

When an arm swings, the pixels behind it (the torso) must exist. A single
painted pose doesn't contain them. Every cutout animation system (Spine,
DragonBones, Live2D, Flash puppets) therefore starts from **layered parts**.
The cheapest way to get parts from an artist or an image model is an
**exploded view**:

- flat `#FF00FF` background, no shadows, no text;
- every part (head, torso with hips, both upper arms, both forearms with
  hands, both thighs, both shins with feet) pulled apart by a small even gap;
- a solid `#00FF00` dot centred in the gap at **every joint**: neck, both
  shoulders, both elbows, both hips, both knees. Shoulder dots may be omitted;
  the rig then infers them;
- both arms and both legs drawn, with the far ones slightly darker; the torso
  drawn complete under the arms; rounded limb ends;
- hair, skin, eyes and each cloth piece in **one hue with 3–4 shades**.

Prompt templates for an image model are at the end of this page.

### Auto-rig (`rig.py`)

1. **Segment.** Key out magenta and its spill (`min(r,b) - g > 90`). Find the
   green dots and grow them by 1 px. Parts are the remaining components of 300
   px or more. Figures standing side by side are split at the widest empty
   column runs.
2. **Label the dots** by height. The top dot is the neck, the bottom two are
   the knees, the next two are the hips, and the rest are shoulders and elbows.
   Sides are named by screen position: `a` = screen-left, `b` = screen-right.
3. **Assign parts.** The torso is the part nearest the midpoint of neck and
   hips. The head is the nearest part above the neck. Each limb segment takes
   the nearest unused part to a sample point along its joints. Leftover parts
   (sword, cape, accessories) ride rigidly on the nearest body part.
4. **Close the gaps.** Each child moves, with its descendants, until its pixel
   nearest the dot meets the parent's nearest pixel, plus 7 px of overlap. The
   pivot sits in that overlap.
5. **Synthesize a neck.** The head stops short of the torso, and the gap is
   filled with a skin-coloured strip that has dark edges and a chin shadow.
   Without it the head looks bolted on.
6. **Rest pose.** Arms hang down, 8° outward, with forearms in line. Legs point
   straight down, 1.5° outward, with shins in line. Then both legs are pulled
   in until the thighs overlap (2 px in ¾ views, 6 px front/back). Skipping
   this gives bow legs.
7. **Soft joints.** Child pixels near a joint get only part of the animated
   rotation: 0.5 at the joint rising to 1 at the radius. Near elbows and knees,
   the parent's pixels follow by 0.5 falling to 0. Radii scale with figure
   height.

Check `rig_debug.png`: pivots are green, bone ends orange.

### Motion (`bvh.py`, `motion.py`)

Hand-keyed sine curves look like a puppet or a zombie. The builder retargets
**real motion capture**:

1. Parse the BVH. Local rotation is the product of the channels in file order
   (`Z Y X` gives `Rz·Ry·Rx`). Then run forward kinematics.
2. Turn the pelvis to face +Z and remove forward travel with a linear fit,
   keeping sway and bob.
3. **One gait cycle** = two consecutive upward zero crossings of
   (left ankle z − right ankle z), taken from the middle of the take. Remove
   the end drift and resample to 12 frames. Durations keep the real cycle
   length (about 91 ms per frame for CMU 07_01).
4. **Symmetrize.** Average each joint with its mirrored partner half a cycle
   later. This removes the limp present in real captures.
5. **Camera.** Orthographic, pitched 15° down. Yaw is 0 for front, −45 for
   front-left, −90 for side, −135 for back-left and 180 for back. Mocap sides map to rig
   sides by which one projects further left.
6. **Delta retargeting.** Bone angle = projected angle minus its cycle mean,
   added to the drawing's own rest pose. This keeps chibi proportions and
   stance and avoids crossed feet. Amplitudes: legs 0.85, arms 1.0, spine 0.7,
   neck 0.6. Forearms keep half the mean elbow bend. Foreshortening =
   projected length / 3D length / cos(pitch), clamped to 0.3–1.05.
7. **Ground pin.** Each frame, the body moves so the lower foot touches the
   ground. A farther foot touches higher on screen (depth · sin(pitch) · 0.6).
   The offset is smoothed 1-2-1 and capped at 3% of figure height. This
   prevents floating legs.
8. **Draw order.** Far arm, far leg (shin then thigh), near leg, neck, torso,
   head, near arm. The leg order is decided per frame from mocap depth.

The idle animation is keyed breathing: chest scale, a 1 px sink, and arms
lagging behind.

### Pixel reduction (`pixel.py`)

| Step | Why |
| --- | --- |
| Render hi-res by inverse mapping every output pixel (nearest sampling) | No holes when parts rotate or stretch. Output contains only source art colours |
| One fixed source window per view. The feet anchor sits on a pixel boundary | Every frame and view shares the anchor `(w/2, h − 5)` |
| Same scale for all views (character about 44 px tall in 64 px frames) | Directions match in size |
| Alpha = area coverage ≥ 110/255, then hard 0/255 | No semi-transparent fringe |
| Colour = **dominant palette colour per block**; dark "ink" wins at ≥ 30% coverage | Averaging turns eyes into grey mush; this keeps pupils and lines |
| One median-cut palette over every animation and direction (36 colours) | No colour flicker between frames |
| 1 px silhouette outline in the art's own outline colour | Crisp read on any background |
| Whole-body offset and head snapped to whole output pixels; the head stays rigid | Sub-pixel bobbing makes faces shimmer |
| Left-facing views mirrored to make the right-facing ones | Exact and free. One-sided details swap hands |

## Accuracy checklist and fixes

| Symptom | Cause | Fix in this pipeline |
| --- | --- | --- |
| Sprite jitters in place | Frames trimmed and centred one at a time | Common anchor / common crop |
| Colours flicker | Per-frame palettes | Shared palette |
| Grey eyes, mushy lines | Box averaging | Dominant-colour downscale with ink priority |
| Face shimmers in idle | Sub-pixel motion | Pixel snapping, rigid head |
| Limp | Asymmetric capture | Half-cycle mirror average |
| Crossed feet, odd stance | Absolute retargeting | Delta retargeting |
| Bow legs, wide stance | Drawn A-pose legs | Straighten legs and pull them in |
| Floating or heavy steps | No ground contact | Ground pin with depth term |
| Head bolted on | No neck | Synthesized neck strip |
| Walk timing feels off | Guessed fps | Durations from the real gait cycle |

## Outputs

- `sheet.png`: one row per animation and direction (`idle_s`, `idle_sw`, …,
  `walk_se`), in compass order s, sw, w, nw, n, ne, e, se, with 64×64 frames.
- `sheet.json`: Aseprite hash-style JSON with `frames[].duration`,
  `frames[].pivot` (normalized feet anchor) and `meta.frameTags` (one tag per
  row).
- `previews/*.gif`, `viewer.html`, `rig_debug.png`, and `report.json`
  (scale, palette, rig bones, mocap cycle).
- `analyze --out` writes `analysis.json`, Aseprite JSON for the input sheet,
  per-frame PNGs, per-row GIFs and a viewer.

## Prompts for the image model (`prompts.py`)

The image model (Codex) draws; this toolkit rigs and animates. `spritekit
prompts` writes every prompt the art needs, in order, with the file name to
save each result under and the images to attach:

| # | Prompt | Save as | Attach | Purpose |
| --- | --- | --- | --- | --- |
| 1 | Base model: character design | `design.png` | none | One full-body design in a 45° front-left A-pose. Fixes palette, proportions and outline |
| 2 | Base model in every view | `turnaround.png` | design | Turnaround of the same character in each view, standing on one baseline. Keeps the views consistent |
| 3+ | Exploded rig view, one per view | `view_<view>.png` | design, turnaround | The rig input: parts apart, 9 joint dots |
| last | Hairstyle head sheet (with `--hair`) | `heads.png` | design | Hairstyle swaps |

Views per direction count (`--directions`):

| Directions | Views drawn | Mirrored to make |
| --- | --- | --- |
| 8 (default) | front, 45° front-left, side (profile, facing left), 45° back-left, back | 45° front-right, side right, 45° back-right |
| 4 | 45° front-left, 45° back-left | 45° front-right, 45° back-right |

Every prompt repeats the rules the rig depends on: flat `#FF00FF` background;
no pure green or magenta on the character; one hue with 3–4 shades per
material; 1 px outline; no text or shadows. Every exploded prompt also asks for
the 9 `#00FF00` dots, rounded limb ends and a torso drawn complete under the
arms. Some instructions differ by view:

- **front / back**: left and right limbs at the same brightness;
- **45° views**: far arm and leg drawn too, slightly darker (the rig uses
  mocap depth for draw order, so either screen side can be far);
- **side**: far arm and leg shifted slightly to the LEFT of the near ones so
  nothing overlaps. The rig treats the screen-left limb as the far one in
  profile views.

`--layout separate` (default) asks for one image per view. That is more
reliable than `--layout sheet`, which puts all views in one image, left to
right. Other options: `--name`, `--style`, `--game`, `--heads`, `--accent`,
`--loose-part` (a hanging piece such as a scarf tail, drawn as its own part),
`--weapon`, `--hair` and `--sprite-size` (default 64). `--sprite-size` tells the model
the final frame size so it designs for readability, and is passed to the
suggested build command. Without `--out` the prompts are printed as Markdown; with `--out` you
also get `prompts.json`.

### Checking what comes back

`spritekit check` runs the rig's own segmentation on each returned image and
reports anything that would break rigging. It checks for: a background that
isn't flat magenta, the wrong number of figures, joint dots that aren't exactly
9, parts touching or merged (fewer than 10 separate pieces), unrigable limb
placement, and figures that are too small. For each problem it prints a
**corrective prompt** to send back to the model with the drawing attached. It
exits with 0 when everything is ready to build and 1 otherwise.

## Limits

- The synthetic demo character and the synthetic walk BVH exist for tests and
  offline runs. The demo falls back to the synthetic walk, with a warning, when
  `--bvh` is missing. Real art and captured motion give the intended quality.
- Only the walk is mocap-driven. Idle is keyed. Attack (mocap body plus keyed
  sword arm and smear frames), region masks for shader recolouring, hair and
  cape spring chains, and heel-to-toe foot roll are not built.
- Props are attached rigidly to the nearest body part. A sword keeps its drawn
  angle relative to the forearm.
- The analyzer's grid inference assumes cells tile the sheet exactly. Sheets
  with margins or spacing between cells fall back to the packed layout (one
  frame per sprite), or take `--cell`.
