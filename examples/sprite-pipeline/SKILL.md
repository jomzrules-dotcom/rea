---
name: sprite-art-prompts
description: Write image-model (Codex) prompts, run them through codex exec, for a 2D game character's base model and its front, 45° front, side, 45° back and back exploded rig views; check the returned art; send corrective prompts; then build an animated sprite sheet with spritekit.
---

# Sprite art prompts

The image model draws, spritekit rigs and animates. Don't hand-write prompts:
generate them so they match what the rig expects. Run every command from
`examples/sprite-pipeline`.

## 1. Agree on the character first

Before generating anything, get from the user:

- a one-line description (role, clothing, hair);
- a main accent piece (scarf, cape, sash) and a weapon, if any;
- the direction count: **8** (front, 45° front, side, 45° back, back) or
  **4** (the two 45° views only);
- hairstyles, if they want a head sheet for character creation.

Show the plan and wait for approval before generating art.

## 2. Generate the prompts

```bash
python3 -m spritekit prompts --character "<description>" --directions 8 \
    [--accent "red scarf"] [--weapon "short sword"] [--hair "bob,spiky,long"] --out art
```

`art/prompts.md` lists the prompts in order:

1. **Base model** (`design.png`): full-body design, 45° front-left A-pose.
2. **Turnaround** (`turnaround.png`, attach the design): the same character in
   every view on one baseline.
3. **Exploded rig views**, one per view (`view_front.png`,
   `view_front_left.png`, `view_side_left.png`, `view_back_left.png`,
   `view_back.png`). Attach the design and turnaround to each.
4. **Head sheet** (`heads.png`), only if hairstyles were given.

Ask Codex directly rather than pasting prompts by hand:

```bash
python3 -m spritekit codex art --dry-run   # show the codex exec jobs first
python3 -m spritekit codex art             # design -> turnaround -> views, 3 at a time, auto re-ask
```

This needs the Codex CLI installed and signed in (`npm i -g @openai/codex`,
`codex login`) and network access to `api.openai.com`. On Windows,
`--emit-ps1` writes `run_codex.ps1` for PowerShell instead. Without Codex,
send each prompt from `prompts.md` by hand, attach the listed images, and
save each result under the listed file name. Keep `--layout separate` (one
image per view) unless the user asks for one combined image.

## 3. Check every returned image before building

`spritekit codex` already runs this check on every exploded view and re-asks
Codex up to `--retries` times. Run it yourself for images made by hand.

```bash
python3 -m spritekit check view_front.png view_front_left.png view_side_left.png view_back_left.png view_back.png \
    --views front,front_left,side_left,back_left,back
```

- Exit code 0: all views are ready.
- Exit code 1: send the printed corrective prompt back to Codex with that drawing
  attached, save the new image over the old one, and check again. Fix one view
  at a time and keep the views that already passed.
- Also look at each image yourself. The checker can't see style drift: a
  different palette, missing details, or a scale that differs from the
  turnaround. Ask for a redraw against the turnaround when you see it.

## 4. Build and show the result

```bash
./fetch_mocap.sh mocap        # once; CMU walk 07_01
python3 -m spritekit build view_front.png view_front_left.png view_side_left.png view_back_left.png view_back.png \
    --views front,front_left,side_left,back_left,back --bvh mocap/07_01.bvh --out out
```

Check `out/rig_debug.png` (pivots in green, bone ends in orange) and
`out/previews/*.gif`, and open `out/viewer.html`. Report any rig warnings the
build prints.

## Rules the prompts already enforce (keep them in any hand edits)

- Solid flat `#FF00FF` background. No shadows, text, logos or gradients.
- Never pure green `#00FF00` or magenta on the character itself.
- Exploded view: every part separated by a small even gap. Parts are head
  (hair attached), torso with hips, both upper arms, both forearms with hands,
  both thighs, both shins with feet; weapon and accent tail as separate parts.
- Exactly 9 solid `#00FF00` dots, one centred in each joint gap: neck, both
  shoulders, both elbows, both hips, both knees.
- Front and back views: left and right limbs at the same brightness.
  45° views: far limbs drawn, slightly darker. Side view: far limbs slightly
  darker and shifted to the LEFT of the near ones.
- One hue with 3–4 shades per material, a 1 px dark outline, and the figure
  filling about 80% of the image height.
