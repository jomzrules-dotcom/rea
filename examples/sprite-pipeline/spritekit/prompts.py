"""Image-model prompts for the art the pipeline needs, and feedback on what comes back.

Workflow (the image model draws, this toolkit rigs and animates):

1. ``design``      - the base model: one full-body character design.
2. ``turnaround``  - the same character in every view, standing, for consistency.
3. ``exploded``    - one exploded rig view per direction (parts apart, joint dots).
4. ``heads``       - optional hairstyle head sheet.

Then ``check`` the returned drawings; when something is off, ``review`` writes a
corrective prompt to send back to the image model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import rig as rigmod


@dataclass(frozen=True)
class ViewSpec:
    label: str
    facing: str
    limbs: str  # how to draw near/far limbs in this view


VIEWS: dict[str, ViewSpec] = {
    "front": ViewSpec("front view", "facing the viewer directly",
                      "Left and right limbs face the viewer equally: draw them with the same brightness."),
    "front_left": ViewSpec("45° front-left view", "three-quarter front, facing down-left",
                           "Draw the far arm and far leg too, slightly darker."),
    "side_left": ViewSpec("side view", "full profile, facing left",
                          "Draw the far arm and far leg too, slightly darker, shifted a little to the LEFT of the "
                          "near arm and leg so no parts overlap."),
    "back_left": ViewSpec("45° back-left view", "three-quarter back, facing up-left",
                          "Draw the far arm and far leg too, slightly darker."),
    "back": ViewSpec("back view", "facing directly away from the viewer",
                     "Left and right limbs are seen equally from behind: draw them with the same brightness."),
}

DIRECTION_SETS = {
    4: ["front_left", "back_left"],
    8: ["front", "front_left", "side_left", "back_left", "back"],
}


@dataclass
class Character:
    description: str
    style: str = "pixel-art chibi"
    game: str = "top-down action RPG"
    heads_tall: float = 2.5
    accent: str | None = None  # e.g. "red scarf"
    weapon: str | None = None  # e.g. "short sword"
    hairstyles: list[str] = field(default_factory=list)


@dataclass
class Prompt:
    step: str
    title: str
    file: str  # where to save the image the model returns
    attach: list[str]  # images to attach when sending the prompt
    text: str


@dataclass
class PromptPack:
    character: Character
    views: list[str]
    layout: str
    prompts: list[Prompt]
    check_command: str
    build_command: str

    def to_json(self) -> dict:
        return asdict(self)


def _colour_rules(c: Character) -> str:
    accent = f" Main accent ({c.accent}) one clear colour." if c.accent else ""
    return (
        "Colour rules: hair in ONE hue with 3-4 shades; skin ONE hue with 3 shades; eyes one iris hue, dark pupil, "
        f"small white highlight; each clothing piece one hue with 3 shades.{accent} Never use pure green (#00FF00) "
        "or magenta (#FF00FF) on the character."
    )


def _style(c: Character) -> str:
    return (f"Clean 1 px dark outline. Solid flat #FF00FF background, no shadow, no text, no logos, no gradients, "
            f"roughly left-right symmetric design, about {c.heads_tall:g} heads tall.")


def design_prompt(c: Character) -> str:
    gear = f" Carrying: {c.weapon}." if c.weapon else ""
    return (
        f"{c.style.capitalize()} character for a {c.game}: {c.description}.{gear} "
        "Relaxed A-pose, 45° front-left view (facing down-left), full body.\n"
        f"{_colour_rules(c)}\n{_style(c)}"
    )


def turnaround_prompt(c: Character, views: list[str]) -> str:
    order = ", ".join(f"{i + 1}) {VIEWS[v].label} ({VIEWS[v].facing})" for i, v in enumerate(views))
    return (
        f"Using the attached character design, draw a turnaround model sheet of the SAME character in {len(views)} "
        f"views side by side, left to right: {order}.\n"
        "Relaxed A-pose in every view, same height, same scale, feet on one shared baseline, same palette and "
        "outline as the design. Clear empty space between the figures.\n"
        f"{_colour_rules(c)}\n{_style(c)}"
    )


def _parts(c: Character) -> str:
    extras = []
    if c.weapon:
        extras.append(f"the {c.weapon} as its own part next to the hand that holds it, with a small gap")
    if c.accent:
        extras.append(f"the {c.accent} tail (if any) as its own part")
    tail = (", " + ", ".join(extras)) if extras else ""
    return ("head (hair attached), torso with hips, BOTH upper arms, BOTH forearms with hands, "
            f"BOTH thighs, BOTH shins with feet{tail}")


_EXPLODED_RULES = (
    "Exploded view: relaxed A-pose with every body part pulled slightly apart by a small even gap (about 3% of the "
    "figure height), each part in its natural position. Arms angled well away from the body, legs slightly apart.\n"
    "Keep a clear gap between head and torso. No part touches another part.\n"
    "At EVERY joint put a small solid #00FF00 dot centred in the gap: neck, both shoulders, both elbows, both hips, "
    "both knees (exactly 9 dots, no glow, no outline around the dots).\n"
    "Limb ends rounded so they overlap when rejoined; torso drawn complete under where the arms attach."
)


def exploded_prompt(c: Character, view: str) -> str:
    v = VIEWS[view]
    return (
        f"Using the attached character design and turnaround, draw ONE exploded view of the same character: "
        f"{v.label} ({v.facing}).\n"
        f"Parts: {_parts(c)}.\n{v.limbs}\n{_EXPLODED_RULES}\n"
        "Same scale, palette and 1 px outline as the design. The figure fills about 80% of the image height. "
        "Solid flat #FF00FF background, no shadows, no text."
    )


def exploded_sheet_prompt(c: Character, views: list[str]) -> str:
    order = "\n".join(f"{i + 1}. {VIEWS[v].label} ({VIEWS[v].facing}). {VIEWS[v].limbs}" for i, v in enumerate(views))
    return (
        f"Using the attached character design and turnaround, draw {len(views)} exploded views of the same character "
        f"side by side on one image, left to right:\n{order}\n"
        f"Parts in every view: {_parts(c)}.\n{_EXPLODED_RULES}\n"
        "Leave a wide empty gap between the figures. Same scale, palette and 1 px outline in every view. "
        "Solid flat #FF00FF background, no shadows, no text."
    )


def heads_prompt(c: Character, views: list[str]) -> str:
    rows = ", ".join(f"row {i + 1} {VIEWS[v].label}" for i, v in enumerate(views))
    return (
        f"Using the attached character's face, draw a grid of complete heads (face + hair, no neck, no body): "
        f"{len(c.hairstyles)} columns = {len(c.hairstyles)} hairstyles: {', '.join(c.hairstyles)}. "
        f"{len(views)} rows: {rows}.\n"
        "Same size, same face and skin, chin at the same height in each cell. Hair in ONE hue with 3-4 shades. "
        "Same pixel style and 1 px outline. Clear space between heads. Solid flat #FF00FF background, no text."
    )


def prompt_pack(c: Character, directions: int = 8, layout: str = "separate",
                bvh_path: str = "mocap/07_01.bvh") -> PromptPack:
    if directions not in DIRECTION_SETS:
        raise ValueError(f"directions must be one of {sorted(DIRECTION_SETS)}")
    if layout not in ("separate", "sheet"):
        raise ValueError("layout must be 'separate' (one image per view) or 'sheet' (all views in one image)")
    views = DIRECTION_SETS[directions]
    prompts = [
        Prompt("design", "Base model: character design", "design.png", [], design_prompt(c)),
        Prompt("turnaround", "Base model in every view", "turnaround.png", ["design.png"], turnaround_prompt(c, views)),
    ]
    refs = ["design.png", "turnaround.png"]
    if layout == "separate":
        files = [f"view_{v}.png" for v in views]
        for v, f in zip(views, files):
            prompts.append(Prompt("exploded", f"Exploded rig view: {VIEWS[v].label}", f, refs, exploded_prompt(c, v)))
    else:
        files = ["views.png"]
        prompts.append(Prompt("exploded", "Exploded rig views (one image)", files[0], refs, exploded_sheet_prompt(c, views)))
    if c.hairstyles:
        prompts.append(Prompt("heads", "Hairstyle head sheet", "heads.png", ["design.png"], heads_prompt(c, views)))
    joined = ",".join(views)
    drawings = " ".join(files)
    return PromptPack(
        c, views, layout, prompts,
        check_command=f"python3 -m spritekit check {drawings} --views {joined}",
        build_command=f"python3 -m spritekit build {drawings} --views {joined} --bvh {bvh_path} --out out",
    )


def to_markdown(pack: PromptPack) -> str:
    lines = [f"# Sprite art prompts: {pack.character.description}", "",
             f"Directions: {len(pack.views) * 2 - sum(v in ('front', 'back') for v in pack.views)} "
             f"(draw {len(pack.views)} views: {', '.join(VIEWS[v].label for v in pack.views)}; "
             "left-facing views are mirrored for the right-facing ones).", ""]
    for i, p in enumerate(pack.prompts, 1):
        lines.append(f"## {i}. {p.title}")
        lines.append("")
        lines.append(f"Save as `{p.file}`" + (f"; attach {', '.join(f'`{a}`' for a in p.attach)}." if p.attach else "."))
        lines += ["", "```", p.text, "```", ""]
    lines += ["## Then", "", "```", pack.check_command, pack.build_command, "```", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# feedback on returned drawings


@dataclass
class Finding:
    figure: str  # view name, or "drawing"
    problem: str
    fix: str  # instruction for the image model


@dataclass
class Review:
    findings: list[Finding]
    fix_prompt: str | None

    @property
    def ok(self) -> bool:
        return not self.findings


BODY_PARTS = 10  # head, torso, 2 upper arms, 2 forearms, 2 thighs, 2 shins


def review(drawing: np.ndarray, views: list[str], params: rigmod.RigParams | None = None) -> Review:
    """Check one returned drawing (holding ``views`` left to right) against the rig conventions."""
    findings: list[Finding] = []
    try:
        seg = rigmod.segment(drawing, len(views), params)
    except rigmod.RigError as e:
        msg = str(e)
        fix = ("Use a solid flat #FF00FF background with nothing else on it." if "no body parts" in msg else
               f"Draw exactly {len(views)} figure(s) side by side with a wide empty gap between them.")
        findings.append(Finding("drawing", msg, fix))
        return Review(findings, _fix_prompt(findings, views))
    if seg.magenta_fraction < 0.3:
        findings.append(Finding("drawing", f"only {seg.magenta_fraction:.0%} of the image is #FF00FF background",
                                "Use a solid flat #FF00FF background (no gradient, texture or shadow)."))
    for view, (parts, dots) in zip(views, seg.figures):
        if len(dots) not in (7, 9):
            findings.append(Finding(view, f"found {len(dots)} green joint dots, expected 9",
                                    "Put exactly 9 solid #00FF00 dots, one centred in each joint gap: neck, both "
                                    "shoulders, both elbows, both hips, both knees. No other pure green anywhere."))
        if len(parts) < BODY_PARTS:
            findings.append(Finding(view, f"found {len(parts)} separate parts, expected at least {BODY_PARTS}; "
                                    "some parts touch or are merged",
                                    "Separate every part with a clear even gap: head, torso, both upper arms, both "
                                    "forearms, both thighs, both shins. No part may touch another."))
        if any(f.figure == view for f in findings):
            continue
        try:
            r = rigmod.auto_rig(_crop_figure(drawing, parts), [view], params)[0]
        except rigmod.RigError as e:
            findings.append(Finding(view, str(e), "Keep each limb part next to its joint dots and the gaps even."))
            continue
        if r.height < 150:
            findings.append(Finding(view, f"figure is only {r.height:.0f} px tall",
                                    "Draw the figure larger: it should fill about 80% of the image height."))
    return Review(findings, _fix_prompt(findings, views) if findings else None)


def _crop_figure(drawing: np.ndarray, parts) -> np.ndarray:
    x0 = max(min(p.box[0] for p in parts) - 25, 0)
    x1 = min(max(p.box[2] for p in parts) + 25, drawing.shape[1])
    return drawing[:, x0:x1]


def _fix_prompt(findings: list[Finding], views: list[str]) -> str:
    labels = ", ".join(VIEWS[v].label for v in views)
    fixes = []
    for f in findings:
        where = "" if f.figure == "drawing" else f" In the {VIEWS[f.figure].label}:"
        line = f"-{where} {f.fix}"
        if line not in fixes:
            fixes.append(line)
    return (f"Redraw the attached exploded view ({labels}) of the same character with these corrections:\n"
            + "\n".join(fixes) + "\nKeep everything else the same: palette, outline, scale, solid flat #FF00FF "
            "background, no shadows, no text.")
