"""spritekit: turn character pictures into animated 2D sprite sheets, and
reverse-engineer existing sprite sheets back into frames, grids and timing.

Modules
-------
image    RGBA I/O, background keying, connected components (numpy only).
analyze  Recover grid, frames, anchors, palette, duplicates and loops from a sheet.
rig      Auto-rig an exploded-view drawing (parts + #00FF00 joint dots).
bvh      BVH motion-capture parser and forward kinematics.
motion   Gait-cycle extraction, symmetrization, camera projection, retargeting.
render   Inverse-mapped skeletal renderer with soft joints.
pixel    Hi-res to pixel-art reduction with a shared palette and outline.
sheet    Sheet packing, Aseprite-compatible JSON, GIF previews.
pipeline Picture -> rig -> poses -> pixel frames -> directions -> sheet.
prompts  Image-model prompts (base model, turnaround, exploded views) and art checks.
demo     Synthetic exploded-view character used by the demo and tests.
"""

__version__ = "0.1.0"
