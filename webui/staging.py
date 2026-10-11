"""Stage uploaded images into the layout GenIA's datasets expect.

Layouts (see demo/data/):
  image      {scene}/0000.png, 0000_mask.png, 0000_classes.json
  multiview  {scene}/rgbs/NNNNN.png, {scene}/segmentations/NNNNN.png, classes.json
  dynamic    {scene}/rgbs/NN.png,    {scene}/segmentations/NN.png,    classes.json
             (+ optional poses.json)

Segmentations are label images: pixel value = segment id, and classes.json maps
each id to {class, instance}. We only build single-object scenes, so id 1.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

GENIA_DIR = Path(os.environ.get("GENIA_HOME", "/opt/GenIA"))
DEMO_DATA = GENIA_DIR / "demo" / "data"

CLASSES = {"1": {"class": "object", "instance": 1}}


def new_scene_name() -> str:
    return f"ui_{int(time.time())}"


def _label_mask(mask: Image.Image) -> Image.Image:
    """Any mask image -> L-mode {0, 1} label map."""
    arr = np.asarray(mask.convert("L"))
    return Image.fromarray((arr > 127).astype(np.uint8), mode="L")


def _write_classes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(CLASSES, indent=2) + "\n")


def _scene_dir(setting: str, scene: str) -> Path:
    root = DEMO_DATA / setting / scene
    root.mkdir(parents=True, exist_ok=True)
    return root


def stage_image(rgb: Image.Image, mask: Image.Image, scene: str) -> str:
    root = _scene_dir("image", scene)
    rgb.convert("RGB").save(root / "0000.png")
    _label_mask(mask).save(root / "0000_mask.png")
    _write_classes(root / "0000_classes.json")
    return scene


def stage_multiview(rgbs: list[Image.Image], masks: list[Image.Image], scene: str) -> str:
    root = _scene_dir("multiview", scene)
    (root / "rgbs").mkdir(exist_ok=True)
    (root / "segmentations").mkdir(exist_ok=True)
    for i, (rgb, mask) in enumerate(zip(rgbs, masks)):
        name = f"{i:05d}.png"
        rgb.convert("RGB").save(root / "rgbs" / name)
        _label_mask(mask).save(root / "segmentations" / name)
    _write_classes(root / "segmentations" / "classes.json")
    return scene


def stage_dynamic(frames: list[Image.Image], masks: list[Image.Image],
                  poses_json: Path | None, scene: str) -> str:
    root = _scene_dir("dynamic", scene)
    (root / "rgbs").mkdir(exist_ok=True)
    (root / "segmentations").mkdir(exist_ok=True)
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        name = f"{i:02d}.png"
        frame.convert("RGB").save(root / "rgbs" / name)
        _label_mask(mask).save(root / "segmentations" / name)
    _write_classes(root / "segmentations" / "classes.json")
    if poses_json is not None:
        (root / "poses.json").write_bytes(Path(poses_json).read_bytes())
    return scene
