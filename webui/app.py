"""Gradio Web UI for GenIA (Generative Reconstruction with Test-Time Input Alignment).

Wraps GenIA's three demo settings (single image / multiview / monocular video) by
staging uploads into GenIA's dataset layout and calling demo/run_demo.py, then
collecting the run's final/ outputs. GPU-bound: one job at a time.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autoseg
import jobs
import outputs
import staging

GENIA_DIR = Path(os.environ.get("GENIA_HOME", "/opt/GenIA"))
WEIGHTS_DIR = Path(os.environ.get("HF_HUB_CACHE", "/data/weights/huggingface/hub")).parents[1]
WEIGHTS_MARKER = WEIGHTS_DIR / ".genia_weights_ok"


# --- helpers ---------------------------------------------------------------

def _pil(value) -> Image.Image:
    return value if isinstance(value, Image.Image) else Image.open(value)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _overrides(text: str) -> list[str]:
    return shlex.split(text or "")


def _stream(setting: str, scene: str, extra: list[str]):
    """Generator yielding (log, video, gallery, downloads) for one run."""
    cmd = jobs.demo_command(setting, scene, extra)
    log = "$ " + " ".join(shlex.quote(c) for c in cmd) + "\n"
    yield log, None, [], []
    try:
        for line in jobs.stream(cmd):
            log += line
            yield log, None, [], []
    except Exception as exc:  # noqa: BLE001 - surface the failure in the UI
        yield log + f"\nERROR: {exc}\n", None, [], []
        return

    final = outputs.find_latest_final()
    if final is None:
        yield log + "\nNo final/ output found under GENIA_RESULTS.\n", None, [], []
        return
    result = outputs.collect(final)
    yield log + f"\nDone: {result['final_dir']}\n", \
        result["video"], result["images"], result["downloads"]


# --- handlers --------------------------------------------------------------

def run_image(rgb, mask, icp, overrides):
    if rgb is None:
        yield "Upload an RGB image first.", None, [], []
        return
    rgb_pil = _pil(rgb)
    if mask is None:
        yield "No mask given; segmenting the foreground with RMBG-1.4 ...", None, [], []
        mask_pil = autoseg.predict_mask(rgb_pil)
    else:
        mask_pil = _pil(mask)
    scene = staging.new_scene_name()
    staging.stage_image(rgb_pil, mask_pil, scene)
    extra = (["--icp"] if icp else []) + _overrides(overrides)
    yield from _stream("image", scene, extra)


def run_multiview(rgbs, masks, icp, overrides):
    views = _as_list(rgbs)
    if not views:
        yield "Upload at least two views.", None, [], []
        return
    views = sorted(views, key=lambda p: Path(str(p)).name)
    view_imgs = [_pil(v) for v in views]
    given = sorted(_as_list(masks), key=lambda p: Path(str(p)).name)
    if given and len(given) != len(view_imgs):
        yield "Number of masks must match the number of views.", None, [], []
        return
    if given:
        mask_imgs = [_pil(m) for m in given]
    else:
        yield "No masks given; segmenting each view with RMBG-1.4 ...", None, [], []
        mask_imgs = [autoseg.predict_mask(v) for v in view_imgs]
    scene = staging.new_scene_name()
    staging.stage_multiview(view_imgs, mask_imgs, scene)
    extra = (["--icp"] if icp else []) + _overrides(overrides)
    yield from _stream("multiview", scene, extra)


def run_dynamic(frames, masks, poses, per_frame_shapes, overrides):
    files = sorted(_as_list(frames), key=lambda p: Path(str(p)).name)
    if len(files) < 2:
        yield "Upload a frame sequence (2 or more images).", None, [], []
        return
    frame_imgs = [_pil(f) for f in files]
    given = sorted(_as_list(masks), key=lambda p: Path(str(p)).name)
    if given and len(given) != len(frame_imgs):
        yield "Number of masks must match the number of frames.", None, [], []
        return
    if given:
        mask_imgs = [_pil(m) for m in given]
    else:
        yield "No masks given; segmenting each frame with RMBG-1.4 ...", None, [], []
        mask_imgs = [autoseg.predict_mask(f) for f in frame_imgs]
    poses_path = Path(str(poses)) if poses else None
    scene = staging.new_scene_name()
    staging.stage_dynamic(frame_imgs, mask_imgs, poses_path, scene)
    extra = (["--per-frame-shapes"] if per_frame_shapes else []) + _overrides(overrides)
    yield from _stream("dynamic", scene, extra)


def run_download():
    log = ""
    for line in jobs.stream([sys.executable, "-m", "genia.core.download_weights"]):
        log += line
        yield log
    WEIGHTS_MARKER.touch(exist_ok=True)
    yield log + "\nWeights ready.\n"


def status_markdown() -> str:
    lines = []
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            lines.append(f"**GPU:** {torch.cuda.get_device_name(0)} (compute capability {major}.{minor})")
        else:
            lines.append("**GPU:** not available")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"**GPU:** error ({exc})")
    lines.append(f"**Weights:** {'ready' if WEIGHTS_MARKER.is_file() else 'not downloaded yet'}"
                 f" (`{WEIGHTS_DIR}`)")
    return "\n\n".join(lines)


# --- UI --------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GenIA") as demo:
        gr.Markdown("# GenIA — Generative Reconstruction with Test-Time Input Alignment\n"
                    "Single image, a few views, or a monocular video → canonical Gaussians and a mesh.")
        with gr.Row():
            status = gr.Markdown(status_markdown())
            refresh = gr.Button("Refresh status", scale=0)
        refresh.click(status_markdown, outputs=status)

        with gr.Tab("Single image"):
            with gr.Row():
                img_rgb = gr.Image(label="RGB image", type="pil")
                img_mask = gr.Image(label="Mask (optional; auto-segmented if empty)", type="pil")
            img_icp = gr.Checkbox(label="ICP pose refinement (instead of photometric)")
            img_ovr = gr.Textbox(label="Extra Hydra overrides", placeholder="e.g. appearance_init_2.inference_steps=10")
            img_btn = gr.Button("Reconstruct", variant="primary")
            img_outs = _outputs_block()
            img_btn.click(run_image, [img_rgb, img_mask, img_icp, img_ovr], img_outs)

        with gr.Tab("Multiview"):
            with gr.Row():
                mv_rgb = gr.File(label="Views (RGB)", file_count="multiple", file_types=["image"])
                mv_mask = gr.File(label="Masks (optional, matched by file name)", file_count="multiple", file_types=["image"])
            mv_icp = gr.Checkbox(label="ICP pose refinement (instead of photometric)")
            mv_ovr = gr.Textbox(label="Extra Hydra overrides")
            mv_btn = gr.Button("Reconstruct", variant="primary")
            mv_outs = _outputs_block()
            mv_btn.click(run_multiview, [mv_rgb, mv_mask, mv_icp, mv_ovr], mv_outs)

        with gr.Tab("Video"):
            with gr.Row():
                dyn_frames = gr.File(label="Frames (ordered by file name)", file_count="multiple", file_types=["image"])
                dyn_mask = gr.File(label="Masks (optional, matched by file name)", file_count="multiple", file_types=["image"])
            dyn_poses = gr.File(label="poses.json (optional)", file_count="single", file_types=[".json"])
            dyn_perframe = gr.Checkbox(label="Per-frame shapes (skip ActionMesh)")
            dyn_ovr = gr.Textbox(label="Extra Hydra overrides")
            dyn_btn = gr.Button("Reconstruct", variant="primary")
            dyn_outs = _outputs_block()
            dyn_btn.click(run_dynamic, [dyn_frames, dyn_mask, dyn_poses, dyn_perframe, dyn_ovr], dyn_outs)

        with gr.Tab("Setup"):
            gr.Markdown(
                "Model weights are stored on the mounted `/data/weights` volume.\n\n"
                "The gated `facebook/sam-3d-objects` repository requires an accepted licence "
                "and an `HF_TOKEN` passed to the container."
            )
            setup_log = gr.Textbox(label="Download log", lines=12, max_lines=12)
            dl_btn = gr.Button("Download weights now")
            dl_btn.click(run_download, outputs=setup_log)

    return demo


def _outputs_block():
    log = gr.Textbox(label="Log", lines=16, max_lines=16, autoscroll=True)
    with gr.Row():
        video = gr.Video(label="Orbit")
        gallery = gr.Gallery(label="Renders (input views)", columns=4)
    downloads = gr.Files(label="Downloads (final.zip, mesh, poses.json)")
    return [log, video, gallery, downloads]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
