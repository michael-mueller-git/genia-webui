"""Locate and collect a finished run's outputs under GENIA_RESULTS."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

RESULTS = Path(os.environ.get("GENIA_RESULTS", "/data/results"))


def find_latest_final(results_root: Path | None = None) -> Path | None:
    """Newest results/.../<timestamp>/final directory holding poses.json."""
    root = Path(results_root or RESULTS)
    finals = [
        Path(entry)
        for entry, _dirs, files in os.walk(root)
        if os.path.basename(entry) == "final" and "poses.json" in files
    ]
    if not finals:
        return None
    return max(finals, key=lambda p: p.stat().st_mtime)


def _zip_final(final_dir: Path) -> str:
    out = final_dir.parent / f"{final_dir.parent.name}_final.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(final_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(final_dir.parent))
    return str(out)


def collect(final_dir: Path) -> dict:
    """Bundle a final/ dir into UI-ready paths."""
    final_dir = Path(final_dir)

    video = None
    orbit = final_dir / "viz" / "orbit" / "orbit.mp4"
    if orbit.is_file():
        video = str(orbit)

    images = sorted(str(p) for p in (final_dir / "renders_train").glob("**/*.png"))
    if not images:
        images = sorted(str(p) for p in (final_dir / "viz").glob("**/*.png"))[:8]

    meshes = sorted((final_dir / "meshes").glob("*.glb"))
    poses = final_dir / "poses.json"

    downloads = [_zip_final(final_dir)]
    if meshes:
        downloads.append(str(meshes[0]))
    if poses.is_file():
        downloads.append(str(poses))

    return {
        "video": video,
        "images": images,
        "downloads": downloads,
        "final_dir": str(final_dir),
    }
