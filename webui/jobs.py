"""Run GenIA as a subprocess, streaming its output."""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

GENIA_DIR = Path(os.environ.get("GENIA_HOME", "/opt/GenIA"))


def demo_command(setting: str, scene: str, extra: list[str]) -> list[str]:
    """The `demo/run_demo.py` invocation for one scene (reuses GenIA's overrides)."""
    return [sys.executable, "demo/run_demo.py", setting, "--scene", scene, *extra]


def stream(cmd: list[str]) -> Iterator[str]:
    """Yield merged stdout/stderr lines; raise on a non-zero exit."""
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(GENIA_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"GenIA exited with code {proc.returncode}")
