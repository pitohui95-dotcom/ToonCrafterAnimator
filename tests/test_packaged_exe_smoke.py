from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "ToonCrafterAnimator"


def _exe() -> Path | None:
    for name in ("ToonCrafterAnimator.exe", "ToonCrafterAnimator"):
        candidate = RELEASE / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


@pytest.mark.packaged
def test_packaged_exe_selftest() -> None:
    exe = _exe()
    if exe is None:
        pytest.skip(
            "No packaged binary in release/ToonCrafterAnimator/. "
            "On Windows run build_exe.bat; on Linux this skip is expected."
        )
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [str(exe), "--selftest"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(RELEASE),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data.get("ok") is True


def test_module_selftest() -> None:
    """Always-available stand-in so Linux CI still boots the app object."""
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "tooncrafter_animator", "--selftest"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data.get("ok") is True
    assert data.get("window")
    assert data.get("tooncrafter_ok") is True
    assert data.get("inference_yaml_ok") is True
