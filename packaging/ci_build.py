"""Run the Windows packaging steps and abort on the first failure."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    py = sys.executable
    steps = [
        [py, str(ROOT / "packaging" / "generate_icon.py")],
        [py, str(ROOT / "packaging" / "fetch_ffmpeg.py")],
        [py, "-m", "PyInstaller", "--noconfirm", "--clean", str(ROOT / "packaging" / "ToonCrafterAnimator.spec")],
        [
            py,
            str(ROOT / "packaging" / "make_release.py"),
            "--dist",
            str(ROOT / "dist" / "ToonCrafterAnimator"),
            "--ffmpeg-dir",
            str(ROOT / "packaging" / "ffmpeg_cache"),
        ],
    ]
    for cmd in steps:
        print("+", *cmd, flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            print(f"step failed with {proc.returncode}: {cmd}", flush=True)
            return proc.returncode
    exe = ROOT / "release" / "ToonCrafterAnimator" / "ToonCrafterAnimator.exe"
    if not exe.is_file():
        # Linux CI / local: the binary may be extensionless.
        alt = ROOT / "release" / "ToonCrafterAnimator" / "ToonCrafterAnimator"
        if not alt.is_file():
            print(f"missing packaged binary at {exe}", flush=True)
            return 1
    print("ci_build_ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
