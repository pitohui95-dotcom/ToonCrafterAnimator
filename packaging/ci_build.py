"""Run the Windows packaging steps and abort on the first failure."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_torchvision_bundle(dist: Path) -> None:
    """Fail the Windows pack if torchvision's native ops extension is missing."""
    inits = list(dist.rglob("torchvision/__init__.py"))
    if not inits:
        raise SystemExit(f"packaged tree is missing torchvision: {dist}")
    pkg = inits[0].parent
    natives = [
        p
        for p in pkg.iterdir()
        if p.is_file()
        and (
            p.suffix.lower() in {".pyd", ".so", ".dll", ".dylib"}
            or p.name.startswith("_C")
            or p.name.startswith("image_stable")
        )
    ]
    if not natives:
        natives = list(pkg.glob("_C_stable*")) + list(pkg.glob("_C.*")) + list(pkg.glob("image_stable*"))
    if not natives:
        raise SystemExit(
            f"torchvision C++ extension missing next to {pkg} "
            "(need _C_stable / _C so torchvision::nms exists in the EXE)"
        )
    print("torchvision_native", [p.name for p in natives], flush=True)


def main() -> int:
    py = sys.executable
    dist = ROOT / "dist" / "ToonCrafterAnimator"
    steps = [
        [py, str(ROOT / "packaging" / "generate_icon.py")],
        [py, str(ROOT / "packaging" / "fetch_ffmpeg.py")],
        [py, "-m", "PyInstaller", "--noconfirm", "--clean", str(ROOT / "packaging" / "ToonCrafterAnimator.spec")],
        [
            py,
            str(ROOT / "packaging" / "make_release.py"),
            "--dist",
            str(dist),
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
        if cmd[1:3] == ["-m", "PyInstaller"] and dist.is_dir():
            verify_torchvision_bundle(dist)
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
