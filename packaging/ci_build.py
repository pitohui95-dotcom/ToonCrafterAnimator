"""Run the Windows packaging steps and abort on the first failure."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CUDA_NAME_HINTS = (
    "cublas",
    "cudnn",
    "cudart",
    "nvrtc",
    "c10_cuda",
    "torch_cuda",
    "cusolver",
    "cusparse",
    "cufft",
    "curand",
    "nvjitlink",
    "cupti",
)


def verify_tooncrafter_bundle(dist: Path) -> None:
    """Fail the pack if the real ToonCrafter package / 512 yaml never landed in the onedir."""
    inits = [p for p in dist.rglob("ToonCrafter/__init__.py") if p.is_file()]
    if not inits:
        raise SystemExit(f"packaged tree is missing ToonCrafter/__init__.py: {dist}")
    utils = [p for p in dist.rglob("ToonCrafter/utils/utils.py") if p.is_file()]
    if not utils:
        raise SystemExit(f"packaged tree is missing ToonCrafter/utils/utils.py: {dist}")
    yamls = [
        p
        for p in dist.rglob("inference_512_v1.0.yaml")
        if p.is_file() and "ToonCrafter" in p.as_posix()
    ]
    if not yamls:
        raise SystemExit(f"packaged tree is missing ToonCrafter configs/inference_512_v1.0.yaml: {dist}")
    lvdm = [p for p in dist.rglob("lvdm/basics.py") if p.is_file()]
    if not lvdm:
        raise SystemExit(f"packaged tree is missing lvdm/basics.py: {dist}")
    print(
        "tooncrafter_package",
        [str(p.relative_to(dist)) for p in inits[:4]],
        "yaml",
        [str(p.relative_to(dist)) for p in yamls[:4]],
        flush=True,
    )


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


def _native_files(root: Path) -> list[Path]:
    out = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".dll", ".pyd", ".so", ".dylib"}:
            out.append(path)
    return out


def verify_cuda_bundle(dist: Path) -> None:
    """Fail if this was supposed to be a CUDA onedir but only CPU torch landed."""
    try:
        import torch

        cuda_ver = getattr(torch.version, "cuda", None)
    except Exception as exc:
        raise SystemExit(f"cannot import torch while verifying CUDA bundle: {exc}") from exc
    if not cuda_ver:
        raise SystemExit(
            "TORCH_VARIANT=cuda but torch.version.cuda is empty — refusing to label a CPU wheel as GPU"
        )
    natives = _native_files(dist)
    hits = [
        p
        for p in natives
        if any(hint in p.name.lower() for hint in CUDA_NAME_HINTS)
    ]
    if len(hits) < 2:
        sample = sorted({p.name for p in natives})[:30]
        raise SystemExit(
            f"CUDA native libraries missing from {dist} (need cublas/cudnn/c10_cuda/torch_cuda). "
            f"found_hints={ [p.name for p in hits] } sample={sample}"
        )
    print(
        "cuda_bundle",
        "torch.version.cuda",
        cuda_ver,
        "hits",
        sorted({p.name for p in hits})[:20],
        flush=True,
    )


def packaging_variant() -> str:
    raw = os.environ.get("TORCH_VARIANT", "").strip().lower()
    if raw in {"cuda", "cpu"}:
        return raw
    try:
        import torch

        return "cuda" if getattr(torch.version, "cuda", None) else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    py = sys.executable
    dist = ROOT / "dist" / "ToonCrafterAnimator"
    variant = packaging_variant()
    print("packaging_variant", variant, flush=True)
    if variant == "cuda":
        try:
            import torch

            if not getattr(torch.version, "cuda", None):
                print("refusing CUDA pack: torch.version.cuda is empty", flush=True)
                return 2
        except Exception as exc:
            print(f"refusing CUDA pack: torch import failed: {exc}", flush=True)
            return 2
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
            "--variant",
            variant,
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
            verify_tooncrafter_bundle(dist)
            if variant == "cuda":
                verify_cuda_bundle(dist)
    release = ROOT / "release" / "ToonCrafterAnimator"
    if release.is_dir():
        verify_tooncrafter_bundle(release)
        tv_inits = list(release.rglob("torchvision/__init__.py"))
        if tv_inits:
            verify_torchvision_bundle(release)
        if variant == "cuda":
            verify_cuda_bundle(release)
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
