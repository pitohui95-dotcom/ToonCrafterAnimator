"""Assemble release/ToonCrafterAnimator/ exactly as specified."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "ToonCrafterAnimator"
TEMPLATE = ROOT / "packaging" / "release_template"
LICENSES_SRC = ROOT / "packaging" / "licenses"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "checksums.sha256":
            continue
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        type=Path,
        default=None,
        help="PyInstaller dist/ToonCrafterAnimator folder (optional on Linux).",
    )
    parser.add_argument("--ffmpeg-dir", type=Path, default=None)
    args = parser.parse_args()

    RELEASE.mkdir(parents=True, exist_ok=True)

    # README / notices from templates (always).
    for name in ("README.txt", "THIRD_PARTY_NOTICES.txt"):
        src = TEMPLATE / name
        if src.is_file():
            shutil.copy2(src, RELEASE / name)

    licenses_dest = RELEASE / "licenses"
    licenses_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "LICENSE", licenses_dest / "APP_LICENSE.txt")
    shutil.copy2(ROOT / "NOTICE", licenses_dest / "NOTICE.txt")
    vendor = ROOT / "src" / "tooncrafter_animator" / "vendor" / "tooncrafter"
    if (vendor / "LICENSE").is_file():
        shutil.copy2(vendor / "LICENSE", licenses_dest / "ComfyUI-ToonCrafter_LICENSE.txt")
    if (vendor / "ToonCrafter" / "LICENSE").is_file():
        shutil.copy2(vendor / "ToonCrafter" / "LICENSE", licenses_dest / "ToonCrafter_LICENSE.txt")
    if LICENSES_SRC.is_dir():
        for path in LICENSES_SRC.iterdir():
            if path.is_file():
                shutil.copy2(path, licenses_dest / path.name)
    shutil.copy2(vendor / "PROVENANCE.md", licenses_dest / "PROVENANCE.md")

    assets_dest = RELEASE / "assets"
    assets_src = ROOT / "src" / "tooncrafter_animator" / "assets"
    if assets_src.is_dir():
        assets_dest.mkdir(parents=True, exist_ok=True)
        for path in assets_src.iterdir():
            if path.is_file():
                shutil.copy2(path, assets_dest / path.name)

    ffmpeg_dest = RELEASE / "ffmpeg"
    ffmpeg_dest.mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = args.ffmpeg_dir or (ROOT / "packaging" / "ffmpeg_cache")
    copied = False
    if ffmpeg_dir.is_dir():
        for name in ("ffmpeg.exe", "ffmpeg"):
            src = ffmpeg_dir / name
            if src.is_file():
                shutil.copy2(src, ffmpeg_dest / src.name)
                copied = True
        for dll in ffmpeg_dir.glob("*.dll"):
            shutil.copy2(dll, ffmpeg_dest / dll.name)
            copied = True
    readme_ff = ffmpeg_dest / "README.txt"
    if copied:
        readme_ff.write_text(
            "LGPL ffmpeg build fetched by packaging/fetch_ffmpeg.py.\n"
            "See licenses/LGPLv2.1.txt and THIRD_PARTY_NOTICES.txt.\n",
            encoding="utf-8",
        )
    else:
        readme_ff.write_text(
            "No ffmpeg binary in this tree.\n"
            "On Windows, run packaging\\fetch_ffmpeg.py from build_exe.bat / .ps1\n"
            "or copy an LGPL ffmpeg.exe into this folder.\n"
            "GPL ffmpeg builds must not be redistributed with this app.\n",
            encoding="utf-8",
        )

    dist = args.dist
    if dist is None:
        for candidate in (
            ROOT / "dist" / "ToonCrafterAnimator",
            ROOT / "packaging" / "dist" / "ToonCrafterAnimator",
        ):
            if candidate.is_dir():
                dist = candidate
                break
    if dist and dist.is_dir():
        for item in dist.iterdir():
            dest = RELEASE / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    else:
        placeholder = RELEASE / "WINDOWS_EXE_NOT_BUILT.txt"
        placeholder.write_text(
            "A native Windows .exe cannot be cross-compiled from Linux.\n"
            "Run build_exe.bat or build_exe.ps1 on Windows 10/11 x64 (Python 3.10).\n"
            "Until then, run the app from source:\n"
            "  python -m tooncrafter_animator\n",
            encoding="utf-8",
        )

    write_checksums(RELEASE)
    print(f"Release assembled at {RELEASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
