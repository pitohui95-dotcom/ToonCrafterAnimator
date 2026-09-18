"""Zip the Windows onedir with ZIP64, split if GitHub's 2 GB file limit applies."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# GitHub.com release asset limit is 2 GiB. Stay under it with room for headers.
MAX_GITHUB_BYTES = 1900 * 1024 * 1024


def zip_onedir(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(src).as_posix())
    print(f"zipped {src} -> {dest} ({dest.stat().st_size} bytes)", flush=True)
    return dest


def split_if_needed(path: Path, max_bytes: int = MAX_GITHUB_BYTES) -> list[Path]:
    size = path.stat().st_size
    if size <= max_bytes:
        return [path]
    parts: list[Path] = []
    idx = 1
    with path.open("rb") as src:
        while True:
            chunk = src.read(max_bytes)
            if not chunk:
                break
            part = path.with_name(f"{path.name}.part{idx:02d}")
            part.write_bytes(chunk)
            parts.append(part)
            print(f"part {part.name} {len(chunk)} bytes", flush=True)
            idx += 1
    join = path.with_name(path.name + ".JOIN.txt")
    names = "+".join(p.name for p in parts)
    join.write_text(
        "GitHub 单个附件不能超过 2 GB，因此 CUDA 包被分成了多个分卷。\n\n"
        "在 PowerShell / cmd 中合并：\n\n"
        f"  copy /b {names} {path.name}\n\n"
        "然后解压合并后的 zip。不要只解压其中一个 .part 文件。\n",
        encoding="utf-8",
    )
    parts.append(join)
    print(f"split {path.name} into {len(parts) - 1} parts", flush=True)
    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("release/ToonCrafterAnimator"))
    parser.add_argument("--dest", type=Path, default=Path("ToonCrafterAnimator-windows-x64-cuda.zip"))
    parser.add_argument("--out-dir", type=Path, default=Path("release_upload"))
    args = parser.parse_args()
    if not args.src.is_dir():
        print(f"missing onedir {args.src}", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zipped = zip_onedir(args.src, args.out_dir / args.dest.name)
    outputs = split_if_needed(zipped)
    if len(outputs) > 1:
        # Do not attach the oversized combined zip to GitHub Releases.
        zipped.unlink(missing_ok=True)
        outputs = [p for p in outputs if p.exists()]
    for path in sorted(args.out_dir.iterdir()):
        if path.is_file():
            print(f"upload_candidate {path.name} {path.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
