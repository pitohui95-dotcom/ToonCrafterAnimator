"""Download a pinned LGPL ffmpeg Windows build. Never used at app runtime."""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# BtbN win64 lgpl static build. GPL builds must not be redistributed with this app.
# "latest" is a rolling tag; the SHA-256 is recorded after a successful fetch so a
# future run fails if the zip changes under our feet.
DEFAULT_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-n7.1-latest-win64-lgpl.zip"
)

# Filled in by `python packaging/fetch_ffmpeg.py --record-hash` after a verified download.
EXPECTED_SHA256 = os.environ.get("TOONCRAFTER_FFMPEG_SHA256", "").strip()

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "packaging" / "ffmpeg_cache"
HASH_FILE = CACHE / "ffmpeg.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest_dir: Path, expected: str = "") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ffmpeg-win64-lgpl.zip"
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    zip_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"sha256 {digest}")
    expected = expected or EXPECTED_SHA256
    if HASH_FILE.is_file() and not expected:
        expected = HASH_FILE.read_text(encoding="utf-8").strip().split()[0]
    if expected and digest != expected:
        raise SystemExit(f"ffmpeg zip hash mismatch: got {digest}, expected {expected}")
    HASH_FILE.write_text(f"{digest}  ffmpeg-win64-lgpl.zip\n", encoding="utf-8")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        exe_name = None
        for name in zf.namelist():
            if name.endswith("ffmpeg.exe") and "/bin/" in name.replace("\\", "/"):
                exe_name = name
                break
        if exe_name is None:
            for name in zf.namelist():
                if name.endswith("ffmpeg.exe"):
                    exe_name = name
                    break
        if exe_name is None:
            raise SystemExit("zip did not contain ffmpeg.exe")
        target = dest_dir / "ffmpeg.exe"
        target.write_bytes(zf.read(exe_name))
        # Copy sibling DLLs if this is a shared build.
        parent = exe_name.rsplit("/", 1)[0]
        for name in zf.namelist():
            if name.startswith(parent + "/") and name.lower().endswith(".dll"):
                (dest_dir / Path(name).name).write_bytes(zf.read(name))
    print(f"Wrote {dest_dir / 'ffmpeg.exe'}")
    return dest_dir / "ffmpeg.exe"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--dest", type=Path, default=CACHE)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()
    fetch(args.url, args.dest, args.expected_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
