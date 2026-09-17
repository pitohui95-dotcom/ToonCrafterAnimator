from __future__ import annotations

import os
import shutil
from pathlib import Path

from tooncrafter_animator.paths import bundled_ffmpeg_dir


def resolve_ffmpeg(override: str | None = None) -> Path | None:
    """bundled ffmpeg/ffmpeg.exe → settings override → PATH."""
    candidates: list[Path] = []
    bundled = bundled_ffmpeg_dir()
    candidates.extend(
        [
            bundled / "ffmpeg.exe",
            bundled / "ffmpeg",
            bundled / "bin" / "ffmpeg.exe",
            bundled / "bin" / "ffmpeg",
        ]
    )
    if override:
        candidates.insert(0, Path(override).expanduser())
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK if os.name != "nt" else os.F_OK):
            return path
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)
    return None
