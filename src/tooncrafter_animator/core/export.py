from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from tooncrafter_animator.core.errors import ExportError
from tooncrafter_animator.core.ffmpeg import resolve_ffmpeg
from tooncrafter_animator.core.images import array_to_image
from tooncrafter_animator import copy as t


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_replace(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)


def export_png_sequence(frames: list[np.ndarray], dest_dir: Path) -> Path:
    dest_dir = Path(dest_dir)
    parent = dest_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    part = parent / f".{dest_dir.name}.part-{os.getpid()}"
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    try:
        for index, frame in enumerate(frames):
            path = part / f"frame_{index:04d}.png"
            array_to_image(frame).save(path, format="PNG")
        _fsync_dir(part)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        os.replace(part, dest_dir)
    except Exception:
        shutil.rmtree(part, ignore_errors=True)
        raise
    return dest_dir


def _even_dims(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    new_w = w - (w % 2)
    new_h = h - (h % 2)
    if new_w == w and new_h == h:
        return frame
    if new_w <= 0 or new_h <= 0:
        raise ExportError(t.ERR_FRAME_TOO_SMALL)
    return frame[:new_h, :new_w]


def export_mp4(frames: list[np.ndarray], dest: Path, fps: int, ffmpeg_path: Path | None = None) -> Path:
    dest = Path(dest)
    ffmpeg = ffmpeg_path or resolve_ffmpeg()
    if ffmpeg is None:
        raise ExportError(t.ERR_NO_FFMPEG)
    if fps <= 0:
        raise ExportError(t.ERR_FPS_POSITIVE)
    even = [_even_dims(f) for f in frames]
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".mp4-part-", dir=str(parent)))
    staging = dest.with_name(f"{dest.stem}.part-{os.getpid()}.mp4")
    try:
        for index, frame in enumerate(even):
            array_to_image(frame).save(work / f"frame_{index:04d}.png")
        cmd = [
            str(ffmpeg),
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(work / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(staging),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ExportError(t.ERR_FFMPEG_MP4.format(stderr=proc.stderr[-2000:]))
        _atomic_replace(staging, dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if staging.exists():
            staging.unlink(missing_ok=True)
    return dest


def export_gif(
    frames: list[np.ndarray],
    dest: Path,
    fps: int,
    ffmpeg_path: Path | None = None,
) -> Path:
    dest = Path(dest)
    ffmpeg = ffmpeg_path or resolve_ffmpeg()
    duration_ms = max(1, int(round(1000 / max(fps, 1))))
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(f"{dest.stem}.part-{os.getpid()}.gif")
    if ffmpeg is not None:
        work = Path(tempfile.mkdtemp(prefix=".gif-part-", dir=str(parent)))
        try:
            for index, frame in enumerate(frames):
                array_to_image(frame).save(work / f"frame_{index:04d}.png")
            palette = work / "palette.png"
            gen = subprocess.run(
                [
                    str(ffmpeg),
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(work / "frame_%04d.png"),
                    "-vf",
                    "palettegen",
                    str(palette),
                ],
                capture_output=True,
                text=True,
            )
            if gen.returncode != 0:
                raise ExportError(t.ERR_FFMPEG_PALETTEGEN.format(stderr=gen.stderr[-2000:]))
            use = subprocess.run(
                [
                    str(ffmpeg),
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(work / "frame_%04d.png"),
                    "-i",
                    str(palette),
                    "-lavfi",
                    "paletteuse",
                    str(staging),
                ],
                capture_output=True,
                text=True,
            )
            if use.returncode != 0:
                raise ExportError(t.ERR_FFMPEG_PALETTEUSE.format(stderr=use.stderr[-2000:]))
            _atomic_replace(staging, dest)
            return dest
        finally:
            shutil.rmtree(work, ignore_errors=True)
            if staging.exists():
                staging.unlink(missing_ok=True)
    images = [array_to_image(f) for f in frames]
    if not images:
        raise ExportError(t.ERR_NO_FRAMES)
    images[0].save(
        staging,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    _atomic_replace(staging, dest)
    return dest
