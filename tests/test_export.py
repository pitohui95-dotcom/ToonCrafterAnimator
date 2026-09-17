from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tooncrafter_animator.core.export import export_gif, export_mp4, export_png_sequence
from tooncrafter_animator.core.ffmpeg import resolve_ffmpeg


def _frames(n: int = 4, w: int = 64, h: int = 48) -> list[np.ndarray]:
    out = []
    for i in range(n):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 40
        arr[:, :, 1] = 80
        arr[:, :, 2] = 160
        out.append(arr)
    return out


def test_png_sequence_atomic(tmp_path: Path) -> None:
    dest = tmp_path / "seq"
    export_png_sequence(_frames(3), dest)
    files = sorted(dest.glob("frame_*.png"))
    assert len(files) == 3
    with Image.open(files[0]) as img:
        assert img.size == (64, 48)
    leftovers = list(tmp_path.glob(".*part*"))
    assert leftovers == []


def test_gif_pillow_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tooncrafter_animator.core.export.resolve_ffmpeg", lambda *a, **k: None)
    dest = tmp_path / "out.gif"
    export_gif(_frames(3), dest, fps=8, ffmpeg_path=None)
    assert dest.is_file()
    with Image.open(dest) as img:
        assert img.format == "GIF"
        assert img.n_frames == 3


def test_mp4_with_system_ffmpeg(tmp_path: Path) -> None:
    ffmpeg = resolve_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    dest = tmp_path / "out.mp4"
    export_mp4(_frames(4, w=64, h=48), dest, fps=8, ffmpeg_path=ffmpeg)
    assert dest.is_file()
    assert dest.stat().st_size > 0
