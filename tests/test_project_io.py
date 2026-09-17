from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tooncrafter_animator.core.errors import ProjectError
from tooncrafter_animator.core.models import GenerationParams
from tooncrafter_animator.core.project import load_project, save_project


def _png(path: Path) -> Path:
    Image.new("RGB", (32, 32), (12, 34, 56)).save(path)
    return path


def test_roundtrip(tmp_path: Path) -> None:
    start = _png(tmp_path / "a.png")
    end = _png(tmp_path / "b.png")
    dest = tmp_path / "job.json"
    params = GenerationParams(intermediates=7, fps=12, prompt="a cat")
    save_project(dest, start, end, params, export_dir=str(tmp_path))
    loaded = load_project(dest)
    assert loaded.generation.intermediates == 7
    assert loaded.generation.fps == 12
    assert loaded.generation.prompt == "a cat"
    assert loaded.start.resolved(tmp_path) == start.resolve()
    assert loaded.end.resolved(tmp_path) == end.resolve()
    assert loaded.missing_keyframes == []


def test_missing_keyframes_reported(tmp_path: Path) -> None:
    dest = tmp_path / "job.json"
    save_project(dest, tmp_path / "gone.png", None, GenerationParams())
    loaded = load_project(dest)
    assert "start" in loaded.missing_keyframes


def test_schema_rejection(tmp_path: Path) -> None:
    dest = tmp_path / "old.json"
    dest.write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(ProjectError):
        load_project(dest)
