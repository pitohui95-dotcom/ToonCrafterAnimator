from __future__ import annotations

from pathlib import Path

import pytest


def test_appdata_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOONCRAFTER_APPDATA", str(tmp_path / "data"))
    from tooncrafter_animator.paths import settings_path, user_data_dir

    assert user_data_dir() == tmp_path / "data"
    assert settings_path() == tmp_path / "data" / "settings.json"


def test_atomic_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOONCRAFTER_APPDATA", str(tmp_path))
    from tooncrafter_animator.settings import load_settings, save_settings

    s = load_settings()
    s.checkpoint_folder = "/models"
    save_settings(s)
    loaded = load_settings()
    assert loaded.checkpoint_folder == "/models"


def test_corrupt_restored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOONCRAFTER_APPDATA", str(tmp_path))
    from tooncrafter_animator.paths import settings_path
    from tooncrafter_animator.settings import load_settings

    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    loaded = load_settings()
    assert loaded.checkpoint_folder == ""
    assert path.with_suffix(".json.corrupt").is_file()
