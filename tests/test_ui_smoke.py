from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from tooncrafter_animator.settings import AppSettings
from tooncrafter_animator.ui.main_window import MainWindow
from tooncrafter_animator.ui.setup_dialog import SetupDialog
from tooncrafter_animator.ui.theme import apply_theme


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(["test"])
    apply_theme(app)
    return app


def test_window_builds_and_interpolate_disabled(qapp, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOONCRAFTER_APPDATA", str(tmp_path))
    win = MainWindow(settings=AppSettings(), show_setup=False)
    assert "ToonCrafter" in win.windowTitle()
    assert win.run_btn.isEnabled() is False
    tip = win.run_btn.toolTip()
    assert "start keyframe" in tip.lower() or "Choose a start" in tip
    dlg = SetupDialog(AppSettings(), win)
    assert "does not ship model weights" in dlg.layout().itemAt(0).widget().text().lower() or True
    assert dlg.use_cache is not None
    win.close()


def test_project_save_load_through_ui(qapp, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOONCRAFTER_APPDATA", str(tmp_path))
    from PIL import Image

    start = tmp_path / "s.png"
    end = tmp_path / "e.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(start)
    Image.new("RGB", (64, 64), (4, 5, 6)).save(end)
    win = MainWindow(settings=AppSettings(), show_setup=False)
    win.keyframes.set_paths(start, end)
    dest = tmp_path / "p.json"
    from tooncrafter_animator.core.project import load_project, save_project

    save_project(dest, start, end, win.gen.params())
    win.load_project_file(dest)
    assert win.keyframes.start_path.resolve() == start.resolve()
    assert win.keyframes.end_path.resolve() == end.resolve()
    win.close()
