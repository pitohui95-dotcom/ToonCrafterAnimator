from __future__ import annotations

import os

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
    assert win.run_btn.text() == "插帧"
    assert win.run_btn.isEnabled() is False
    tip = win.run_btn.toolTip()
    assert "起始" in tip
    dlg = SetupDialog(AppSettings(), win)
    blurb = dlg.layout().itemAt(0).widget().text()
    assert "不附带模型权重" in blurb
    assert win.menuBar().actions()[0].text() == "文件(&F)"
    assert win.preview.view.text() == "完成插帧后，帧预览会显示在这里。"
    assert win.progress.status.text().startswith("空闲")
    win.close()


def test_settings_default_to_cuda_fp16_when_gpu_listed(qapp, tmp_path, monkeypatch) -> None:
    from tooncrafter_animator.hardware import DeviceInfo

    fake = [
        DeviceInfo(id="cpu", name="CPU", kind="cpu", warning="slow"),
        DeviceInfo(
            id="cuda:0",
            name="NVIDIA GeForce RTX Test",
            kind="cuda",
            total_vram_bytes=24 * 1024**3,
            capability="8.9",
        ),
    ]
    monkeypatch.setenv("TOONCRAFTER_APPDATA", str(tmp_path))
    monkeypatch.setattr("tooncrafter_animator.ui.settings_panel.list_devices", lambda: fake)
    win = MainWindow(settings=AppSettings(), show_setup=False)
    assert win.gen.device.currentData() == "cuda:0"
    assert win.gen.precision.currentData() == "fp16"
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
