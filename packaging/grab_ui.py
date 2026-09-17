"""Grab the real Qt windows to PNG. Run on a live X display or offscreen."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "xcb"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from tooncrafter_animator.settings import AppSettings  # noqa: E402
from tooncrafter_animator.ui.main_window import MainWindow  # noqa: E402
from tooncrafter_animator.ui.setup_dialog import SetupDialog  # noqa: E402
from tooncrafter_animator.ui.theme import apply_theme  # noqa: E402


def _keyframe(path: Path, color: tuple[int, int, int], label: str) -> None:
    img = Image.new("RGB", (512, 320), color)
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 40, 240, 200), fill=(255, 230, 80), outline=(20, 20, 20), width=6)
    draw.rectangle((300, 90, 460, 250), fill=(40, 180, 220), outline=(20, 20, 20), width=6)
    draw.text((16, 16), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def grab(widget, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pix = widget.grab()
    pix.save(str(dest), "PNG")
    print(dest, pix.width(), pix.height())


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tca-shots")
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/tca-demo-frames")
    start = tmp / "start.png"
    end = tmp / "end.png"
    _keyframe(start, (36, 48, 92), "START")
    _keyframe(end, (92, 36, 48), "END")

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)

    settings = AppSettings()
    win = MainWindow(settings=settings, show_setup=False)
    win.resize(1600, 920)
    win.show()
    app.processEvents()
    grab(win, out / "main_window_empty.png")

    win.keyframes.set_paths(start, end)
    app.processEvents()
    grab(win, out / "main_window_keyframes.png")

    dlg = SetupDialog(settings, win)
    dlg.show()
    app.processEvents()
    grab(dlg, out / "setup_dialog.png")
    dlg.close()
    win.close()


if __name__ == "__main__":
    main()
