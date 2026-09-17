from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tooncrafter_animator.ui.widgets.drop_label import IMAGE_FILTER, DropPixmap


class KeyframePanel(QGroupBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Keyframes", parent)
        self.start_path: Path | None = None
        self.end_path: Path | None = None

        self.start_view = DropPixmap("Drop a start keyframe\nor click to browse")
        self.end_view = DropPixmap("Drop an end keyframe\nor click to browse")
        self.start_view.setAccessibleName("Start keyframe preview")
        self.end_view.setAccessibleName("End keyframe preview")
        self.start_view.clicked.connect(lambda: self._browse("start"))
        self.end_view.clicked.connect(lambda: self._browse("end"))
        self.start_view.files_dropped.connect(lambda files: self._set("start", files[0]))
        self.end_view.files_dropped.connect(lambda files: self._set("end", files[0]))

        pick_start = QPushButton("Choose start…")
        pick_start.setAccessibleName("Choose start keyframe")
        pick_start.clicked.connect(lambda: self._browse("start"))
        pick_end = QPushButton("Choose end…")
        pick_end.setAccessibleName("Choose end keyframe")
        pick_end.clicked.connect(lambda: self._browse("end"))

        swap = QPushButton("Swap")
        swap.setAccessibleName("Swap start and end keyframes")
        swap.setToolTip("Swap the start and end keyframes.")
        swap.clicked.connect(self.swap)
        clear = QPushButton("Clear")
        clear.setAccessibleName("Clear keyframes")
        clear.clicked.connect(self.clear)

        self.note = QLabel(
            "Generation always runs at 320×512 (the 512-interp checkpoint). "
            "Your output width/height is applied afterwards."
        )
        self.note.setWordWrap(True)
        self.note.setObjectName("hint")
        self.note.setProperty("class", "hint")

        grid = QGridLayout()
        grid.addWidget(QLabel("Start"), 0, 0)
        grid.addWidget(QLabel("End"), 0, 1)
        grid.addWidget(self.start_view, 1, 0)
        grid.addWidget(self.end_view, 1, 1)
        grid.addWidget(pick_start, 2, 0)
        grid.addWidget(pick_end, 2, 1)

        buttons = QHBoxLayout()
        buttons.addWidget(swap)
        buttons.addWidget(clear)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addLayout(buttons)
        layout.addWidget(self.note)

    def _browse(self, which: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose keyframe", "", IMAGE_FILTER)
        if path:
            self._set(which, Path(path))

    def _set(self, which: str, path: Path) -> None:
        path = Path(path)
        if which == "start":
            self.start_path = path
            self.start_view.set_image(path)
        else:
            self.end_path = path
            self.end_view.set_image(path)
        self.changed.emit()

    def set_paths(self, start: Path | None, end: Path | None) -> None:
        self.start_path = start
        self.end_path = end
        self.start_view.set_image(start)
        self.end_view.set_image(end)
        self.changed.emit()

    def swap(self) -> None:
        self.start_path, self.end_path = self.end_path, self.start_path
        self.start_view.set_image(self.start_path)
        self.end_view.set_image(self.end_path)
        self.changed.emit()

    def clear(self) -> None:
        self.set_paths(None, None)
