from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


from tooncrafter_animator import copy as t


def _array_to_pixmap(frame: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    image = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class PreviewPanel(QGroupBox):
    export_png = Signal()
    export_mp4 = Signal()
    export_gif = Signal()
    open_folder = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(t.GROUP_PREVIEW, parent)
        self.frames: list[np.ndarray] = []
        self.index = 0
        self.playing = False
        self.fps = 8
        self.last_export_dir: Path | None = None

        self.view = QLabel(t.PREVIEW_EMPTY)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(220)
        self.view.setAccessibleName(t.ACC_PREVIEW)
        self.view.setStyleSheet("QLabel { background: #262626; border: 1px solid #3A3A3A; border-radius: 6px; }")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setAccessibleName(t.ACC_TIMELINE)
        self.slider.valueChanged.connect(self._scrub)

        self.frame_label = QLabel(t.NO_FRAMES)
        self.frame_label.setObjectName("hint")

        self.play_btn = QPushButton(t.BTN_PLAY)
        self.play_btn.setAccessibleName(t.ACC_PLAY)
        self.play_btn.clicked.connect(self.toggle_play)
        self.loop = QPushButton(t.BTN_LOOP_ON)
        self.loop.setCheckable(True)
        self.loop.setChecked(True)
        self.loop.setAccessibleName(t.ACC_LOOP)
        self.loop.clicked.connect(
            lambda: self.loop.setText(t.BTN_LOOP_ON if self.loop.isChecked() else t.BTN_LOOP_OFF)
        )

        self.png_btn = QPushButton(t.BTN_EXPORT_PNG)
        self.mp4_btn = QPushButton(t.BTN_EXPORT_MP4)
        self.gif_btn = QPushButton(t.BTN_EXPORT_GIF)
        self.folder_btn = QPushButton(t.BTN_OPEN_FOLDER)
        for btn, sig in (
            (self.png_btn, self.export_png),
            (self.mp4_btn, self.export_mp4),
            (self.gif_btn, self.export_gif),
            (self.folder_btn, self.open_folder),
        ):
            btn.clicked.connect(sig)
        self._set_export_enabled(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        controls = QHBoxLayout()
        controls.addWidget(self.play_btn)
        controls.addWidget(self.loop)
        controls.addWidget(self.frame_label, 1)

        exports = QHBoxLayout()
        exports.addWidget(self.png_btn)
        exports.addWidget(self.mp4_btn)
        exports.addWidget(self.gif_btn)
        exports.addWidget(self.folder_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.slider)
        layout.addLayout(controls)
        layout.addLayout(exports)

    def set_frames(self, frames: list[np.ndarray], fps: int) -> None:
        self.frames = list(frames)
        self.fps = max(1, fps)
        self.index = 0
        n = len(self.frames)
        self.slider.blockSignals(True)
        self.slider.setEnabled(n > 0)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_export_enabled(n > 0)
        self._show(0)
        self.stop()

    def clear(self) -> None:
        self.set_frames([], self.fps)
        self.view.setText(t.PREVIEW_EMPTY)

    def _set_export_enabled(self, on: bool) -> None:
        for btn in (self.png_btn, self.mp4_btn, self.gif_btn):
            btn.setEnabled(on)
            btn.setToolTip("" if on else t.TIP_EXPORT_NEED_JOB)
        self.folder_btn.setEnabled(self.last_export_dir is not None)
        if self.last_export_dir is None:
            self.folder_btn.setToolTip(t.TIP_EXPORT_NEED_FOLDER)
        else:
            self.folder_btn.setToolTip(str(self.last_export_dir))

    def _show(self, index: int) -> None:
        if not self.frames:
            self.frame_label.setText(t.NO_FRAMES)
            return
        self.index = max(0, min(index, len(self.frames) - 1))
        pix = _array_to_pixmap(self.frames[self.index])
        self.view.setPixmap(pix.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.frame_label.setText(f"第 {self.index + 1} / {len(self.frames)} 帧")
        self.slider.blockSignals(True)
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)

    def _scrub(self, value: int) -> None:
        self._show(value)

    def toggle_play(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.play()

    def play(self) -> None:
        if not self.frames:
            return
        self.playing = True
        self.play_btn.setText(t.BTN_PAUSE)
        self.timer.start(int(1000 / self.fps))

    def stop(self) -> None:
        self.playing = False
        self.play_btn.setText(t.BTN_PLAY)
        self.timer.stop()

    def _tick(self) -> None:
        if not self.frames:
            self.stop()
            return
        nxt = self.index + 1
        if nxt >= len(self.frames):
            if self.loop.isChecked():
                nxt = 0
            else:
                self.stop()
                return
        self._show(nxt)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.frames:
            self._show(self.index)
