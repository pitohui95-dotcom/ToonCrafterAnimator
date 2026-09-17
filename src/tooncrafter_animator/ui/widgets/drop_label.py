from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"


class DropPixmap(QLabel):
    """Clickable / droppable keyframe preview."""

    files_dropped = Signal(list)
    clicked = Signal()

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setWordWrap(True)
        self._placeholder = placeholder
        self._path: Path | None = None
        self.setText(placeholder)
        self.setAccessibleName(placeholder)
        self.setStyleSheet(
            "QLabel { background: #262626; border: 1px dashed #5A5A5A; border-radius: 6px; color: #C8C8C8; padding: 8px; }"
        )
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        images = [p for p in paths if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
        if images:
            self.files_dropped.emit(images)
            event.acceptProposedAction()

    def set_image(self, path: Path | None) -> None:
        self._path = path
        if path is None or not path.is_file():
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self.setText(f"Could not read\n{path.name}")
            return
        self.setText("")
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._path:
            self.set_image(self._path)
