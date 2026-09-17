from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tooncrafter_animator.core.models import ProgressEvent


class ProgressPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Progress", parent)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setAccessibleName("Interpolation progress")
        self.status = QLabel("Idle.")
        self.status.setObjectName("hint")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setAccessibleName("Cancel interpolation")
        self.cancel_btn.setToolTip("Cancel is only available while a job is running. The current DDIM step finishes first.")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setVisible(False)
        self.log.setAccessibleName("Log")
        self.toggle = QPushButton("Show logs")
        self.toggle.setCheckable(True)
        self.toggle.setAccessibleName("Toggle log pane")
        self.toggle.toggled.connect(self._toggle_log)

        top = QHBoxLayout()
        top.addWidget(self.bar, 1)
        top.addWidget(self.cancel_btn)
        top.addWidget(self.toggle)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.status)
        layout.addWidget(self.log)

    def _toggle_log(self, on: bool) -> None:
        self.log.setVisible(on)
        self.toggle.setText("Hide logs" if on else "Show logs")

    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def set_running(self, running: bool) -> None:
        self.cancel_btn.setEnabled(running)
        if running:
            self.status.setText("Running…")
        else:
            self.bar.setValue(0)

    def apply(self, event: ProgressEvent) -> None:
        self.bar.setValue(int(event.fraction * 1000))
        self.status.setText(
            f"Pass {event.pass_index} of {event.n_passes} — {event.message or f'step {event.step}/{event.n_steps}'}"
        )
        self.append_log(self.status.text())

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        self.append_log(text)
