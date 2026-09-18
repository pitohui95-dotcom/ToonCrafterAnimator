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
from tooncrafter_animator import copy as t


class ProgressPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(t.GROUP_PROGRESS, parent)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setAccessibleName(t.ACC_PROGRESS)
        self.status = QLabel(t.IDLE)
        self.status.setObjectName("hint")
        self.cancel_btn = QPushButton(t.BTN_CANCEL)
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setAccessibleName(t.ACC_CANCEL)
        self.cancel_btn.setToolTip(t.TIP_CANCEL)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setVisible(False)
        self.log.setAccessibleName(t.ACC_LOG)
        self.toggle = QPushButton(t.BTN_SHOW_LOGS)
        self.toggle.setCheckable(True)
        self.toggle.setAccessibleName(t.ACC_TOGGLE_LOG)
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
        self.toggle.setText(t.BTN_HIDE_LOGS if on else t.BTN_SHOW_LOGS)

    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def set_running(self, running: bool) -> None:
        self.cancel_btn.setEnabled(running)
        if running:
            self.status.setText(t.RUNNING)
        else:
            self.bar.setValue(0)

    def apply(self, event: ProgressEvent) -> None:
        self.bar.setValue(int(event.fraction * 1000))
        self.status.setText(
            t.STATUS_PASS.format(
                pass_index=event.pass_index,
                n_passes=event.n_passes,
                message=event.message or t.STATUS_STEP.format(step=event.step, n_steps=event.n_steps),
            )
        )
        self.append_log(self.status.text())

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        self.append_log(text)
