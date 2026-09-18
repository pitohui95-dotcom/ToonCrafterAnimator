from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from tooncrafter_animator.core.checkpoints import discover_checkpoints, probe_checkpoint
from tooncrafter_animator.inference.adapter import find_openclip_cache
from tooncrafter_animator.settings import AppSettings
from tooncrafter_animator import copy as t

CKPT_CARD = "https://huggingface.co/Doubiiu/ToonCrafter"
PRUNED_CARD = "https://huggingface.co/Kijai/DynamiCrafter_pruned"
CLIP_CARD = "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class SetupDialog(QDialog):
    """First-run checkpoint setup. Never downloads, never runs scripts."""

    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t.TITLE_SETUP)
        self.setModal(True)
        self.resize(640, 560)
        self.settings = settings

        blurb = QLabel(t.SETUP_BLURB)
        blurb.setWordWrap(True)
        blurb.setOpenExternalLinks(True)

        self.folder = QLineEdit(settings.checkpoint_folder)
        self.folder.setReadOnly(True)
        self.folder.setAccessibleName(t.ACC_CKPT_FOLDER)
        browse = QPushButton(t.BTN_CHOOSE_FOLDER)
        browse.clicked.connect(self._browse_folder)

        self.clip = QLineEdit(settings.clip_file)
        self.clip.setReadOnly(True)
        self.clip.setAccessibleName(t.ACC_OPENCLIP)
        browse_clip = QPushButton(t.BTN_CHOOSE_CLIP)
        browse_clip.clicked.connect(self._browse_clip)

        self.use_cache = QCheckBox(t.SETUP_USE_CACHE)
        self.use_cache.setChecked(settings.use_openclip_cache)
        self.use_cache.setAccessibleName(t.ACC_USE_CACHE)

        self.status = QLabel(t.SETUP_STATUS_INITIAL)
        self.status.setWordWrap(True)

        open_ckpt = QPushButton(t.BTN_OPEN_CKPT_CARD)
        open_ckpt.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(CKPT_CARD)))
        open_pruned = QPushButton(t.BTN_OPEN_PRUNED_CARD)
        open_pruned.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(PRUNED_CARD)))
        open_clip = QPushButton(t.BTN_OPEN_CLIP_CARD)
        open_clip.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(CLIP_CARD)))
        for btn in (open_ckpt, open_pruned, open_clip):
            btn.setToolTip(t.TIP_MODEL_CARD)

        validate = QPushButton(t.BTN_VALIDATE)
        validate.setObjectName("primary")
        validate.clicked.connect(self._validate)

        buttons = QDialogButtonBox()
        save_btn = buttons.addButton(t.BTN_SAVE, QDialogButtonBox.AcceptRole)
        cancel_btn = buttons.addButton(t.BTN_DIALOG_CANCEL, QDialogButtonBox.RejectRole)
        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)

        form = QFormLayout()
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip, 1)
        clip_row.addWidget(browse_clip)
        form.addRow(t.LABEL_CHECKPOINT_FOLDER, folder_row)
        form.addRow(t.LABEL_OPENCLIP_SHORT, clip_row)

        links = QHBoxLayout()
        links.addWidget(open_ckpt)
        links.addWidget(open_pruned)
        links.addWidget(open_clip)

        layout = QVBoxLayout(self)
        layout.addWidget(blurb)
        layout.addLayout(form)
        layout.addWidget(self.use_cache)
        layout.addLayout(links)
        layout.addWidget(validate)
        layout.addWidget(self.status)
        layout.addStretch(1)
        layout.addWidget(buttons)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, t.TITLE_CHECKPOINT_FOLDER)
        if folder:
            self.folder.setText(folder)

    def _browse_clip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, t.TITLE_OPENCLIP, "", t.FILTER_WEIGHTS
        )
        if path:
            self.clip.setText(path)

    def _validate(self) -> bool:
        folder = Path(self.folder.text()) if self.folder.text() else None
        if folder is None or not folder.is_dir():
            self.status.setText(t.STATUS_CHOOSE_FOLDER)
            return False
        ckpts = discover_checkpoints(folder)
        ok_probes = [probe_checkpoint(p) for p in ckpts]
        good = [p for p in ok_probes if p.ok]
        if not good:
            reasons = "; ".join(p.reason for p in ok_probes[:3]) or t.STATUS_NO_CANDIDATES
            self.status.setText(t.STATUS_NO_VALID_CKPT.format(reasons=reasons))
            return False
        clip_ok = False
        if self.clip.text() and Path(self.clip.text()).is_file():
            clip_ok = True
        elif self.use_cache.isChecked() and find_openclip_cache() is not None:
            clip_ok = True
        if not clip_ok:
            self.status.setText(t.STATUS_CLIP_MISSING)
            return False
        names = ", ".join(p.path.name for p in good)
        self.status.setText(t.STATUS_READY.format(names=names))
        self.settings.checkpoint_folder = str(folder)
        self.settings.checkpoint_file = str(good[0].path)
        self.settings.clip_file = self.clip.text()
        self.settings.use_openclip_cache = self.use_cache.isChecked()
        self.settings.accepted_setup = True
        return True

    def _save(self) -> None:
        if not self._validate():
            QMessageBox.warning(self, t.TITLE_SETUP_INCOMPLETE, self.status.text())
            return
        self.accept()
