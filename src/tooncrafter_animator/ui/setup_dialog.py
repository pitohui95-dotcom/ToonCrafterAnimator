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

CKPT_CARD = "https://huggingface.co/Doubiiu/ToonCrafter"
PRUNED_CARD = "https://huggingface.co/Kijai/DynamiCrafter_pruned"
CLIP_CARD = "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class SetupDialog(QDialog):
    """First-run checkpoint setup. Never downloads, never runs scripts."""

    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Checkpoint setup — ToonCrafter Animator")
        self.setModal(True)
        self.resize(640, 560)
        self.settings = settings

        blurb = QLabel(
            "<p><b>This app does not ship model weights and will not download them.</b></p>"
            "<p>You need two files you obtain yourself:</p>"
            "<ul>"
            "<li><code>model.ckpt</code> (~10 GB fp32, Doubiiu/ToonCrafter) or "
            "<code>tooncrafter_512_interp-fp16.safetensors</code> (~5 GB, Kijai/DynamiCrafter_pruned)</li>"
            "<li><code>open_clip_pytorch_model.bin</code> (~4 GB) — OpenCLIP ViT-H-14 / laion2b_s32b_b79k. "
            "Without it the text/image encoder tries to download weights, which this app refuses to do.</li>"
            "</ul>"
            "<p>Put the checkpoint in a folder you choose. The folder path is stored in this machine's "
            "app data. Nothing is uploaded. No scripts from the model folder are executed.</p>"
        )
        blurb.setWordWrap(True)
        blurb.setOpenExternalLinks(True)

        self.folder = QLineEdit(settings.checkpoint_folder)
        self.folder.setReadOnly(True)
        self.folder.setAccessibleName("Checkpoint folder")
        browse = QPushButton("Choose folder…")
        browse.clicked.connect(self._browse_folder)

        self.clip = QLineEdit(settings.clip_file)
        self.clip.setReadOnly(True)
        self.clip.setAccessibleName("OpenCLIP weights path")
        browse_clip = QPushButton("Choose CLIP file…")
        browse_clip.clicked.connect(self._browse_clip)

        self.use_cache = QCheckBox("I already cached OpenCLIP via open_clip (search local cache, never download)")
        self.use_cache.setChecked(settings.use_openclip_cache)
        self.use_cache.setAccessibleName("Use existing OpenCLIP cache")

        self.status = QLabel("Select the files, then Validate.")
        self.status.setWordWrap(True)

        open_ckpt = QPushButton("Open ToonCrafter model card")
        open_ckpt.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(CKPT_CARD)))
        open_pruned = QPushButton("Open fp16 pruned model card")
        open_pruned.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(PRUNED_CARD)))
        open_clip = QPushButton("Open OpenCLIP model card")
        open_clip.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(CLIP_CARD)))
        for btn in (open_ckpt, open_pruned, open_clip):
            btn.setToolTip("Opens your browser. This app still will not download anything.")

        validate = QPushButton("Validate")
        validate.setObjectName("primary")
        validate.clicked.connect(self._validate)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip, 1)
        clip_row.addWidget(browse_clip)
        form.addRow("Checkpoint folder", folder_row)
        form.addRow("OpenCLIP weights", clip_row)

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
        folder = QFileDialog.getExistingDirectory(self, "Checkpoint folder")
        if folder:
            self.folder.setText(folder)

    def _browse_clip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "OpenCLIP weights", "", "Weights (*.bin *.safetensors *.pt);;All files (*)"
        )
        if path:
            self.clip.setText(path)

    def _validate(self) -> bool:
        folder = Path(self.folder.text()) if self.folder.text() else None
        if folder is None or not folder.is_dir():
            self.status.setText("Choose a checkpoint folder.")
            return False
        ckpts = discover_checkpoints(folder)
        ok_probes = [probe_checkpoint(p) for p in ckpts]
        good = [p for p in ok_probes if p.ok]
        if not good:
            reasons = "; ".join(p.reason for p in ok_probes[:3]) or "no candidate files"
            self.status.setText(f"No valid ToonCrafter interpolation checkpoint in that folder. {reasons}")
            return False
        clip_ok = False
        if self.clip.text() and Path(self.clip.text()).is_file():
            clip_ok = True
        elif self.use_cache.isChecked() and find_openclip_cache() is not None:
            clip_ok = True
        if not clip_ok:
            self.status.setText(
                "OpenCLIP weights are missing. Choose open_clip_pytorch_model.bin, or tick the cache box "
                "only if that file is already on disk. The app will not download it."
            )
            return False
        names = ", ".join(p.path.name for p in good)
        self.status.setText(f"Ready. Valid checkpoints: {names}")
        self.settings.checkpoint_folder = str(folder)
        self.settings.checkpoint_file = str(good[0].path)
        self.settings.clip_file = self.clip.text()
        self.settings.use_openclip_cache = self.use_cache.isChecked()
        self.settings.accepted_setup = True
        return True

    def _save(self) -> None:
        if not self._validate():
            QMessageBox.warning(self, "Setup incomplete", self.status.text())
            return
        self.accept()
