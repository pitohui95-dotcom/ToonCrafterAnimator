from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tooncrafter_animator import APP_DISPLAY_NAME, __version__
from tooncrafter_animator.core.checkpoints import probe_checkpoint
from tooncrafter_animator.core.export import export_gif, export_mp4, export_png_sequence
from tooncrafter_animator.core.ffmpeg import resolve_ffmpeg
from tooncrafter_animator.core.models import LoadPlan, ProgressEvent
from tooncrafter_animator.core.pipeline import plan_intermediates
from tooncrafter_animator.core.project import load_project, save_project
from tooncrafter_animator.core.worker import InferenceWorker, WorkerHost
from tooncrafter_animator.hardware import torch_available
from tooncrafter_animator.logging_setup import setup_logging
from tooncrafter_animator.paths import asset_path
from tooncrafter_animator.settings import AppSettings, load_settings, save_settings
from tooncrafter_animator.ui.keyframe_panel import KeyframePanel
from tooncrafter_animator.ui.preview_panel import PreviewPanel
from tooncrafter_animator.ui.progress_panel import ProgressPanel
from tooncrafter_animator.ui.settings_panel import SettingsPanel
from tooncrafter_animator.ui.setup_dialog import SetupDialog
from tooncrafter_animator.ui.theme import apply_theme

log = logging.getLogger("tooncrafter")


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings | None = None, show_setup: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_DISPLAY_NAME} {__version__}")
        icon = asset_path("icon.png")
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.settings = settings or load_settings()
        self.worker_host = WorkerHost()
        self.project_path: Path | None = None
        self.result_frames: list[np.ndarray] = []

        self.keyframes = KeyframePanel()
        self.gen = SettingsPanel()
        self.preview = PreviewPanel()
        self.progress = ProgressPanel()
        self.progress.cancel_btn.clicked.connect(self.cancel_job)

        if self.settings.checkpoint_folder:
            self.gen.set_checkpoint_folder(Path(self.settings.checkpoint_folder))
        if self.settings.clip_file:
            self.gen.clip_path.setText(self.settings.clip_file)
        self.gen.use_cache.setChecked(self.settings.use_openclip_cache)

        from PySide6.QtWidgets import QPushButton

        self.run_btn = QPushButton("Interpolate")
        self.run_btn.setObjectName("primary")
        self.run_btn.setAccessibleName("Interpolate")
        self.run_btn.clicked.connect(self.start_job)
        self.setup_btn = QPushButton("Checkpoint setup…")
        self.setup_btn.setAccessibleName("Open checkpoint setup")
        self.setup_btn.clicked.connect(self.open_setup)

        actions = QHBoxLayout()
        actions.addWidget(self.run_btn, 1)
        actions.addWidget(self.setup_btn)

        left = QVBoxLayout()
        left.addWidget(self.keyframes, 1)
        left.addLayout(actions)
        left_w = QWidget()
        left_w.setLayout(left)

        mid_scroll = QScrollArea()
        mid_scroll.setWidgetResizable(True)
        mid_scroll.setWidget(self.gen)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(mid_scroll)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)

        root = QVBoxLayout()
        root.addWidget(splitter, 1)
        root.addWidget(self.progress)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1440, 900)

        self.keyframes.changed.connect(self._refresh_actions)
        self.gen.changed.connect(self._refresh_actions)
        self.preview.export_png.connect(self._export_png)
        self.preview.export_mp4.connect(self._export_mp4)
        self.preview.export_gif.connect(self._export_gif)
        self.preview.open_folder.connect(self._open_folder)

        self._build_menu()
        self._refresh_actions()
        if show_setup and not self._setup_complete():
            # Shown by run_app after the window is visible so tests can skip it.
            self._needs_setup = True
        else:
            self._needs_setup = False

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_p = QAction("Open project…", self)
        open_p.setShortcut(QKeySequence.Open)
        open_p.triggered.connect(self.open_project)
        save_p = QAction("Save project…", self)
        save_p.setShortcut(QKeySequence.Save)
        save_p.triggered.connect(self.save_project)
        setup = QAction("Checkpoint setup…", self)
        setup.triggered.connect(self.open_setup)
        quit_a = QAction("Quit", self)
        quit_a.setShortcut(QKeySequence.Quit)
        quit_a.triggered.connect(self.close)
        file_menu.addAction(open_p)
        file_menu.addAction(save_p)
        file_menu.addSeparator()
        file_menu.addAction(setup)
        file_menu.addSeparator()
        file_menu.addAction(quit_a)

        run_menu = self.menuBar().addMenu("&Run")
        go = QAction("Interpolate", self)
        go.setShortcut(QKeySequence("Ctrl+Return"))
        go.triggered.connect(self.start_job)
        cancel = QAction("Cancel", self)
        cancel.setShortcut(QKeySequence("Escape"))
        cancel.triggered.connect(self.cancel_job)
        run_menu.addAction(go)
        run_menu.addAction(cancel)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _setup_complete(self) -> bool:
        folder = self.settings.checkpoint_folder
        if not folder or not Path(folder).is_dir():
            return False
        ckpt = self.gen.selected_checkpoint()
        if ckpt is None or not probe_checkpoint(ckpt).ok:
            return False
        clip = self.gen.selected_clip()
        if clip and clip.is_file():
            return True
        return bool(self.gen.use_cache.isChecked())

    def maybe_show_setup(self) -> None:
        if self._needs_setup:
            self.open_setup()

    def open_setup(self) -> None:
        dlg = SetupDialog(self.settings, self)
        if dlg.exec():
            save_settings(self.settings)
            if self.settings.checkpoint_folder:
                self.gen.set_checkpoint_folder(Path(self.settings.checkpoint_folder))
            self.gen.clip_path.setText(self.settings.clip_file)
            self.gen.use_cache.setChecked(self.settings.use_openclip_cache)
            self._refresh_actions()

    def _blockers(self) -> list[str]:
        reasons: list[str] = []
        if self.worker_host.is_running():
            reasons.append("A job is already running.")
        if self.keyframes.start_path is None or not self.keyframes.start_path.is_file():
            reasons.append("Choose a start keyframe.")
        if self.keyframes.end_path is None or not self.keyframes.end_path.is_file():
            reasons.append("Choose an end keyframe.")
        if not torch_available():
            reasons.append("PyTorch is not installed, so real ToonCrafter inference cannot run.")
        ckpt = self.gen.selected_checkpoint()
        if ckpt is None:
            reasons.append("Select a validated ToonCrafter checkpoint.")
        else:
            probe = probe_checkpoint(ckpt)
            if not probe.ok:
                reasons.append(probe.reason)
        clip = self.gen.selected_clip()
        if not (clip and clip.is_file()) and not self.gen.use_cache.isChecked():
            reasons.append("Select OpenCLIP weights, or enable the local-cache option after placing them on disk.")
        params = self.gen.params()
        if params.precision == "fp16" and not str(params.device).startswith("cuda"):
            reasons.append("FP16 is only available on CUDA.")
        return reasons

    def _refresh_actions(self) -> None:
        reasons = self._blockers()
        self.run_btn.setEnabled(not reasons)
        self.run_btn.setToolTip("\n".join(reasons) if reasons else "Run one or more real ToonCrafter passes.")
        plan = plan_intermediates(self.gen.params().intermediates)
        if not self.worker_host.is_running():
            self.progress.set_status("Idle. " + plan.description)

    def start_job(self) -> None:
        reasons = self._blockers()
        if reasons:
            QMessageBox.warning(self, "Cannot interpolate", "\n".join(reasons))
            return
        params = self.gen.params()
        ckpt = self.gen.selected_checkpoint()
        assert ckpt is not None
        load = LoadPlan(
            checkpoint=ckpt,
            clip_weights=self.gen.selected_clip(),
            use_openclip_cache=self.gen.use_cache.isChecked(),
            device=params.device,
            precision=params.precision,
        )
        worker = InferenceWorker(self.keyframes.start_path, self.keyframes.end_path, params, load)
        worker.progressed.connect(self._on_progress)
        worker.staged.connect(self.progress.set_status)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self.worker_host.start(worker)
        self.progress.set_running(True)
        self._refresh_actions()
        self._persist_settings()

    def cancel_job(self) -> None:
        if self.worker_host.is_running():
            self.progress.set_status("Cancelling after this DDIM step…")
            self.worker_host.cancel()

    def _on_progress(self, event: ProgressEvent) -> None:
        self.progress.apply(event)

    def _on_finished(self, frames: object) -> None:
        self.result_frames = list(frames)  # type: ignore[arg-type]
        self.preview.set_frames(self.result_frames, self.gen.params().fps)
        self.progress.set_running(False)
        self.progress.set_status(f"Done. {len(self.result_frames)} frames.")
        self._refresh_actions()

    def _on_failed(self, message: str) -> None:
        self.progress.set_running(False)
        self.progress.set_status("Failed.")
        QMessageBox.critical(self, "Interpolation failed", message)
        self._refresh_actions()

    def _on_cancelled(self) -> None:
        self.progress.set_running(False)
        self.progress.set_status("Cancelled.")
        self._refresh_actions()

    def _persist_settings(self) -> None:
        params = self.gen.params()
        self.settings.checkpoint_folder = self.gen.ckpt_folder.text()
        ckpt = self.gen.selected_checkpoint()
        self.settings.checkpoint_file = str(ckpt) if ckpt else ""
        clip = self.gen.selected_clip()
        self.settings.clip_file = str(clip) if clip else ""
        self.settings.use_openclip_cache = self.gen.use_cache.isChecked()
        self.settings.device = params.device
        self.settings.precision = params.precision
        self.settings.last_output_width = params.output_width
        self.settings.last_output_height = params.output_height
        self.settings.vram_strategy = params.vram_strategy
        save_settings(self.settings)

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "", "ToonCrafter project (*.json)")
        if not path:
            return
        dest = Path(path)
        save_project(
            dest,
            self.keyframes.start_path,
            self.keyframes.end_path,
            self.gen.params(),
            export_dir=self.settings.last_export_dir,
            checkpoint_folder=self.gen.ckpt_folder.text(),
            checkpoint_file=str(self.gen.selected_checkpoint() or ""),
            clip_file=self.gen.clip_path.text(),
        )
        self.project_path = dest
        self.statusBar().showMessage(f"Saved {dest}", 4000)

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "ToonCrafter project (*.json)")
        if not path:
            return
        self.load_project_file(Path(path))

    def load_project_file(self, path: Path) -> None:
        project = load_project(path)
        start = project.start.resolved(path.parent)
        end = project.end.resolved(path.parent)
        self.keyframes.set_paths(start, end)
        self.gen.apply_params(project.generation)
        if project.checkpoint_folder:
            self.gen.set_checkpoint_folder(Path(project.checkpoint_folder))
        if project.clip_file:
            self.gen.clip_path.setText(project.clip_file)
        self.project_path = path
        if project.missing_keyframes:
            QMessageBox.warning(
                self,
                "Missing keyframes",
                "This project refers to keyframe files that are not on disk: "
                + ", ".join(project.missing_keyframes)
                + ". Paths were kept; choose replacements before interpolating.",
            )
        self._refresh_actions()

    def _ask_export(self, title: str, filt: str) -> Path | None:
        start = self.settings.last_export_dir or ""
        path, _ = QFileDialog.getSaveFileName(self, title, start, filt)
        if not path:
            return None
        dest = Path(path)
        self.settings.last_export_dir = str(dest.parent)
        self.preview.last_export_dir = dest.parent
        save_settings(self.settings)
        self.preview._set_export_enabled(bool(self.result_frames))
        return dest

    def _export_png(self) -> None:
        if not self.result_frames:
            return
        folder = QFileDialog.getExistingDirectory(self, "PNG sequence folder", self.settings.last_export_dir or "")
        if not folder:
            return
        dest = Path(folder)
        export_png_sequence(self.result_frames, dest)
        self.settings.last_export_dir = str(dest)
        self.preview.last_export_dir = dest
        save_settings(self.settings)
        self.preview._set_export_enabled(True)
        self.statusBar().showMessage(f"Wrote PNG sequence to {dest}", 4000)

    def _export_mp4(self) -> None:
        dest = self._ask_export("Export MP4", "MP4 (*.mp4)")
        if dest is None:
            return
        try:
            export_mp4(self.result_frames, dest, self.gen.params().fps, resolve_ffmpeg(self.settings.ffmpeg_path))
        except Exception as exc:
            QMessageBox.critical(self, "MP4 export failed", str(exc))
            return
        self.statusBar().showMessage(f"Wrote {dest}", 4000)

    def _export_gif(self) -> None:
        dest = self._ask_export("Export GIF", "GIF (*.gif)")
        if dest is None:
            return
        try:
            export_gif(self.result_frames, dest, self.gen.params().fps, resolve_ffmpeg(self.settings.ffmpeg_path))
        except Exception as exc:
            QMessageBox.critical(self, "GIF export failed", str(exc))
            return
        self.statusBar().showMessage(f"Wrote {dest}", 4000)

    def _open_folder(self) -> None:
        folder = self.preview.last_export_dir
        if folder is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"<h3>{APP_DISPLAY_NAME}</h3>"
            f"<p>Version {__version__}. Real ToonCrafter 512-interp, no substitute interpolator.</p>"
            "<p>Vendored from AIGODLIKE/ComfyUI-ToonCrafter and ToonCrafter/ToonCrafter (Apache-2.0). "
            "Weights are not included.</p>",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker_host.stop()
        self._persist_settings()
        super().closeEvent(event)


def run_app(project_path: str | None = None) -> int:
    setup_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ToonCrafterAnimator")
    app.setOrganizationName("ToonCrafterAnimator")
    apply_theme(app)
    settings = load_settings()
    window = MainWindow(settings=settings, show_setup=True)
    window.show()
    if project_path:
        window.load_project_file(Path(project_path))
    window.maybe_show_setup()
    return app.exec()
