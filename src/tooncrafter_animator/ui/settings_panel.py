from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tooncrafter_animator.core.checkpoints import discover_checkpoints, probe_checkpoint
from tooncrafter_animator.core.models import AspectMode, GenerationParams
from tooncrafter_animator.core.pipeline import plan_intermediates
from tooncrafter_animator.hardware import list_devices, precision_allowed, vram_warning
from tooncrafter_animator import copy as t


class SettingsPanel(QGroupBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(t.GROUP_GENERATION, parent)
        self._devices = list_devices()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setSingleStep(8)
        self.width_spin.setValue(512)
        self.width_spin.setMinimumWidth(96)
        self.width_spin.setAccessibleName(t.ACC_OUTPUT_WIDTH)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setSingleStep(8)
        self.height_spin.setValue(320)
        self.height_spin.setMinimumWidth(96)
        self.height_spin.setAccessibleName(t.ACC_OUTPUT_HEIGHT)

        self.aspect = QComboBox()
        self.aspect.addItem(t.ASPECT_PRESERVE, AspectMode.PRESERVE.value)
        self.aspect.addItem(t.ASPECT_CROP, AspectMode.CROP.value)
        self.aspect.setAccessibleName(t.ACC_ASPECT)
        self.aspect.setToolTip(t.TIP_ASPECT)

        self.intermediates = QSpinBox()
        self.intermediates.setRange(1, 60)
        self.intermediates.setValue(14)
        self.intermediates.setAccessibleName(t.ACC_INTERMEDIATES)
        self.intermediates.setToolTip(t.TIP_INTERMEDIATES)
        self.plan_label = QLabel()
        self.plan_label.setWordWrap(True)
        self.plan_label.setObjectName("hint")

        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(8)
        self.fps.setAccessibleName(t.ACC_FPS)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(123)
        self.seed.setAccessibleName(t.ACC_SEED)

        self.steps = QSpinBox()
        self.steps.setRange(1, 60)
        self.steps.setValue(50)
        self.steps.setAccessibleName(t.ACC_STEPS)
        self.steps.setToolTip(t.TIP_STEPS)

        self.cfg = QDoubleSpinBox()
        self.cfg.setRange(1.0, 15.0)
        self.cfg.setSingleStep(0.5)
        self.cfg.setValue(7.5)
        self.cfg.setAccessibleName(t.ACC_CFG)
        self.cfg.setToolTip(t.TIP_CFG)

        self.eta = QDoubleSpinBox()
        self.eta.setRange(0.0, 15.0)
        self.eta.setSingleStep(0.1)
        self.eta.setValue(1.0)
        self.eta.setAccessibleName(t.ACC_ETA)

        self.motion = QSpinBox()
        self.motion.setRange(5, 30)
        self.motion.setValue(10)
        self.motion.setAccessibleName(t.ACC_MOTION)
        self.motion.setToolTip(t.TIP_MOTION)

        self.gen_size = QComboBox()
        self.gen_size.addItem(t.GEN_TRAINED, (320, 512))
        self.gen_size.addItem(t.GEN_OFF_256, (256, 256))
        self.gen_size.addItem(t.GEN_OFF_320, (320, 320))
        self.gen_size.addItem(t.GEN_OFF_1024, (576, 1024))
        self.gen_size.setAccessibleName(t.ACC_GEN_SIZE)
        self.gen_size.setToolTip(t.TIP_GEN_SIZE)
        self.offdist = QLabel("")
        self.offdist.setWordWrap(True)
        self.offdist.setStyleSheet("color: #FFE082;")

        self.device = QComboBox()
        self.device.setAccessibleName(t.ACC_DEVICE)
        for info in self._devices:
            label = info.name if info.id == "cpu" else f"{info.name} ({info.id})"
            self.device.addItem(label, info.id)

        self.precision = QComboBox()
        self.precision.addItem(t.PRECISION_FP16, "fp16")
        self.precision.addItem(t.PRECISION_FP32, "fp32")
        self.precision.setAccessibleName(t.ACC_PRECISION)

        self.vram = QComboBox()
        self.vram.addItem(t.VRAM_NONE, "none")
        self.vram.addItem(t.VRAM_LOW, "low")
        self.vram.setAccessibleName(t.ACC_VRAM)
        self.vram.setToolTip(t.TIP_VRAM)

        self.hw_warning = QLabel("")
        self.hw_warning.setWordWrap(True)
        self.hw_warning.setStyleSheet("color: #FFE082;")

        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(t.PROMPT_PLACEHOLDER)
        self.prompt.setAccessibleName(t.ACC_PROMPT)
        self.prompt.setFixedHeight(64)

        self.ckpt_folder = QLineEdit()
        self.ckpt_folder.setReadOnly(True)
        self.ckpt_folder.setAccessibleName(t.ACC_CKPT_FOLDER)
        browse_folder = QPushButton(t.BTN_FOLDER)
        browse_folder.setAccessibleName(t.ACC_CHOOSE_CKPT_FOLDER)
        browse_folder.clicked.connect(self._browse_folder)

        self.ckpt = QComboBox()
        self.ckpt.setAccessibleName(t.ACC_CKPT_FILE)
        self.ckpt_status = QLabel(t.NO_CHECKPOINT)
        self.ckpt_status.setWordWrap(True)

        self.clip_path = QLineEdit()
        self.clip_path.setReadOnly(True)
        self.clip_path.setAccessibleName(t.ACC_OPENCLIP)
        browse_clip = QPushButton(t.BTN_CLIP)
        browse_clip.setAccessibleName(t.ACC_CHOOSE_OPENCLIP)
        browse_clip.clicked.connect(self._browse_clip)
        self.use_cache = QCheckBox(t.USE_OPENCLIP_CACHE)
        self.use_cache.setAccessibleName(t.ACC_USE_CACHE)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(t.LABEL_OUTPUT_WIDTH, self.width_spin)
        form.addRow(t.LABEL_OUTPUT_HEIGHT, self.height_spin)
        form.addRow(t.LABEL_ASPECT, self.aspect)
        form.addRow(t.LABEL_INTERMEDIATES, self.intermediates)
        form.addRow("", self.plan_label)
        form.addRow(t.LABEL_FPS, self.fps)
        form.addRow(t.LABEL_SEED, self.seed)
        form.addRow(t.LABEL_STEPS, self.steps)
        form.addRow(t.LABEL_GUIDANCE, self.cfg)
        form.addRow(t.LABEL_ETA, self.eta)
        form.addRow(t.LABEL_MOTION, self.motion)
        form.addRow(t.LABEL_GENERATE_AT, self.gen_size)
        form.addRow("", self.offdist)
        form.addRow(t.LABEL_DEVICE, self.device)
        form.addRow(t.LABEL_PRECISION, self.precision)
        form.addRow(t.LABEL_VRAM, self.vram)
        form.addRow("", self.hw_warning)
        form.addRow(t.LABEL_PROMPT, self.prompt)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.ckpt_folder, 1)
        ckpt_row.addWidget(browse_folder)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip_path, 1)
        clip_row.addWidget(browse_clip)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel(t.LABEL_CHECKPOINT_FOLDER))
        layout.addLayout(ckpt_row)
        layout.addWidget(self.ckpt)
        layout.addWidget(self.ckpt_status)
        layout.addWidget(QLabel(t.LABEL_OPENCLIP))
        layout.addLayout(clip_row)
        layout.addWidget(self.use_cache)

        for widget in (
            self.width_spin,
            self.height_spin,
            self.aspect,
            self.intermediates,
            self.fps,
            self.seed,
            self.steps,
            self.cfg,
            self.eta,
            self.motion,
            self.gen_size,
            self.device,
            self.precision,
            self.vram,
            self.use_cache,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(lambda *_: self._refresh_hints())
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(lambda *_: self._refresh_hints())
            if hasattr(widget, "toggled"):
                widget.toggled.connect(lambda *_: self.changed.emit())
        self.prompt.textChanged.connect(lambda: self.changed.emit())
        self.ckpt.currentIndexChanged.connect(self._on_ckpt)
        self._refresh_hints()
        cuda = next((d for d in self._devices if d.kind == "cuda"), None)
        if cuda:
            idx = self.device.findData(cuda.id)
            if idx >= 0:
                self.device.setCurrentIndex(idx)
            self.precision.setCurrentIndex(self.precision.findData("fp16"))

    def _refresh_hints(self) -> None:
        plan = plan_intermediates(self.intermediates.value())
        self.plan_label.setText(plan.description)
        gen = self.gen_size.currentData()
        if gen and gen != (320, 512):
            self.offdist.setText(t.OFFDIST_SIZE)
        else:
            self.offdist.setText("")
        device_id = self.device.currentData() or "cpu"
        precision = self.precision.currentData() or "fp32"
        ok, reason = precision_allowed(device_id, precision)
        info = next((d for d in self._devices if d.id == device_id), None)
        bits = []
        if not ok:
            bits.append(reason)
            # snap back to fp32 if fp16 illegal
            if precision == "fp16":
                self.precision.blockSignals(True)
                self.precision.setCurrentIndex(self.precision.findData("fp32"))
                self.precision.blockSignals(False)
                precision = "fp32"
        if info:
            warn = vram_warning(info, precision)
            if warn:
                bits.append(warn)
        self.hw_warning.setText(" ".join(bits))
        self.changed.emit()

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, t.TITLE_CHECKPOINT_FOLDER)
        if folder:
            self.set_checkpoint_folder(Path(folder))

    def _browse_clip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, t.TITLE_OPENCLIP, "", t.FILTER_OPENCLIP
        )
        if path:
            self.clip_path.setText(path)
            self.changed.emit()

    def set_checkpoint_folder(self, folder: Path) -> None:
        self.ckpt_folder.setText(str(folder))
        self.ckpt.blockSignals(True)
        self.ckpt.clear()
        for path in discover_checkpoints(folder):
            self.ckpt.addItem(path.name, str(path))
        self.ckpt.blockSignals(False)
        if self.ckpt.count() == 0:
            self.ckpt_status.setText(t.NO_CKPT_IN_FOLDER)
        else:
            self._on_ckpt()
        self.changed.emit()

    def _on_ckpt(self) -> None:
        path = self.ckpt.currentData()
        if not path:
            self.ckpt_status.setText(t.NO_CHECKPOINT)
            self.changed.emit()
            return
        probe = probe_checkpoint(Path(path))
        color = "#80CBC4" if probe.ok else "#FF8A80"
        self.ckpt_status.setText(probe.reason)
        self.ckpt_status.setStyleSheet(f"color: {color};")
        self.changed.emit()

    def apply_params(self, params: GenerationParams) -> None:
        self.width_spin.setValue(params.output_width)
        self.height_spin.setValue(params.output_height)
        idx = self.aspect.findData(params.aspect.value)
        if idx >= 0:
            self.aspect.setCurrentIndex(idx)
        self.intermediates.setValue(params.intermediates)
        self.fps.setValue(params.fps)
        self.seed.setValue(params.seed)
        self.steps.setValue(params.steps)
        self.cfg.setValue(params.cfg_scale)
        self.eta.setValue(params.eta)
        self.motion.setValue(params.motion_stride)
        self.prompt.setPlainText(params.prompt)
        gen_idx = self.gen_size.findData((params.gen_height, params.gen_width))
        if gen_idx >= 0:
            self.gen_size.setCurrentIndex(gen_idx)
        d_idx = self.device.findData(params.device)
        if d_idx >= 0:
            self.device.setCurrentIndex(d_idx)
        p_idx = self.precision.findData(params.precision)
        if p_idx >= 0:
            self.precision.setCurrentIndex(p_idx)
        v_idx = self.vram.findData(params.vram_strategy)
        if v_idx >= 0:
            self.vram.setCurrentIndex(v_idx)

    def params(self) -> GenerationParams:
        gen = self.gen_size.currentData() or (320, 512)
        return GenerationParams(
            output_width=self.width_spin.value(),
            output_height=self.height_spin.value(),
            aspect=AspectMode(self.aspect.currentData()),
            intermediates=self.intermediates.value(),
            fps=self.fps.value(),
            seed=self.seed.value(),
            steps=self.steps.value(),
            cfg_scale=self.cfg.value(),
            eta=self.eta.value(),
            motion_stride=self.motion.value(),
            precision=self.precision.currentData() or "fp32",
            device=self.device.currentData() or "cpu",
            prompt=self.prompt.toPlainText(),
            gen_height=int(gen[0]),
            gen_width=int(gen[1]),
            vram_strategy=self.vram.currentData() or "none",
        )

    def selected_checkpoint(self) -> Path | None:
        data = self.ckpt.currentData()
        return Path(data) if data else None

    def selected_clip(self) -> Path | None:
        text = self.clip_path.text().strip()
        return Path(text) if text else None
