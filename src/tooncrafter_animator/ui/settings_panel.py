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


class SettingsPanel(QGroupBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Generation", parent)
        self._devices = list_devices()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setSingleStep(8)
        self.width_spin.setValue(512)
        self.width_spin.setAccessibleName("Output width")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setSingleStep(8)
        self.height_spin.setValue(320)
        self.height_spin.setAccessibleName("Output height")

        self.aspect = QComboBox()
        self.aspect.addItem("Preserve (fit, pad)", AspectMode.PRESERVE.value)
        self.aspect.addItem("Crop (cover, centre-crop)", AspectMode.CROP.value)
        self.aspect.setAccessibleName("Aspect mode")
        self.aspect.setToolTip(
            "Preserve fits the 320×512 generation inside your W×H and pads with the keyframe border colour. "
            "Crop covers W×H and centre-crops the overflow."
        )

        self.intermediates = QSpinBox()
        self.intermediates.setRange(1, 60)
        self.intermediates.setValue(14)
        self.intermediates.setAccessibleName("Intermediate frame count")
        self.intermediates.setToolTip(
            "Number of in-between frames to export. One ToonCrafter pass produces 14 real in-betweens. "
            "Asking for more runs additional real passes between generated anchors — never duplicates or blends."
        )
        self.plan_label = QLabel()
        self.plan_label.setWordWrap(True)
        self.plan_label.setObjectName("hint")

        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(8)
        self.fps.setAccessibleName("Playback FPS")

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(123)
        self.seed.setAccessibleName("Seed")

        self.steps = QSpinBox()
        self.steps.setRange(1, 60)
        self.steps.setValue(50)
        self.steps.setAccessibleName("DDIM steps")
        self.steps.setToolTip("DDIM denoising steps. Upstream default is 50; the node caps at 60.")

        self.cfg = QDoubleSpinBox()
        self.cfg.setRange(1.0, 15.0)
        self.cfg.setSingleStep(0.5)
        self.cfg.setValue(7.5)
        self.cfg.setAccessibleName("Classifier-free guidance scale")
        self.cfg.setToolTip(
            "unconditional_guidance_scale. The 512-interp config uses uncond_type=empty_seq; default 7.5."
        )

        self.eta = QDoubleSpinBox()
        self.eta.setRange(0.0, 15.0)
        self.eta.setSingleStep(0.1)
        self.eta.setValue(1.0)
        self.eta.setAccessibleName("DDIM eta")

        self.motion = QSpinBox()
        self.motion.setRange(5, 30)
        self.motion.setValue(10)
        self.motion.setAccessibleName("Motion frame stride")
        self.motion.setToolTip(
            "This is the model's FPS / frame-stride condition (the ComfyUI node mislabels it as frame_count). "
            "Recommended 5–30. Smaller values produce larger motion."
        )

        self.gen_size = QComboBox()
        self.gen_size.addItem("320×512 (trained)", (320, 512))
        self.gen_size.addItem("256×256 (off-distribution)", (256, 256))
        self.gen_size.addItem("320×320 (off-distribution)", (320, 320))
        self.gen_size.addItem("576×1024 (off-distribution, 1024-class)", (576, 1024))
        self.gen_size.setAccessibleName("Generation resolution")
        self.gen_size.setToolTip(
            "The UNet is convolutional and will run at other 64-divisible sizes, but it was trained at 320×512. "
            "Other sizes are flagged off-distribution."
        )
        self.offdist = QLabel("")
        self.offdist.setWordWrap(True)
        self.offdist.setStyleSheet("color: #FFE082;")

        self.device = QComboBox()
        self.device.setAccessibleName("Device")
        for info in self._devices:
            label = info.name if info.id == "cpu" else f"{info.name} ({info.id})"
            self.device.addItem(label, info.id)

        self.precision = QComboBox()
        self.precision.addItem("FP16 (CUDA)", "fp16")
        self.precision.addItem("FP32", "fp32")
        self.precision.setAccessibleName("Precision")

        self.vram = QComboBox()
        self.vram.addItem("None", "none")
        self.vram.addItem("Low VRAM (TOON_MEM_STRATEGY=low)", "low")
        self.vram.setAccessibleName("VRAM strategy")
        self.vram.setToolTip("Upstream ToonCrafter memory strategy used during VAE decode. Does not substitute the model.")

        self.hw_warning = QLabel("")
        self.hw_warning.setWordWrap(True)
        self.hw_warning.setStyleSheet("color: #FFE082;")

        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Optional text prompt (empty string is in-distribution).")
        self.prompt.setAccessibleName("Text prompt")
        self.prompt.setFixedHeight(64)

        self.ckpt_folder = QLineEdit()
        self.ckpt_folder.setReadOnly(True)
        self.ckpt_folder.setAccessibleName("Checkpoint folder")
        browse_folder = QPushButton("Folder…")
        browse_folder.setAccessibleName("Choose checkpoint folder")
        browse_folder.clicked.connect(self._browse_folder)

        self.ckpt = QComboBox()
        self.ckpt.setAccessibleName("Checkpoint file")
        self.ckpt_status = QLabel("No checkpoint selected.")
        self.ckpt_status.setWordWrap(True)

        self.clip_path = QLineEdit()
        self.clip_path.setReadOnly(True)
        self.clip_path.setAccessibleName("OpenCLIP weights")
        browse_clip = QPushButton("CLIP…")
        browse_clip.setAccessibleName("Choose OpenCLIP weights")
        browse_clip.clicked.connect(self._browse_clip)
        self.use_cache = QCheckBox("Use an OpenCLIP cache already on this machine (never download)")
        self.use_cache.setAccessibleName("Use existing OpenCLIP cache")

        form = QFormLayout()
        size_row = QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        size_wrap = QWidget()
        size_wrap.setLayout(size_row)
        form.addRow("Output W × H", size_wrap)
        form.addRow("Aspect", self.aspect)
        form.addRow("In-betweens", self.intermediates)
        form.addRow("", self.plan_label)
        form.addRow("FPS", self.fps)
        form.addRow("Seed", self.seed)
        form.addRow("Steps", self.steps)
        form.addRow("Guidance", self.cfg)
        form.addRow("Eta", self.eta)
        form.addRow("Motion (fs)", self.motion)
        form.addRow("Generate at", self.gen_size)
        form.addRow("", self.offdist)
        form.addRow("Device", self.device)
        form.addRow("Precision", self.precision)
        form.addRow("VRAM", self.vram)
        form.addRow("", self.hw_warning)
        form.addRow("Prompt", self.prompt)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.ckpt_folder, 1)
        ckpt_row.addWidget(browse_folder)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip_path, 1)
        clip_row.addWidget(browse_clip)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Checkpoint folder"))
        layout.addLayout(ckpt_row)
        layout.addWidget(self.ckpt)
        layout.addWidget(self.ckpt_status)
        layout.addWidget(QLabel("OpenCLIP weights (open_clip_pytorch_model.bin)"))
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
            self.offdist.setText("Off-distribution size: the 512-interp UNet was trained at 320×512.")
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
        folder = QFileDialog.getExistingDirectory(self, "Checkpoint folder")
        if folder:
            self.set_checkpoint_folder(Path(folder))

    def _browse_clip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "OpenCLIP weights", "", "OpenCLIP weights (*.bin *.safetensors *.pt);;All files (*)"
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
            self.ckpt_status.setText("No .ckpt / .safetensors files in that folder (sketch_encoder.ckpt is ignored).")
        else:
            self._on_ckpt()
        self.changed.emit()

    def _on_ckpt(self) -> None:
        path = self.ckpt.currentData()
        if not path:
            self.ckpt_status.setText("No checkpoint selected.")
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
