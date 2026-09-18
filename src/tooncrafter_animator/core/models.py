from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


class AspectMode(str, Enum):
    PRESERVE = "preserve"
    CROP = "crop"


class Precision(str, Enum):
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass(frozen=True)
class CheckpointProbe:
    path: Path
    ok: bool
    reason: str
    size_bytes: int
    format: str  # safetensors | ckpt | unknown
    dtype_family: str  # fp16 | fp32 | mixed | unknown
    keys_checked: int = 0
    marker_hits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "ok": self.ok,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "dtype_family": self.dtype_family,
            "keys_checked": self.keys_checked,
            "marker_hits": list(self.marker_hits),
        }


@dataclass
class GenerationParams:
    output_width: int = 512
    output_height: int = 320
    aspect: AspectMode = AspectMode.PRESERVE
    intermediates: int = 14
    fps: int = 8
    seed: int = 123
    steps: int = 50
    cfg_scale: float = 7.5
    eta: float = 1.0
    motion_stride: int = 10  # ComfyUI's "frame_count" — FPS conditioning, not frame count
    precision: str = "fp32"
    device: str = "cpu"
    prompt: str = ""
    gen_height: int = 320
    gen_width: int = 512
    vram_strategy: str = "none"


@dataclass
class LoadPlan:
    checkpoint: Path
    clip_weights: Path | None
    use_openclip_cache: bool = False
    device: str = "cpu"
    precision: str = "fp32"


@dataclass
class PassRequest:
    start: np.ndarray  # HWC uint8 RGB
    end: np.ndarray
    prompt: str
    seed: int
    steps: int
    cfg_scale: float
    eta: float
    frame_stride: int
    precision: str
    device: str
    gen_height: int = 320
    gen_width: int = 512
    vram_strategy: str = "none"


@dataclass
class ModelInfo:
    checkpoint: Path
    device: str
    precision: str
    temporal_length: int
    native_height: int
    native_width: int
    param_count: int | None = None


@dataclass
class PassSpec:
    index: int
    take_indices: list[int]
    include_end_anchor: bool = False
    source: str = "user"  # user | scout_anchor


@dataclass
class FramePlan:
    n_intermediates: int
    n_passes: int
    scout: bool
    k_segments: int
    passes: list[PassSpec]
    description: str
    anchor_indices: list[int] = field(default_factory=list)


@dataclass
class ProgressEvent:
    pass_index: int
    n_passes: int
    step: int
    n_steps: int
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.n_passes <= 0 or self.n_steps <= 0:
            return 0.0
        done_passes = max(self.pass_index - 1, 0)
        return min(1.0, (done_passes + self.step / self.n_steps) / self.n_passes)


class StageCallback(Protocol):
    def __call__(self, message: str) -> None: ...


class ProgressCallback(Protocol):
    def __call__(self, event: ProgressEvent) -> None: ...


@dataclass
class CancellationToken:
    _cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            from tooncrafter_animator.core.errors import InferenceCancelled
            from tooncrafter_animator import copy as t

            raise InferenceCancelled(t.ERR_CANCELLED)
