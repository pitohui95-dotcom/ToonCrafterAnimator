from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    name: str
    kind: str  # cpu | cuda
    total_vram_bytes: int | None = None
    capability: str | None = None
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "total_vram_bytes": self.total_vram_bytes,
            "capability": self.capability,
            "warning": self.warning,
        }


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        # ImportError, or a broken partial install (missing CUDA .so, etc.)
        return False


def list_devices() -> list[DeviceInfo]:
    cpu_warning = (
        "CPU inference of the 16-frame 3D UNet is extremely slow "
        "(tens of minutes to hours per pass at 50 DDIM steps)."
    )
    devices = [
        DeviceInfo(
            id="cpu",
            name="CPU",
            kind="cpu",
            total_vram_bytes=None,
            warning=cpu_warning,
        )
    ]
    try:
        import torch
    except Exception:
        devices[0] = DeviceInfo(
            id="cpu",
            name="CPU (PyTorch not installed)",
            kind="cpu",
            warning="Install PyTorch to run ToonCrafter. Interpolate stays disabled until then.",
        )
        return devices

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            vram = int(props.total_memory)
            warning = ""
            if vram < 11 * 1024**3:
                warning = (
                    f"Reported VRAM is {vram / 1024**3:.1f} GB. FP16 320×512 / 16-frame "
                    "inference typically needs about 11–13 GB."
                )
            devices.append(
                DeviceInfo(
                    id=f"cuda:{index}",
                    name=props.name,
                    kind="cuda",
                    total_vram_bytes=vram,
                    capability=f"{props.major}.{props.minor}",
                    warning=warning,
                )
            )
    return devices


def default_device_id(devices: list[DeviceInfo] | None = None) -> str:
    devices = devices or list_devices()
    for device in devices:
        if device.kind == "cuda":
            return device.id
    return "cpu"


def default_precision(device_id: str) -> str:
    return "fp16" if device_id.startswith("cuda") else "fp32"


def precision_allowed(device_id: str, precision: str) -> tuple[bool, str]:
    if precision == "fp16" and not device_id.startswith("cuda"):
        return False, "FP16 is only offered on CUDA. PyTorch has no usable FP16 CPU kernels for this graph."
    if precision not in {"fp16", "fp32"}:
        return False, f"Unknown precision {precision!r}."
    return True, ""


def vram_warning(device: DeviceInfo, precision: str) -> str:
    if device.kind != "cuda" or device.total_vram_bytes is None:
        if device.kind == "cpu":
            return device.warning
        return ""
    need = 12 * 1024**3 if precision == "fp16" else 22 * 1024**3
    if device.total_vram_bytes < need:
        return (
            f"{device.name} has {device.total_vram_bytes / 1024**3:.1f} GB VRAM; "
            f"{precision.upper()} interpolation typically needs about {need / 1024**3:.0f} GB. "
            "The job may fail with an out-of-memory error."
        )
    return device.warning
