from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tooncrafter_animator.core.errors import CheckpointError
from tooncrafter_animator.hardware import (
    default_device_id,
    default_precision,
    list_devices,
    precision_allowed,
    torch_cuda_build,
)
from tooncrafter_animator.inference.adapter import _bind_aux_devices, _resolve_device


def test_cpu_wheel_is_not_reported_as_cuda_build() -> None:
    # This Linux venv uses CPU torch; a CUDA EXE must not be built from that wheel.
    if getattr(torch.version, "cuda", None):
        pytest.skip("this interpreter is a CUDA wheel")
    assert torch_cuda_build() is False
    devices = list_devices()
    assert devices[0].kind == "cpu"
    assert "CPU" in devices[0].warning or "CPU" in devices[0].name


def test_list_devices_includes_cuda_and_defaults_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(
            name="NVIDIA GeForce RTX Test",
            total_memory=24 * 1024**3,
            major=8,
            minor=9,
        ),
    )
    devices = list_devices()
    cuda = [d for d in devices if d.kind == "cuda"]
    assert len(cuda) == 1
    assert cuda[0].id == "cuda:0"
    assert cuda[0].name == "NVIDIA GeForce RTX Test"
    assert default_device_id(devices) == "cuda:0"
    assert default_precision("cuda:0") == "fp16"
    ok, reason = precision_allowed("cuda:0", "fp16")
    assert ok is True
    assert reason == ""


def test_resolve_device_cuda_requires_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CheckpointError):
        _resolve_device(torch, "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    dev = _resolve_device(torch, "cuda:0")
    assert str(dev).startswith("cuda")


def test_bind_aux_devices_copies_concrete_device() -> None:
    class Sub:
        def __init__(self) -> None:
            self.device = "cuda"

    class Model:
        def __init__(self) -> None:
            self.cond_stage_model = Sub()
            self.embedder = Sub()

    model = Model()
    target = torch.device("cpu")
    _bind_aux_devices(model, target)
    assert model.cond_stage_model.device == target
    assert model.embedder.device == target


def test_fp16_rejected_on_cpu() -> None:
    ok, reason = precision_allowed("cpu", "fp16")
    assert ok is False
    assert "CUDA" in reason or "cuda" in reason.lower() or "FP16" in reason
