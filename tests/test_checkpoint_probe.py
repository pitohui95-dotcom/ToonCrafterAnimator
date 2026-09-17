from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tooncrafter_animator.core.checkpoints import discover_checkpoints, probe_checkpoint

MARKERS = (
    "model.diffusion_model.input_blocks.0.0.weight",
    "first_stage_model.encoder.conv_in.weight",
    "image_proj_model.proj_in.weight",
    "embedder.model.visual.class_embedding",
    "cond_stage_model.model.token_embedding.weight",
    "model.diffusion_model.fps_embedding.0.weight",
)


def _write_safetensors(path: Path, keys: list[str], dtype: str = "F16") -> None:
    header = {}
    offset = 0
    for key in keys:
        header[key] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + 2]}
        offset += 2
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def test_safetensors_accepts_interp_markers(tmp_path: Path) -> None:
    path = tmp_path / "tooncrafter_512_interp-fp16.safetensors"
    _write_safetensors(path, list(MARKERS), "F16")
    probe = probe_checkpoint(path)
    assert probe.ok, probe.reason
    assert probe.format == "safetensors"
    assert probe.dtype_family == "fp16"


def test_safetensors_rejects_missing_unet(tmp_path: Path) -> None:
    path = tmp_path / "random.safetensors"
    _write_safetensors(path, ["some.other.key"], "F32")
    probe = probe_checkpoint(path)
    assert not probe.ok
    assert "missing" in probe.reason.lower() or "Not a ToonCrafter" in probe.reason


def test_skips_sketch_encoder(tmp_path: Path) -> None:
    (tmp_path / "sketch_encoder.ckpt").write_bytes(b"nope")
    good = tmp_path / "model.safetensors"
    _write_safetensors(good, list(MARKERS))
    found = discover_checkpoints(tmp_path)
    names = {p.name for p in found}
    assert "sketch_encoder.ckpt" not in names
    assert "model.safetensors" in names


def test_missing_file(tmp_path: Path) -> None:
    probe = probe_checkpoint(tmp_path / "nope.ckpt")
    assert not probe.ok


def test_torch_ckpt_roundtrip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.ckpt"
    state = {k: torch.zeros(1) for k in MARKERS}
    torch.save({"state_dict": state}, path)
    probe = probe_checkpoint(path)
    assert probe.ok, probe.reason
    assert probe.format == "ckpt"
