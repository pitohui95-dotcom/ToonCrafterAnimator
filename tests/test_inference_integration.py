from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.integration

CKPT = os.environ.get("TOONCRAFTER_CKPT")
CLIP = os.environ.get("TOONCRAFTER_CLIP")


@pytest.mark.skipif(not CKPT or not CLIP, reason="TOONCRAFTER_CKPT and TOONCRAFTER_CLIP are not set")
def test_real_inference_two_frames(tmp_path: Path) -> None:
    """The only test that can prove real ToonCrafter inference. Needs CUDA + weights."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for a practical integration run")

    from tooncrafter_animator.core.models import LoadPlan, PassRequest
    from tooncrafter_animator.inference.adapter import ToonCrafterAdapter

    start = tmp_path / "start.png"
    end = tmp_path / "end.png"
    Image.new("RGB", (512, 320), (220, 40, 40)).save(start)
    Image.new("RGB", (512, 320), (40, 40, 220)).save(end)

    adapter = ToonCrafterAdapter()
    adapter.load(
        LoadPlan(
            checkpoint=Path(CKPT),
            clip_weights=Path(CLIP),
            device="cuda:0",
            precision="fp16",
        )
    )
    try:
        frames = adapter.interpolate(
            PassRequest(
                start=np.asarray(Image.open(start).convert("RGB"), dtype=np.uint8),
                end=np.asarray(Image.open(end).convert("RGB"), dtype=np.uint8),
                prompt="",
                seed=123,
                steps=4,
                cfg_scale=7.5,
                eta=1.0,
                frame_stride=10,
                precision="fp16",
                device="cuda:0",
            )
        )
        assert frames.shape[0] == 16
        assert frames.shape[-1] == 3
        assert frames.dtype == np.uint8
    finally:
        adapter.unload()
