from __future__ import annotations

from pathlib import Path

import pytest

from tooncrafter_animator.inference.adapter import ensure_vendor_on_path
from tooncrafter_animator.paths import inference_config_path

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")
pytest.importorskip("einops")
pytest.importorskip("pytorch_lightning")


def test_instantiate_unet_and_vae() -> None:
    """Build the real lvdm UNet and Dualref VAE from the vendored 512 config.

    Does not instantiate FrozenOpenCLIPEmbedder (that needs the ~4 GB CLIP file
    and would try to download it if USER_DEF_CLIP is unset).
    """
    ensure_vendor_on_path()
    from omegaconf import OmegaConf
    from ToonCrafter.utils.utils import instantiate_from_config

    config = OmegaConf.load(inference_config_path().as_posix())
    model_cfg = config.model
    unet_cfg = model_cfg.params.unet_config
    unet_cfg.params.use_checkpoint = False
    vae_cfg = model_cfg.params.first_stage_config

    unet = instantiate_from_config(unet_cfg)
    vae = instantiate_from_config(vae_cfg)
    assert unet.__class__.__name__ == "UNetModel"
    assert vae.__class__.__name__ == "AutoencoderKL_Dualref"
    assert int(unet.temporal_length) == 16
    x = torch.zeros(1, 8, 16, 10, 16)  # tiny spatial to keep RAM down
    # Don't run a full forward (needs context tensors); just prove parameters exist.
    n = sum(p.numel() for p in unet.parameters())
    assert n > 1_000_000
    del unet, vae, x
