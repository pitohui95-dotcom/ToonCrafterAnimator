from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tooncrafter_animator.core.models import CancellationToken, PassRequest, ProgressEvent
from tooncrafter_animator.inference import adapter as adapter_mod
from tooncrafter_animator.inference.adapter import ToonCrafterAdapter, batch_ddim_sampling


class StubEmbedder(torch.nn.Module):
    def forward(self, x):
        # x: b c h w → b l c  (open_clip image tokens; 1 token is enough for the cat)
        b = x.shape[0]
        return torch.zeros(b, 1, 1024, device=x.device, dtype=x.dtype)


class StubProj(torch.nn.Module):
    def forward(self, x):
        return x


class StubFirstStage:
    def encode(self, x, return_hidden_states=False):
        b, c, h, w = x.shape
        z = torch.zeros(b, 4, h // 8 if h >= 8 else 1, w // 8 if w >= 8 else 1, device=x.device, dtype=x.dtype)
        hidden = [torch.zeros(b, 8, h, w, device=x.device, dtype=x.dtype)]
        if return_hidden_states:
            return z, hidden
        return z


class StubDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 4


class StubSampler:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        shape = kwargs["shape"]  # c, t, h, w
        b = kwargs["batch_size"]
        samples = torch.zeros(b, *shape, device=self.model.device, dtype=self.model.dtype)
        if kwargs.get("callback"):
            for i in range(kwargs["S"]):
                kwargs["callback"](i)
        return samples, None


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.uncond_type = "empty_seq"
        self.temporal_length = 16
        self.model = SimpleNamespace(diffusion_model=StubDiffusion())
        self.embedder = StubEmbedder()
        self.image_proj_model = StubProj()
        self.first_stage_model = StubFirstStage()
        self._device = torch.device("cpu")
        self.decode_calls = []
        self.cond_calls = []

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return next(self.parameters()).dtype if list(self.parameters()) else torch.float32

    def get_learned_conditioning(self, prompts):
        self.cond_calls.append(list(prompts))
        return torch.zeros(len(prompts), 77, 1024)

    def get_first_stage_encoding(self, posterior):
        return posterior

    def decode_first_stage(self, samples, **kwargs):
        self.decode_calls.append({"shape": tuple(samples.shape), "ref_context": kwargs.get("ref_context")})
        # samples: b c t h w → b c t H W  (pixel, same spatial for the stub)
        b, c, t, h, w = samples.shape
        return torch.zeros(b, 3, t, h * 8 if h else 8, w * 8 if w else 8, dtype=torch.float32)


def test_batch_ddim_shapes_and_ref_context(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter_mod.ensure_vendor_on_path()
    import lvdm.models.samplers.ddim as ddim_mod

    model = StubModel()
    sampler = StubSampler(model)
    monkeypatch.setattr(ddim_mod, "DDIMSampler", lambda m: sampler)

    b, c, t, h, w = 1, 4, 16, 40, 64
    cond = {
        "c_crossattn": [torch.zeros(1, 77 + 1, 1024)],
        "fs": torch.tensor([10]),
        "c_concat": [torch.zeros(b, c, t, h, w)],
    }
    hs = [torch.zeros(1, 8, 2, 320, 512)]
    out = batch_ddim_sampling(
        model, cond, [b, c, t, h, w], n_samples=1, ddim_steps=3, cfg_scale=7.5, hs=hs, callback=lambda i: None
    )
    assert out.shape[0] == 1  # batch
    assert out.shape[1] == 1  # n_samples
    assert sampler.calls, "DDIMSampler.sample was not called"
    call = sampler.calls[0]
    assert tuple(call["shape"]) == (c, t, h, w)
    assert "fs" in call
    assert call["unconditional_guidance_scale"] == 7.5
    assert call["timestep_spacing"] == "uniform_trailing"
    assert call["guidance_rescale"] == 0.7
    assert model.decode_calls, "decode_first_stage was not called"
    assert model.decode_calls[0]["ref_context"] is hs
    # uncond empty_seq
    assert model.cond_calls and model.cond_calls[0] == [""]


def test_adapter_interpolate_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ToonCrafterAdapter()
    model = StubModel()
    adapter._model = model
    adapter._torch = torch
    adapter._info = None

    def boom(*a, **k):
        raise adapter_mod.InferenceCancelled("stop")

    monkeypatch.setattr(adapter_mod, "batch_ddim_sampling", boom)
    monkeypatch.setattr(adapter_mod, "get_latent_z_with_hidden_states", lambda *a, **k: (torch.zeros(1, 4, 16, 40, 64), []))

    start = np.zeros((320, 512, 3), dtype=np.uint8)
    end = np.ones((320, 512, 3), dtype=np.uint8) * 255
    job = PassRequest(
        start=start,
        end=end,
        prompt="",
        seed=1,
        steps=2,
        cfg_scale=7.5,
        eta=1.0,
        frame_stride=10,
        precision="fp32",
        device="cpu",
    )
    token = CancellationToken()
    with pytest.raises(adapter_mod.InferenceCancelled):
        adapter.interpolate(job, cancel=token)


def test_unload_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ToonCrafterAdapter()
    adapter._model = StubModel()
    adapter.unload()
    assert adapter.is_loaded is False
