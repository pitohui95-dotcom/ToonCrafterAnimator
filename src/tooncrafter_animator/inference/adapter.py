"""ToonCrafterAdapter — the only module that imports vendored ToonCrafter / lvdm.

Call sequence is taken from AIGODLIKE/ComfyUI-ToonCrafter ``__init__.py``
(ToonCrafterNode.init / get_image) and ``ToonCrafter/scripts/evaluation/funcs.py``
(load_model_checkpoint, batch_ddim_sampling). Nothing about the model API is invented.
"""
from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Callable

import numpy as np

from tooncrafter_animator.core.checkpoints import probe_checkpoint
from tooncrafter_animator.core.errors import CheckpointError, InferenceCancelled, MissingDependency
from tooncrafter_animator.core.models import (
    CancellationToken,
    CheckpointProbe,
    LoadPlan,
    ModelInfo,
    PassRequest,
    ProgressEvent,
    StageCallback,
)
from tooncrafter_animator.hardware import precision_allowed
from tooncrafter_animator.memory import release_torch_memory
from tooncrafter_animator.paths import inference_config_path, vendor_root
from tooncrafter_animator import copy as t

log = logging.getLogger("tooncrafter")

NATIVE_SIZE = (320, 512)  # h, w
FRAMES_PER_PASS = 16

_PATH_READY = False


def ensure_vendor_on_path() -> None:
    """Make ``import lvdm`` and ``from ToonCrafter.utils.utils`` work."""
    global _PATH_READY
    if _PATH_READY:
        return
    root = vendor_root()
    inner = root / "ToonCrafter"
    for path in (str(root), str(inner)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATH_READY = True


def inspect_checkpoint(path: Path) -> CheckpointProbe:
    return probe_checkpoint(Path(path))


def find_openclip_cache() -> Path | None:
    """Locate a previously downloaded ViT-H-14 / laion2b_s32b_b79k weight file.

    Never downloads. Returns None if the cache is absent.
    """
    names = (
        "open_clip_pytorch_model.bin",
        "open_clip_pytorch_model.safetensors",
    )
    hints = (
        Path.home() / ".cache" / "clip",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers",
    )
    env = os.environ.get("USER_DEF_CLIP")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for hint in hints:
        if not hint.exists():
            continue
        for name in names:
            direct = hint / name
            if direct.is_file():
                return direct
        for match in hint.rglob("*laion2b_s32b_b79k*"):
            if match.is_file() and match.suffix in {".bin", ".safetensors", ".pt"}:
                return match
        for match in hint.rglob("open_clip_pytorch_model.bin"):
            if match.is_file():
                return match
    return None


def _require_torch():
    try:
        import torch
        import torchvision  # noqa: F401
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise MissingDependency(t.ERR_NO_TORCH_STACK) from exc
    return torch, OmegaConf


class ToonCrafterAdapter:
    NATIVE_SIZE = NATIVE_SIZE
    FRAMES_PER_PASS = FRAMES_PER_PASS

    def __init__(self) -> None:
        self._model = None
        self._info: ModelInfo | None = None
        self._torch = None

    @staticmethod
    def inspect_checkpoint(path: Path) -> CheckpointProbe:
        return inspect_checkpoint(path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> ModelInfo | None:
        return self._info

    def load(self, plan: LoadPlan, on_stage: StageCallback | None = None) -> ModelInfo:
        def stage(msg: str) -> None:
            log.info("load_stage msg=%s", msg)
            if on_stage:
                on_stage(msg)

        torch, OmegaConf = _require_torch()
        self._torch = torch
        ok, reason = precision_allowed(plan.device, plan.precision)
        if not ok:
            raise CheckpointError(reason)

        probe = inspect_checkpoint(plan.checkpoint)
        if not probe.ok:
            raise CheckpointError(probe.reason)

        clip_path = plan.clip_weights
        if clip_path is None and plan.use_openclip_cache:
            clip_path = find_openclip_cache()
            if clip_path is None:
                raise CheckpointError(t.ERR_NO_CLIP_CACHE)
        if clip_path is None or not Path(clip_path).is_file():
            raise CheckpointError(t.ERR_CLIP_REQUIRED)

        ensure_vendor_on_path()
        os.environ["USER_DEF_CLIP"] = str(Path(clip_path).resolve())
        # Prevent Hugging Face / open_clip from going to the network if something
        # still asks for a pretrained tag.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from ToonCrafter.utils.utils import instantiate_from_config

        config_file = inference_config_path()
        if not config_file.is_file():
            raise CheckpointError(t.ERR_MISSING_CONFIG.format(path=config_file))
        stage(t.STAGE_READ_YAML)
        config = OmegaConf.load(config_file.as_posix())
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False

        stage(t.STAGE_INSTANTIATE)
        model = instantiate_from_config(model_config)
        stage(t.STAGE_LOAD_CKPT.format(name=plan.checkpoint.name))
        model = load_model_checkpoint(model, Path(plan.checkpoint))
        model.eval()

        device = _resolve_device(torch, plan.device)
        if plan.precision == "fp16":
            stage(t.STAGE_CAST_FP16)
            model = model.half()
        stage(t.STAGE_MOVE.format(device=device))
        model = model.to(device)

        temporal = int(getattr(model, "temporal_length", FRAMES_PER_PASS))
        n_params = sum(p.numel() for p in model.parameters())
        info = ModelInfo(
            checkpoint=Path(plan.checkpoint),
            device=str(device),
            precision=plan.precision,
            temporal_length=temporal,
            native_height=NATIVE_SIZE[0],
            native_width=NATIVE_SIZE[1],
            param_count=n_params,
        )
        self._model = model
        self._info = info
        stage(t.STAGE_READY)
        return info

    def unload(self) -> None:
        model = self._model
        self._model = None
        self._info = None
        if model is None:
            release_torch_memory()
            return
        torch = self._torch
        try:
            if torch is not None:
                model.to("cpu")
        except Exception as exc:  # pragma: no cover
            log.warning("model_to_cpu_failed error=%s", exc)
        del model
        release_torch_memory()

    def interpolate(
        self,
        job: PassRequest,
        progress: Callable[[ProgressEvent], None] | None = None,
        cancel: CancellationToken | None = None,
        pass_index: int = 1,
        n_passes: int = 1,
    ) -> np.ndarray:
        """Run one real ToonCrafter pass. Returns uint8 array shaped (16, H, W, 3)."""
        if self._model is None:
            raise CheckpointError(t.ERR_MODEL_NOT_LOADED)
        torch = self._torch
        if torch is None:
            torch, _ = _require_torch()
            self._torch = torch
        token = cancel or CancellationToken()
        token.raise_if_cancelled()

        from einops import repeat
        from torchvision import transforms

        model = self._model
        os.environ["TOON_MEM_STRATEGY"] = job.vram_strategy or "none"
        device = model.device if hasattr(model, "device") else _resolve_device(torch, job.device)
        half = job.precision == "fp16"
        resolution = (int(job.gen_height), int(job.gen_width))
        seed = int(job.seed) % 4294967295
        try:
            from pytorch_lightning import seed_everything

            seed_everything(seed)
        except Exception:
            torch.manual_seed(seed)
            np.random.seed(seed)

        transform = transforms.Compose(
            [
                transforms.Resize(min(resolution)),
                transforms.CenterCrop(resolution),
            ]
        )

        start = _hwc_uint8_to_nchw01(torch, job.start).to(device)
        end = _hwc_uint8_to_nchw01(torch, job.end).to(device)
        if half:
            start = start.half()
            end = end.half()

        batch_size = 1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        latent_h, latent_w = resolution[0] // 8, resolution[1] // 8
        noise_shape = [batch_size, channels, frames, latent_h, latent_w]

        def cb(step: int) -> None:
            token.raise_if_cancelled()
            if progress:
                progress(
                    ProgressEvent(
                        pass_index=pass_index,
                        n_passes=n_passes,
                        step=step + 1,
                        n_steps=job.steps,
                        message=t.STATUS_DDIM.format(step=step + 1, n_steps=job.steps),
                    )
                )

        amp = torch.cuda.amp.autocast() if (torch.cuda.is_available() and str(device).startswith("cuda")) else nullcontext()
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            stack.enter_context(amp)
            text_emb = model.get_learned_conditioning([job.prompt])

            img_tensor = start[0]
            img_tensor = (img_tensor - 0.5) * 2
            image_tensor_resized = transform(img_tensor)
            videos = image_tensor_resized.unsqueeze(0).unsqueeze(2)
            videos = repeat(videos, "b c t h w -> b c (repeat t) h w", repeat=frames // 2)

            img_tensor2 = end[0]
            img_tensor2 = (img_tensor2 - 0.5) * 2
            image_tensor_resized2 = transform(img_tensor2)
            videos2 = image_tensor_resized2.unsqueeze(0).unsqueeze(2)
            videos2 = repeat(videos2, "b c t h w -> b c (repeat t) h w", repeat=frames // 2)
            videos = torch.cat([videos, videos2], dim=2)

            z, hs = get_latent_z_with_hidden_states(model, videos)

            img_tensor_repeat = torch.zeros_like(z).to(dtype=model.dtype)
            img_tensor_repeat[:, :, :1, :, :] = z[:, :, :1, :, :]
            img_tensor_repeat[:, :, -1:, :, :] = z[:, :, -1:, :, :]

            cond_images = model.embedder(img_tensor.unsqueeze(0))
            img_emb = model.image_proj_model(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)

            fs = torch.tensor([job.frame_stride], dtype=torch.long, device=model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}

            token.raise_if_cancelled()
            batch_samples = batch_ddim_sampling(
                model,
                cond,
                noise_shape,
                n_samples=1,
                ddim_steps=job.steps,
                ddim_eta=job.eta,
                cfg_scale=job.cfg_scale,
                hs=hs,
                callback=cb,
            )

        # b,samples,c,t,h,w → t,h,w,c in [0,1]
        frames_t = batch_samples[0][0].permute(1, 2, 3, 0)
        if half or frames_t.dtype != torch.float32:
            frames_t = frames_t.to(dtype=torch.float32)
        frames_t = torch.clamp(frames_t, -1.0, 1.0)
        frames_t = (frames_t + 1.0) * 0.5
        cpu = frames_t.detach().cpu().numpy()
        uint8 = np.clip(np.round(cpu * 255.0), 0, 255).astype(np.uint8)
        if uint8.shape[0] != FRAMES_PER_PASS:
            log.warning("unexpected_frame_count got=%s expected=%s", uint8.shape[0], FRAMES_PER_PASS)
        return uint8


def _hwc_uint8_to_nchw01(torch, arr: np.ndarray):
    if arr.ndim != 3:
        raise ValueError("Keyframe must be HWC")
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def _resolve_device(torch, spec: str):
    spec = spec or "cpu"
    if spec == "cpu":
        return torch.device("cpu")
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise CheckpointError(t.ERR_CUDA_UNAVAILABLE)
        return torch.device(spec)
    return torch.device(spec)


def get_latent_z_with_hidden_states(model, videos):
    """Identical to ToonCrafterNode.get_latent_z_with_hidden_states."""
    import torch
    from einops import rearrange

    b, c, t, h, w = videos.shape
    x = rearrange(videos, "b c t h w -> (b t) c h w")
    encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)

    hidden_states_first_last = []
    for hid in hidden_states:
        hid = rearrange(hid, "(b t) c h w -> b c t h w", t=t)
        hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
        hidden_states_first_last.append(hid_new)

    z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
    return z, hidden_states_first_last


def load_model_checkpoint(model, ckpt: Path):
    """Upstream funcs.load_model_checkpoint, with weights_only=True on torch.load."""
    import torch

    def _load(model, ckpt: Path, full_strict: bool):
        suffix = ckpt.suffix.lower()
        if suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(ckpt), device="cpu")
        else:
            try:
                state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            except TypeError:
                # Torch too old for weights_only — still refuse pickle extras when possible.
                state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=full_strict)
        except Exception:
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v
            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        return model

    _load(model, Path(ckpt), full_strict=True)
    log.info("checkpoint_loaded path=%s", ckpt)
    return model


def batch_ddim_sampling(
    model,
    cond,
    noise_shape,
    n_samples=1,
    ddim_steps=50,
    ddim_eta=1.0,
    cfg_scale=1.0,
    hs=None,
    temporal_cfg_scale=None,
    callback=None,
    **kwargs,
):
    """Byte-level translation of funcs.batch_ddim_sampling (no cv2/decord)."""
    ensure_vendor_on_path()
    import torch
    from lvdm.models.samplers.ddim import DDIMSampler

    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    fs = cond["fs"]
    del cond["fs"]
    if noise_shape[-1] == 32:
        timestep_spacing = "uniform"
        guidance_rescale = 0.0
    else:
        timestep_spacing = "uniform_trailing"
        guidance_rescale = 0.7
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
        else:
            raise RuntimeError(f"Unsupported uncond_type {uncond_type!r}")
        if hasattr(model, "embedder"):
            uc_img = torch.zeros(noise_shape[0], 3, 224, 224).to(model.device)
            if uc_img.dtype != model.dtype:
                uc_img = uc_img.to(model.dtype)
            uc_img = model.embedder(uc_img)
            uc_img = model.image_proj_model(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        if isinstance(cond, dict):
            uc = {key: cond[key] for key in cond.keys()}
            uc.update({"c_crossattn": [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None

    additional_decode_kwargs = {"ref_context": hs}
    x_T = None
    batch_variants = []
    for _ in range(n_samples):
        kwargs.update({"clean_cond": True})
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=noise_shape[0],
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            temporal_length=noise_shape[2],
            conditional_guidance_scale_temporal=temporal_cfg_scale,
            x_T=x_T,
            fs=fs,
            precision=16 if model.dtype == torch.float16 else 32,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            callback=callback,
            **kwargs,
        )
        batch_images = model.decode_first_stage(samples, **additional_decode_kwargs)
        index = list(range(samples.shape[2]))
        del index[1]
        del index[-2]
        samples_mid = samples[:, :, index, :, :]
        batch_images_middle = model.decode_first_stage(samples_mid, **additional_decode_kwargs)
        batch_images[:, :, batch_images.shape[2] // 2 - 1 : batch_images.shape[2] // 2 + 1] = (
            batch_images_middle[:, :, batch_images.shape[2] // 2 - 2 : batch_images.shape[2] // 2]
        )
        batch_variants.append(batch_images)
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants
