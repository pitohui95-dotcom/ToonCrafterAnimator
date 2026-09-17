# Vendored ToonCrafter provenance

This directory redistributes a **subset** of

* [AIGODLIKE/ComfyUI-ToonCrafter](https://github.com/AIGODLIKE/ComfyUI-ToonCrafter)
  (Apache-2.0), commit `96024189ecb2bcc7014a439b3c8676108cc26738` (2024-07-17)
* which itself vendors [ToonCrafter/ToonCrafter](https://github.com/ToonCrafter/ToonCrafter)
  (Apache-2.0)

Both upstream `LICENSE` files are kept verbatim:

* `LICENSE` — ComfyUI-ToonCrafter / Tencent Apache-2.0 text
* `ToonCrafter/LICENSE` — ToonCrafter Apache-2.0 text

## Why a subset

The standalone animator only needs the **512 interpolation** inference graph.
ComfyUI node wrappers, training, Gradio, ControlNet/sketch (`cldm/`), MiDaS/BSRGAN
(`ldm/`), and dataset loaders are not used and are not shipped.

## Unmodified files (byte-for-byte with the commit above)

* `ToonCrafter/__init__.py`
* `ToonCrafter/LICENSE`
* `ToonCrafter/configs/inference_512_v1.0.yaml`
* `ToonCrafter/lvdm/__init__.py`
* `ToonCrafter/lvdm/basics.py`
* `ToonCrafter/lvdm/common.py`
* `ToonCrafter/lvdm/distributions.py`
* `ToonCrafter/lvdm/ema.py`
* `ToonCrafter/lvdm/models/autoencoder.py`
* `ToonCrafter/lvdm/models/autoencoder_dualref.py`
* `ToonCrafter/lvdm/models/ddpm3d.py`
* `ToonCrafter/lvdm/models/utils_diffusion.py`
* `ToonCrafter/lvdm/models/samplers/ddim.py`
* `ToonCrafter/lvdm/models/samplers/ddim_multiplecond.py`
* `ToonCrafter/lvdm/modules/attention.py`
* `ToonCrafter/lvdm/modules/attention_svd.py`
* `ToonCrafter/lvdm/modules/x_transformer.py`
* `ToonCrafter/lvdm/modules/encoders/condition.py`
* `ToonCrafter/lvdm/modules/encoders/resampler.py`
* `ToonCrafter/lvdm/modules/networks/ae_modules.py`
* `ToonCrafter/lvdm/modules/networks/openaimodel3d.py`

## Files that differ from upstream (Apache-2.0 §4(b))

* `ToonCrafter/utils/utils.py` — slimmed. Removed `cv2` image helpers and
  `torch.distributed` setup; kept `count_params`, `instantiate_from_config`,
  and `get_obj_from_str` with original behaviour. Header comment in the file.

## Files added here (not in upstream)

* `PROVENANCE.md` (this file)
* `ToonCrafter/utils/__init__.py` — empty package marker so
  `ToonCrafter.utils.utils` imports cleanly without relying on implicit
  namespace packages.

## Intentionally not vendored

ComfyUI-ToonCrafter `__init__.py` (ComfyUI node, `huggingface_hub` auto-download,
`comfy.model_management`), `cldm/` (imports `comfy.ldm.util`), `ldm/`, `main/`,
`scripts/`, `lvdm/data/`, Gradio apps, prompts, and training configs.

## Runtime patches applied *outside* this tree

`tooncrafter_animator.inference.adapter` is the only application module that
imports `lvdm`. It:

* sets `USER_DEF_CLIP` to a user-selected local OpenCLIP weight file (no download)
* forces `unet_config.params.use_checkpoint = False` (same as the ComfyUI node)
* calls `torch.load(..., weights_only=True)` when loading `.ckpt` files
* never executes scripts from a checkpoint folder

Model **weights** are not redistributed.
