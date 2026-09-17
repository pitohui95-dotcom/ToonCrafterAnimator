# Third-party notices — ToonCrafter Animator

This file is the source-tree copy. The Windows release folder ships the same
content as `THIRD_PARTY_NOTICES.txt` plus verbatim license files under `licenses/`.

## ToonCrafter Animator

Apache License 2.0. See `LICENSE`.

## ToonCrafter (vendored inference code)

https://github.com/ToonCrafter/ToonCrafter
Apache License 2.0. Copyright Tencent.
Verbatim license: `src/tooncrafter_animator/vendor/tooncrafter/ToonCrafter/LICENSE`
Modifications: `src/tooncrafter_animator/vendor/tooncrafter/PROVENANCE.md`

## ComfyUI-ToonCrafter (source of the ComfyUI node this app replaces)

https://github.com/AIGODLIKE/ComfyUI-ToonCrafter
Apache License 2.0.
Verbatim license: `src/tooncrafter_animator/vendor/tooncrafter/LICENSE`
The ComfyUI node, auto-download, and ControlNet/sketch path are **not** vendored.

## PyTorch / torchvision

BSD-style license (https://github.com/pytorch/pytorch/blob/main/LICENSE).
Not bundled in the source tree. The Windows packager collects the wheels the
builder installed (CPU or CUDA).

## PySide6 / Qt

GNU Lesser General Public License v3.
This app **dynamically links** unmodified PySide6 wheels. You can relink against
a different Qt/PySide6 build by installing PySide6 from PyPI or building Qt
from https://code.qt.io/cgit/qt/qtbase.git and replacing the bundled PySide6
DLLs/pyd files in the onedir `_internal` folder. Source for Qt is available
from The Qt Company. A copy of the LGPL-3.0 text is shipped in
`licenses/LGPLv3.txt` of the release folder.

## Pillow (PIL)

MIT-CMU license. https://github.com/python-pillow/Pillow

## NumPy

BSD-3-Clause. https://numpy.org/

## einops

MIT. https://github.com/arogozhnikov/einops

## OmegaConf

BSD-3-Clause. https://github.com/omry/omegaconf

## safetensors

Apache-2.0. https://github.com/huggingface/safetensors

## open_clip (open_clip_torch)

MIT. https://github.com/mlfoundations/open_clip
The **ViT-H-14 / laion2b_s32b_b79k weights** are a separate asset, not shipped.

## transformers (Hugging Face)

Apache-2.0. https://github.com/huggingface/transformers

## kornia

Apache-2.0. https://github.com/kornia/kornia

## pytorch-lightning

Apache-2.0. https://github.com/Lightning-AI/pytorch-lightning

## PyYAML

MIT. https://github.com/yaml/pyyaml

## tqdm

MPL-2.0 / MIT. https://github.com/tqdm/tqdm

## ffmpeg (release folder only)

The Windows packager is required to fetch an **LGPL** ffmpeg build
(`packaging/fetch_ffmpeg.py` pins the URL and records the SHA-256).
A GPL ffmpeg build must not be redistributed with this app.
LGPL-2.1 text is shipped in `licenses/LGPLv2.1.txt`.

## Model weights (not redistributed)

* Doubiiu/ToonCrafter `model.ckpt` — terms on the Hugging Face model card
* Kijai/DynamiCrafter_pruned `tooncrafter_512_interp-fp16.safetensors` — terms on the model card
* OpenCLIP `ViT-H-14 laion2b_s32b_b79k` — terms of the OpenCLIP / LAION weight release

The Apache-2.0 license on the ToonCrafter **code** does not grant rights to the
weights. This application will not download them for you.
