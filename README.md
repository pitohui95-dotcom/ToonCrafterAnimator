# ToonCrafter Animator

Standalone **Windows** desktop app that interpolates two cartoon keyframes using
the **real** ToonCrafter 512-interp latent video diffusion model.

There is no ComfyUI node, no auto-download, and **no substitute interpolator**.
If the checkpoint cannot run, Interpolate stays disabled and the UI says why.

Source of the model API: [AIGODLIKE/ComfyUI-ToonCrafter](https://github.com/AIGODLIKE/ComfyUI-ToonCrafter)
(vendors [ToonCrafter/ToonCrafter](https://github.com/ToonCrafter/ToonCrafter)), both Apache-2.0.
Inspected at commit `96024189ecb2bcc7014a439b3c8676108cc26738`.

## What you get

- Start / end keyframes with previews, swap, drag-and-drop
- Output width × height; preserve (fit + pad) or crop aspect
- Intermediate frame count, playback FPS, seed, DDIM steps, CFG, eta
- **Motion (frame stride)** — the model's FPS condition (5–30). This is *not* the
  frame count; the ComfyUI node mislabels it. One pass always yields **16 frames**.
- FP16 (CUDA only) / FP32, device picker, checkpoint folder, OpenCLIP weights
- Progress, pass i/N, cooperative cancel (finishes the current DDIM step)
- Timeline preview, PNG / MP4 / GIF export, open output folder
- Save / load project JSON
- First-run setup stored under `%LOCALAPPDATA%\ToonCrafterAnimator` (or
  `~/.local/share/ToonCrafterAnimator` on Linux)

## Weights (you bring them)

Never shipped, never downloaded by this app:

| File | Typical size | Where people get it |
| --- | --- | --- |
| `model.ckpt` or `tooncrafter_512_interp-fp16.safetensors` | ~10 GB / ~5 GB | Doubiiu/ToonCrafter, Kijai/DynamiCrafter_pruned |
| `open_clip_pytorch_model.bin` (ViT-H-14 / laion2b_s32b_b79k) | ~4 GB | OpenCLIP / LAION card |

The Apache-2.0 license on the **code** does not cover the weights.

## Run from source (Linux or Windows)

Python 3.10+ (3.12 is fine for development). This repository is developed on Linux
and ships a complete Windows packaging pipeline.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python packaging/generate_icon.py
python -m tooncrafter_animator
```

CUDA (Windows/Linux NVIDIA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Headless self-test:

```bash
QT_QPA_PLATFORM=offscreen python -m tooncrafter_animator --selftest
```

Tests:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest
```

Real-weights integration (CUDA machine with files on disk):

```bash
export TOONCRAFTER_CKPT=/path/to/tooncrafter_512_interp-fp16.safetensors
export TOONCRAFTER_CLIP=/path/to/open_clip_pytorch_model.bin
python -m pytest -m integration
```

## Windows `.exe`

PyInstaller does **not** cross-compile. On a Windows 10/11 x64 machine with
Python 3.10:

```bat
build_exe.bat
```

or

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

That produces `release/ToonCrafterAnimator/` with the onedir exe, `ffmpeg/`
(LGPL build), `assets/`, `licenses/`, `README.txt`, `THIRD_PARTY_NOTICES.txt`,
and `checksums.sha256`.

Onedir is intentional. A onefile bundle of torch unpacks several GB to `%TEMP%`
on every launch.

On this Linux VM the same `packaging/make_release.py` still assembles the folder
layout (minus a native `.exe`). Run from source to exercise the UI here.

## Model facts that drive the UI

- `temporal_length = 16` → one pass = start + 14 in-betweens + end
- Trained at 320×512; other 64-divisible sizes run but are flagged off-distribution
- `cfg_scale` (default 7.5) is real classifier-free guidance (`uncond_type: empty_seq`)
- `guidance_rescale` is 0.7 at 512 (0.0 only when latent width is 32)
- Dual-reference decoder: `decode_first_stage(..., ref_context=hs)`
- More than 14 in-betweens → extra **real** passes between generated anchors.
  No frame duplication, no crossfade.

All repo-specific inference code lives in
`src/tooncrafter_animator/inference/adapter.py`.

## License

Apache-2.0 for this application. See `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`,
and `src/tooncrafter_animator/vendor/tooncrafter/PROVENANCE.md`.
