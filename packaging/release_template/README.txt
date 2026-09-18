ToonCrafter Animator
====================

Standalone Windows app that interpolates two cartoon keyframes with the real
ToonCrafter 512-interp model. No ComfyUI. No substitute interpolator.

This folder is meant to be copied as-is. Double-click ToonCrafterAnimator.exe.

First run
---------
The app will not download weights. You need:

  1. A ToonCrafter interpolation checkpoint, either
       model.ckpt  (~10 GB, Doubiiu/ToonCrafter)
     or
       tooncrafter_512_interp-fp16.safetensors  (~5 GB, Kijai/DynamiCrafter_pruned)
  2. OpenCLIP ViT-H-14 / laion2b_s32b_b79k
       open_clip_pytorch_model.bin  (~4 GB)

Point the first-run setup dialog at the folder that contains the checkpoint and
at the OpenCLIP file. Paths are stored in %LOCALAPPDATA%\ToonCrafterAnimator.
Nothing is uploaded. Scripts sitting next to the weights are never executed.

Hardware
--------
The published GitHub Actions build bundles **CPU PyTorch**. FP16 on an NVIDIA
GPU needs a CUDA PyTorch install (source checkout, or a local rebuild of the
exe). FP32 on CUDA needs ~22 GB. CPU FP32 works and is extremely slow. CUDA is
not bundled in the CI `.exe`.

One ToonCrafter pass always produces 16 frames (start + 14 in-betweens + end).
The "Motion (frame stride)" control is the model's FPS condition (5–30, smaller
= more motion). It is not the output frame count. Asking for more than 14
in-betweens runs additional real passes between generated anchors.

ffmpeg
------
An LGPL ffmpeg build belongs in the ffmpeg\ folder next to the exe. The
Windows packager fetches it. GPL ffmpeg must not be shipped with this app.

Licenses
--------
See licenses\ and THIRD_PARTY_NOTICES.txt. This app is Apache-2.0. Qt/PySide6
is LGPLv3 (dynamically linked, unmodified). ffmpeg is LGPLv2.1+.

Model weights are not covered by the code license. You accept the upstream
model-card terms yourself.
