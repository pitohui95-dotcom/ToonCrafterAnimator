ToonCrafter 动画器
=================

独立 Windows 应用：用真实的 ToonCrafter 512 插帧模型，在两张卡通关键帧之间生成中间帧。
没有 ComfyUI，也没有替代插值器。

请整文件夹复制使用。双击 ToonCrafterAnimator.exe 即可启动。

首次运行
--------
本应用不会下载权重。你需要自行准备：

  1. ToonCrafter 插帧检查点，任选其一：
       model.ckpt  （约 10 GB，Doubiiu/ToonCrafter）
     或
       tooncrafter_512_interp-fp16.safetensors  （约 5 GB，Kijai/DynamiCrafter_pruned）
  2. OpenCLIP ViT-H-14 / laion2b_s32b_b79k
       open_clip_pytorch_model.bin  （约 4 GB）

在首次运行的设置对话框中，指向包含检查点的文件夹以及 OpenCLIP 文件。
路径保存在 %LOCALAPPDATA%\ToonCrafterAnimator。不会上传任何内容。
也不会执行权重旁边的脚本。

硬件
----
GitHub Actions 发布包捆绑的是 **CPU 版 PyTorch**。要在 NVIDIA GPU 上使用 FP16，
需要自行安装 CUDA 版 PyTorch（源码环境，或本地重新打包 exe）。
CUDA 上的 FP32 大约需要 22 GB 显存。CPU FP32 可以运行，但会非常慢。
CI 生成的 `.exe` 不包含 CUDA。

一次 ToonCrafter 推理固定生成 16 帧（起始 + 14 个中间帧 + 结束）。
「运动幅度 (fs)」是模型的 FPS 条件（5–30，越小运动越大），不是输出帧数。
如果中间帧超过 14 个，会在已生成的锚点之间再跑真实推理。

ffmpeg
------
LGPL 版 ffmpeg 应放在 exe 旁边的 ffmpeg\ 文件夹中。打包脚本会自动获取。
请勿将 GPL 版 ffmpeg 与本应用一起分发。

许可证
------
见 licenses\ 与 THIRD_PARTY_NOTICES.txt。本应用为 Apache-2.0。
Qt/PySide6 为 LGPLv3（动态链接、未修改）。ffmpeg 为 LGPLv2.1+。

模型权重不受代码许可证覆盖。你需要自行接受上游模型页条款。
