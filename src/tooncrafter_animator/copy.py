"""User-visible Simplified Chinese copy.

Legal / attribution texts (LICENSE, THIRD_PARTY_NOTICES, vendor files) stay
in their original language. Logging and type hints stay English.
"""
from __future__ import annotations

APP_DISPLAY_NAME = "ToonCrafter 动画器"

# --- menus ---
MENU_FILE = "文件(&F)"
MENU_RUN = "运行(&R)"
MENU_HELP = "帮助(&H)"
ACTION_OPEN_PROJECT = "打开项目…"
ACTION_SAVE_PROJECT = "保存项目…"
ACTION_CHECKPOINT_SETUP = "检查点设置…"
ACTION_QUIT = "退出"
ACTION_INTERPOLATE = "插帧"
ACTION_CANCEL = "取消"
ACTION_ABOUT = "关于"

# --- buttons / accessible names ---
BTN_INTERPOLATE = "插帧"
BTN_INTERPOLATE_ACCESSIBLE = "开始插帧"
BTN_SETUP = "检查点设置…"
BTN_SETUP_ACCESSIBLE = "打开检查点设置"
BTN_CHOOSE_START = "选择起始帧…"
BTN_CHOOSE_END = "选择结束帧…"
BTN_SWAP = "交换"
BTN_CLEAR = "清除"
BTN_PLAY = "播放"
BTN_PAUSE = "暂停"
BTN_LOOP_ON = "循环：开"
BTN_LOOP_OFF = "循环：关"
BTN_EXPORT_PNG = "导出 PNG 序列…"
BTN_EXPORT_MP4 = "导出 MP4…"
BTN_EXPORT_GIF = "导出 GIF…"
BTN_OPEN_FOLDER = "打开输出文件夹"
BTN_CANCEL = "取消"
BTN_SHOW_LOGS = "显示日志"
BTN_HIDE_LOGS = "隐藏日志"
BTN_FOLDER = "文件夹…"
BTN_CLIP = "CLIP…"
BTN_CHOOSE_FOLDER = "选择文件夹…"
BTN_CHOOSE_CLIP = "选择 CLIP 文件…"
BTN_VALIDATE = "验证"
BTN_SAVE = "保存"
BTN_DIALOG_CANCEL = "取消"
BTN_OPEN_CKPT_CARD = "打开 ToonCrafter 模型页"
BTN_OPEN_PRUNED_CARD = "打开 FP16 精简模型页"
BTN_OPEN_CLIP_CARD = "打开 OpenCLIP 模型页"

# --- group titles ---
GROUP_KEYFRAMES = "关键帧"
GROUP_GENERATION = "生成参数"
GROUP_PREVIEW = "预览 / 时间轴 / 导出"
GROUP_PROGRESS = "进度"
LABEL_START = "起始帧"
LABEL_END = "结束帧"

# --- form labels ---
LABEL_OUTPUT_WIDTH = "输出宽度"
LABEL_OUTPUT_HEIGHT = "输出高度"
LABEL_ASPECT = "画面比例"
LABEL_INTERMEDIATES = "中间帧数量"
LABEL_FPS = "帧率"
LABEL_SEED = "随机种子"
LABEL_STEPS = "步数"
LABEL_GUIDANCE = "引导强度"
LABEL_ETA = "Eta"
LABEL_MOTION = "运动幅度 (fs)"
LABEL_GENERATE_AT = "生成分辨率"
LABEL_DEVICE = "设备"
LABEL_PRECISION = "精度"
LABEL_VRAM = "显存策略"
LABEL_PROMPT = "提示词"
LABEL_CHECKPOINT_FOLDER = "检查点文件夹"
LABEL_OPENCLIP = "OpenCLIP 权重（open_clip_pytorch_model.bin）"
LABEL_OPENCLIP_SHORT = "OpenCLIP 权重"

ASPECT_PRESERVE = "保持比例（适应并留边）"
ASPECT_CROP = "裁剪（铺满并居中裁切）"
GEN_TRAINED = "320×512（训练分辨率）"
GEN_OFF_256 = "256×256（非训练分布）"
GEN_OFF_320 = "320×320（非训练分布）"
GEN_OFF_1024 = "576×1024（非训练分布，1024 档）"
PRECISION_FP16 = "FP16（CUDA）"
PRECISION_FP32 = "FP32"
VRAM_NONE = "无"
VRAM_LOW = "低显存（TOON_MEM_STRATEGY=low）"
DEVICE_CPU = "CPU"
DEVICE_CPU_NO_TORCH = "CPU（未安装 PyTorch）"

# --- placeholders / empty states ---
DROP_START = "拖入起始关键帧\n或点击浏览"
DROP_END = "拖入结束关键帧\n或点击浏览"
PREVIEW_EMPTY = "完成插帧后，帧预览会显示在这里。"
NO_FRAMES = "暂无帧"
IDLE = "空闲。"
RUNNING = "正在运行…"
FAILED = "失败。"
CANCELLED = "已取消。"
NO_CHECKPOINT = "尚未选择检查点。"
PROMPT_PLACEHOLDER = "可选文本提示（留空符合训练分布）。"
SETUP_STATUS_INITIAL = "选择文件后点击「验证」。"

# --- tooltips ---
TIP_ASPECT = (
    "「保持比例」把 320×512 的生成结果放入你设定的宽×高，并用关键帧边框颜色留边。"
    "「裁剪」铺满宽×高后居中裁掉溢出部分。"
)
TIP_INTERMEDIATES = (
    "要导出的中间帧数量。一次 ToonCrafter 推理会生成 14 个真实中间帧。"
    "需要更多时，会在已生成的锚点之间再跑真实推理——不会复制或混合帧。"
)
TIP_STEPS = "DDIM 去噪步数。上游默认 50，节点上限 60。"
TIP_CFG = "unconditional_guidance_scale。512 插帧配置使用 uncond_type=empty_seq；默认 7.5。"
TIP_MOTION = (
    "这是模型的 FPS / 帧步长条件（ComfyUI 节点误标为 frame_count）。"
    "建议 5–30。数值越小，运动幅度越大。"
)
TIP_GEN_SIZE = (
    "UNet 是卷积结构，其他能被 64 整除的尺寸也能跑，但训练分辨率是 320×512。"
    "其他尺寸会标为非训练分布。"
)
TIP_VRAM = "上游 ToonCrafter 在 VAE 解码时使用的显存策略。不会替换模型本身。"
TIP_SWAP = "交换起始帧和结束帧。"
TIP_CANCEL = "仅在任务运行时可以取消。当前 DDIM 步骤会先跑完。"
TIP_EXPORT_NEED_JOB = "需要先完成一次插帧才能导出。"
TIP_EXPORT_NEED_FOLDER = "先导出一次，才会记住输出文件夹。"
TIP_RUN_OK = "运行一次或多次真实的 ToonCrafter 推理。"
TIP_MODEL_CARD = "会打开浏览器。本应用仍然不会下载任何内容。"
TIP_INTERPOLATE_ACCESSIBLE = "开始插帧"
ACC_START_PREVIEW = "起始关键帧预览"
ACC_END_PREVIEW = "结束关键帧预览"
ACC_CHOOSE_START = "选择起始关键帧"
ACC_CHOOSE_END = "选择结束关键帧"
ACC_SWAP = "交换起始帧和结束帧"
ACC_CLEAR = "清除关键帧"
ACC_PREVIEW = "帧预览"
ACC_TIMELINE = "时间轴"
ACC_PLAY = "播放预览"
ACC_LOOP = "循环播放"
ACC_PROGRESS = "插帧进度"
ACC_CANCEL = "取消插帧"
ACC_LOG = "日志"
ACC_TOGGLE_LOG = "切换日志面板"
ACC_OUTPUT_WIDTH = "输出宽度"
ACC_OUTPUT_HEIGHT = "输出高度"
ACC_ASPECT = "画面比例"
ACC_INTERMEDIATES = "中间帧数量"
ACC_FPS = "播放帧率"
ACC_SEED = "随机种子"
ACC_STEPS = "DDIM 步数"
ACC_CFG = "无分类器引导强度"
ACC_ETA = "DDIM eta"
ACC_MOTION = "运动帧步长"
ACC_GEN_SIZE = "生成分辨率"
ACC_DEVICE = "设备"
ACC_PRECISION = "精度"
ACC_VRAM = "显存策略"
ACC_PROMPT = "文本提示"
ACC_CKPT_FOLDER = "检查点文件夹"
ACC_CHOOSE_CKPT_FOLDER = "选择检查点文件夹"
ACC_CKPT_FILE = "检查点文件"
ACC_OPENCLIP = "OpenCLIP 权重"
ACC_CHOOSE_OPENCLIP = "选择 OpenCLIP 权重"
ACC_USE_CACHE = "使用本机已有的 OpenCLIP 缓存"

# --- notes / blurbs ---
KEYFRAME_NOTE = "生成始终在 320×512（512 插帧检查点）上进行。你设定的输出宽高会在之后缩放。"
OFFDIST_SIZE = "非训练分布尺寸：512 插帧 UNet 的训练分辨率是 320×512。"
NO_CKPT_IN_FOLDER = "该文件夹中没有 .ckpt / .safetensors 文件（会忽略 sketch_encoder.ckpt）。"
USE_OPENCLIP_CACHE = "使用本机已有的 OpenCLIP 缓存（绝不下载）"
SETUP_USE_CACHE = "我已经通过 open_clip 缓存了 OpenCLIP（只搜索本地缓存，绝不下载）"
SETUP_BLURB = (
    "<p><b>本应用不附带模型权重，也不会下载它们。</b></p>"
    "<p>你需要自行准备这两个文件：</p>"
    "<ul>"
    "<li><code>model.ckpt</code>（约 10 GB fp32，Doubiiu/ToonCrafter）或 "
    "<code>tooncrafter_512_interp-fp16.safetensors</code>（约 5 GB，Kijai/DynamiCrafter_pruned）</li>"
    "<li><code>open_clip_pytorch_model.bin</code>（约 4 GB）— OpenCLIP ViT-H-14 / laion2b_s32b_b79k。"
    "没有它，文本/图像编码器会尝试下载权重，而本应用会拒绝下载。</li>"
    "</ul>"
    "<p>把检查点放到你选择的文件夹。路径只保存在本机应用数据中。不会上传任何内容。"
    "也不会执行模型文件夹里的脚本。</p>"
)
ABOUT_BODY = (
    "<p>版本 {version}。真实 ToonCrafter 512 插帧，没有替代插值器。</p>"
    "<p>推理代码来自 AIGODLIKE/ComfyUI-ToonCrafter 与 ToonCrafter/ToonCrafter（Apache-2.0）。"
    "不包含模型权重。</p>"
)

# --- dialog titles ---
TITLE_SETUP = "检查点设置 — ToonCrafter 动画器"
TITLE_CANNOT_RUN = "无法插帧"
TITLE_RUN_FAILED = "插帧失败"
TITLE_SETUP_INCOMPLETE = "设置未完成"
TITLE_MISSING_KEYFRAMES = "缺少关键帧"
TITLE_MP4_FAILED = "MP4 导出失败"
TITLE_GIF_FAILED = "GIF 导出失败"
TITLE_ABOUT = "关于"
TITLE_SAVE_PROJECT = "保存项目"
TITLE_OPEN_PROJECT = "打开项目"
TITLE_CHOOSE_KEYFRAME = "选择关键帧"
TITLE_CHECKPOINT_FOLDER = "检查点文件夹"
TITLE_OPENCLIP = "OpenCLIP 权重"
TITLE_PNG_FOLDER = "PNG 序列文件夹"
TITLE_EXPORT_MP4 = "导出 MP4"
TITLE_EXPORT_GIF = "导出 GIF"

FILTER_PROJECT = "ToonCrafter 项目 (*.json)"
FILTER_IMAGES = "图像 (*.png *.jpg *.jpeg *.webp *.bmp)"
FILTER_OPENCLIP = "OpenCLIP 权重 (*.bin *.safetensors *.pt);;所有文件 (*)"
FILTER_WEIGHTS = "权重 (*.bin *.safetensors *.pt);;所有文件 (*)"
FILTER_MP4 = "MP4 (*.mp4)"
FILTER_GIF = "GIF (*.gif)"

# --- status / progress ---
STATUS_IDLE_PREFIX = "空闲。"
STATUS_CANCELLING = "将在当前 DDIM 步骤结束后取消…"
STATUS_LOADING_CKPT = "正在加载 ToonCrafter 检查点…"
STATUS_SCOUT = "侦察推理 — 正在生成锚点帧…"
STATUS_DONE = "完成。共 {n} 帧。"
STATUS_SAVED = "已保存 {path}"
STATUS_WROTE = "已写入 {path}"
STATUS_WROTE_PNG = "已将 PNG 序列写入 {path}"
STATUS_CHOOSE_FOLDER = "请选择检查点文件夹。"
STATUS_NO_VALID_CKPT = "该文件夹中没有有效的 ToonCrafter 插帧检查点。{reasons}"
STATUS_NO_CANDIDATES = "没有候选文件"
STATUS_CLIP_MISSING = (
    "缺少 OpenCLIP 权重。请选择 open_clip_pytorch_model.bin，或仅在该文件已在磁盘上时勾选缓存。"
    "本应用不会下载它。"
)
STATUS_READY = "就绪。有效检查点：{names}"
STATUS_PASS = "第 {pass_index} / {n_passes} 次推理 — {message}"
STATUS_STEP = "步骤 {step}/{n_steps}"
STATUS_DDIM = "DDIM 步骤 {step}/{n_steps}"
STAGE_READ_YAML = "正在读取 inference_512_v1.0.yaml"
STAGE_INSTANTIATE = "正在实例化 LatentVisualDiffusion（从你的本地文件加载 OpenCLIP）"
STAGE_LOAD_CKPT = "正在加载检查点 {name}"
STAGE_CAST_FP16 = "正在将模型转为 FP16"
STAGE_MOVE = "正在将模型移到 {device}"
STAGE_READY = "检查点已就绪"

MISSING_KEYFRAMES_BODY = (
    "此项目引用的关键帧文件不在磁盘上：{names}。"
    "路径已保留；请在插帧前重新选择文件。"
)

# --- blockers ---
BLOCK_RUNNING = "已有任务正在运行。"
BLOCK_START = "请选择起始关键帧。"
BLOCK_END = "请选择结束关键帧。"
BLOCK_NO_TORCH = "未安装 PyTorch，无法运行真实的 ToonCrafter 推理。"
BLOCK_SELECT_CKPT = "请选择已验证的 ToonCrafter 检查点。"
BLOCK_SELECT_CLIP = "请选择 OpenCLIP 权重，或在把文件放到磁盘后启用本地缓存选项。"
BLOCK_FP16_CUDA = "FP16 仅在 CUDA 上可用。"

# --- checkpoint probe ---
CKPT_MISSING_FILE = "文件不存在。"
CKPT_BAD_EXT = "不支持的扩展名 {suffix}。"
CKPT_BAD_SAFETENSORS = "不是有效的 safetensors 文件：{exc}"
CKPT_EMPTY_HEADER = "safetensors 文件头为空。"
CKPT_NOT_ZIP = "不是基于 zip 的 PyTorch 存档。无法在不加载的情况下验证旧版 pickle 检查点。"
CKPT_NO_PKL = "Zip 存档缺少 data.pkl（不是 PyTorch 检查点）。"
CKPT_NEED_TORCH = "PyTorch 存档结构看起来有效，但没有 PyTorch 就无法读取键名。请安装 torch 后再验证。"
CKPT_MISSING_MARKERS = "不是 ToonCrafter 插帧检查点（缺少 {missing}）。"
CKPT_MISSING_STRIDE = "检查点缺少 fps_embedding / framestride_embed（不是插帧 UNet）。"
CKPT_OK = "看起来是 ToonCrafter 插帧检查点（{size}，{family} 权重）。"
CKPT_UNKNOWN_SIZE = "未知大小"

# --- hardware ---
HW_CPU_SLOW = "在 CPU 上跑 16 帧 3D UNet 会非常慢（50 步 DDIM 时，每次推理可能要几十分钟到数小时）。"
HW_CPU_TORCH_BUILD = (
    "当前 PyTorch 是 CPU 构建（没有 CUDA）。NVIDIA GPU 插帧需要使用捆绑 CUDA 版 PyTorch 的发布包。"
)
HW_CUDA_BUILD_NO_GPU = (
    "本包已捆绑 CUDA 版 PyTorch，但 torch.cuda.is_available() 为 False。"
    "请安装较新的 NVIDIA 驱动并确认 GPU 可用。在此之前只能用 CPU。"
)
HW_NO_TORCH = "请安装 PyTorch 才能运行 ToonCrafter。在此之前「插帧」会保持禁用。"
HW_VRAM_SMALL = "报告显存为 {gb:.1f} GB。FP16、320×512 / 16 帧推理通常需要约 11–13 GB。"
HW_FP16_CUDA_ONLY = "FP16 仅在 CUDA 上提供。PyTorch 没有可用于此计算图的 FP16 CPU 内核。"
HW_UNKNOWN_PRECISION = "未知精度 {precision!r}。"
HW_VRAM_MAY_OOM = (
    "{name} 有 {have:.1f} GB 显存；{precision} 插帧通常需要约 {need:.0f} GB。"
    "任务可能会因显存不足而失败。"
)

# --- export / inference errors shown in UI ---
ERR_FRAME_TOO_SMALL = "画面太小，无法按 yuv420p 编码（宽高必须为偶数）。"
ERR_NO_FFMPEG = "未找到 ffmpeg。请把 LGPL 构建放到应用旁的 ffmpeg/ 目录，或在设置中指定路径。"
ERR_FPS_POSITIVE = "帧率必须为正数。"
ERR_FFMPEG_MP4 = "ffmpeg MP4 编码失败：{stderr}"
ERR_FFMPEG_PALETTEGEN = "ffmpeg palettegen 失败：{stderr}"
ERR_FFMPEG_PALETTEUSE = "ffmpeg paletteuse 失败：{stderr}"
ERR_NO_FRAMES = "没有可导出的帧。"
ERR_NO_TORCH_STACK = (
    "ToonCrafter 推理需要 PyTorch、torchvision 和 omegaconf。"
    "请从 requirements.txt / requirements-torch.txt 安装。"
)
ERR_NO_TORCHVISION_NMS = (
    "缺少 torchvision 的 C++ 算子（torchvision::nms）。"
    "请安装与当前 torch 匹配的 torchvision，或重新打包 exe。"
)
ERR_NO_CLIP_CACHE = (
    "未找到 OpenCLIP 缓存。请在设置对话框中指定 open_clip_pytorch_model.bin — 本应用不会下载它。"
)
ERR_CLIP_REQUIRED = (
    "离线推理需要 OpenCLIP 权重（ViT-H-14 / laion2b_s32b_b79k，open_clip_pytorch_model.bin，约 4 GB）。"
    "本应用从不下载它们。"
)
ERR_MISSING_CONFIG = "缺少推理配置：{path}"
ERR_MODEL_NOT_LOADED = "模型尚未加载。"
ERR_CUDA_UNAVAILABLE = "已请求 CUDA，但 torch.cuda.is_available() 为 False。"
ERR_SCOUT_FRAMES = "侦察推理返回了 {got} 帧，期望 {expected} 帧。"
ERR_CANCELLED = "插帧已取消。"
ERR_NEGATIVE_INTERMEDIATES = "中间帧数量不能为负数。"
ERR_NO_INTERMEDIATES = "未请求中间帧 — 无需生成。"
ERR_COULD_NOT_READ = "无法读取\n{name}"
ERR_PROJECT_READ = "无法读取项目文件：{exc}"
ERR_PROJECT_NOT_OBJECT = "项目文件不是 JSON 对象。"
ERR_PROJECT_SCHEMA = "不支持的项目格式 {version}（期望 {expected}）。"
ERR_PROJECT_ASPECT = "未知的画面比例模式：{aspect!r}"

def plan_single_pass(n: int) -> str:
    return (
        f"将进行 1 次 ToonCrafter 推理（生成 16 帧）。"
        f"导出其中 {n} 个中间帧；每一帧都是模型输出。"
    )


def plan_chained(n_passes: int, k: int) -> str:
    return (
        f"将进行 {n_passes} 次真实 ToonCrafter 推理：先用 1 次侦察推理选取 {k + 1} 个生成锚点，"
        f"再在其间做 {k} 次插帧。不会复制或混合帧。"
    )
