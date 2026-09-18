# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec for ToonCrafter Animator.

Run on Windows (Python 3.10 recommended) from the repo root:

    python -m PyInstaller --noconfirm --clean packaging/ToonCrafterAnimator.spec

Onefile is opt-in only: torch + CUDA DLLs unpack several GB to %TEMP% on every
launch, which is the failure mode this spec exists to avoid.

    python -m PyInstaller --noconfirm --clean packaging/ToonCrafterAnimator.spec --onefile
is NOT wired; pass --onefile on the PyInstaller CLI yourself if you insist.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules

# SPECPATH is the directory containing this spec file (not the spec path itself).
_spec_dir = Path(SPECPATH).resolve()
if _spec_dir.is_file():
    _spec_dir = _spec_dir.parent
ROOT = _spec_dir.parent
SRC = ROOT / "src"
ASSETS = SRC / "tooncrafter_animator" / "assets"
VENDOR = SRC / "tooncrafter_animator" / "vendor"
TOONCRAFTER_ROOT = VENDOR / "tooncrafter"
TOONCRAFTER_PKG = TOONCRAFTER_ROOT / "ToonCrafter"
LVDM_PKG = TOONCRAFTER_PKG / "lvdm"

# Analysis-time path so collect_submodules("ToonCrafter") / ("lvdm") can see
# the vendored tree. Frozen runtime path is handled by pyi_rth_tooncrafter.py
# plus datas dests below (``_MEIPASS`` is always on sys.path).
sys.path.insert(0, str(TOONCRAFTER_ROOT))
sys.path.insert(0, str(TOONCRAFTER_PKG))


def _hidden_from_tree(pkg_dir: Path, pkg_name: str) -> list[str]:
    """Every ``pkg.sub.module`` under *pkg_dir*, even if collect_submodules misses it."""
    names = [pkg_name]
    if not pkg_dir.is_dir():
        return names
    for py in pkg_dir.rglob("*.py"):
        rel = py.relative_to(pkg_dir)
        parts = list(rel.parts)
        if any(part == "__pycache__" or part.startswith(".") for part in parts):
            continue
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = Path(parts[-1]).stem
        if not parts:
            continue
        names.append(pkg_name + "." + ".".join(parts))
    return names


hidden = []
for pkg in (
    "open_clip",
    "transformers",
    "kornia",
    "pytorch_lightning",
    "lightning_fabric",
    "omegaconf",
    "safetensors",
    "einops",
    "tqdm",
    "PIL",
    "numpy",
    "lvdm",
    "ToonCrafter",
    "torchvision",
    "torchvision.ops",
    "torchvision.transforms",
):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        hidden.append(pkg)

hidden += _hidden_from_tree(TOONCRAFTER_PKG, "ToonCrafter")
hidden += _hidden_from_tree(LVDM_PKG, "lvdm")

hidden += [
    "tooncrafter_animator",
    "tooncrafter_animator.ui",
    "tooncrafter_animator.inference.adapter",
    "ToonCrafter",
    "ToonCrafter.utils",
    "ToonCrafter.utils.utils",
    "lvdm",
    "lvdm.basics",
    "lvdm.common",
    "lvdm.distributions",
    "lvdm.ema",
    "lvdm.models.autoencoder",
    "lvdm.models.autoencoder_dualref",
    "lvdm.models.ddpm3d",
    "lvdm.models.utils_diffusion",
    "lvdm.models.samplers.ddim",
    "lvdm.models.samplers.ddim_multiplecond",
    "lvdm.modules.attention",
    "lvdm.modules.attention_svd",
    "lvdm.modules.x_transformer",
    "lvdm.modules.encoders.condition",
    "lvdm.modules.encoders.resampler",
    "lvdm.modules.networks.ae_modules",
    "lvdm.modules.networks.openaimodel3d",
    "torch",
    "torch.cuda",
    "torch.cuda.amp",
    "torch.backends.cuda",
    "torch.backends.cudnn",
    "torchvision",
    "torchvision.ops",
    "torchvision.ops.boxes",
    "torchvision.ops.nms",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "torchvision.extension",
    "torchvision._C",
    "torchvision._C_stable",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

datas = []
datas += collect_data_files("open_clip")
datas += collect_data_files("transformers")
datas += collect_data_files("omegaconf")
datas += [(str(ASSETS), "tooncrafter_animator/assets")]
# Match vendor_root(): _MEIPASS/vendor/tooncrafter/ToonCrafter/...
datas += [(str(VENDOR), "vendor")]
# Top-level packages on _MEIPASS (always on frozen sys.path).
datas += [(str(TOONCRAFTER_PKG), "ToonCrafter")]
datas += [(str(LVDM_PKG), "lvdm")]

binaries = []
try:
    binaries += collect_dynamic_libs("torch")
except Exception:
    pass
try:
    binaries += collect_dynamic_libs("torchvision")
except Exception:
    pass

# CUDA wheels put cublas/cudnn/nvrtc/c10_cuda next to torch (and sometimes
# under site-packages/nvidia/). Collect them so the frozen EXE can actually
# call torch.cuda on an NVIDIA GPU — CPU-only torch must not be labeled GPU.
try:
    torch_datas, torch_binaries, torch_hidden = collect_all("torch")
    datas += torch_datas
    binaries += torch_binaries
    hidden += torch_hidden
except Exception:
    pass


def _add_native_tree(directory: Path, dest_root: str) -> None:
    if not directory.is_dir():
        return
    native_ext = {".dll", ".pyd", ".so", ".dylib"}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in native_ext:
            rel = path.parent.relative_to(directory)
            dest = dest_root if str(rel) == "." else f"{dest_root}/{rel.as_posix()}"
            binaries.append((str(path), dest))


try:
    import torch as _torch

    _torch_dir = Path(_torch.__file__).resolve().parent
    _add_native_tree(_torch_dir / "lib", "torch/lib")
    _add_native_tree(_torch_dir / "bin", "torch/bin")
    _add_native_tree(_torch_dir / "lib" / "nvidia", "torch/lib/nvidia")
except Exception:
    pass

try:
    import site as _site

    _sp_list = []
    try:
        _sp_list.extend(_site.getsitepackages())
    except Exception:
        pass
    if getattr(_site, "USER_SITE", None):
        _sp_list.append(_site.USER_SITE)
    seen_nvidia = set()
    for _sp in _sp_list:
        _nvidia = Path(_sp) / "nvidia"
        key = str(_nvidia.resolve()) if _nvidia.exists() else ""
        if key and key not in seen_nvidia:
            seen_nvidia.add(key)
            _add_native_tree(_nvidia, "nvidia")
except Exception:
    pass

# torchvision 0.19+ registers nms from the native ``_C_stable`` / ``image_stable``
# extension, not the older ``_C.pyd`` name. collect_all plus an explicit glob so
# PyInstaller cannot drop torchvision::nms.
try:
    tv_datas, tv_binaries, tv_hidden = collect_all("torchvision")
    datas += tv_datas
    binaries += tv_binaries
    hidden += tv_hidden
except Exception:
    pass

try:
    import torchvision as _tv

    _tv_dir = Path(_tv.__file__).resolve().parent
    for _pat in ("_C*", "image_stable*", "*.pyd", "*.dll", "*.so"):
        for _path in _tv_dir.glob(_pat):
            if _path.is_file():
                binaries.append((str(_path), "torchvision"))
except Exception:
    pass

excludes = [
    "gradio",
    "matplotlib",
    "decord",
    "pandas",
    "tkinter",
    "pytest",
    "IPython",
    "moviepy",
    "cv2",
    "scipy.spatial.ckdtree",
]

block_cipher = None

a = Analysis(
    [str(SRC / "tooncrafter_animator" / "__main__.py")],
    pathex=[str(SRC), str(TOONCRAFTER_ROOT), str(TOONCRAFTER_PKG)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(ROOT / "packaging" / "pyi_rth_tooncrafter.py"),
        str(ROOT / "packaging" / "pyi_rth_torch_cuda.py"),
        str(ROOT / "packaging" / "pyi_rth_torchvision_ops.py"),
    ],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon = str(ASSETS / "icon.ico") if (ASSETS / "icon.ico").is_file() else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ToonCrafterAnimator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ToonCrafterAnimator",
)
