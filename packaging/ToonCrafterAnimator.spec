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

hidden += [
    "tooncrafter_animator",
    "tooncrafter_animator.ui",
    "tooncrafter_animator.inference.adapter",
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
datas += [(str(VENDOR), "tooncrafter_animator/vendor")]

binaries = []
try:
    binaries += collect_dynamic_libs("torch")
except Exception:
    pass
try:
    binaries += collect_dynamic_libs("torchvision")
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
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "pyi_rth_torchvision_ops.py")],
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
