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

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

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
):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        hidden.append(pkg)

hidden += [
    "tooncrafter_animator",
    "tooncrafter_animator.ui",
    "tooncrafter_animator.inference.adapter",
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
    runtime_hooks=[],
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
