"""PyInstaller runtime hook: put CUDA torch DLLs on PATH before torch imports.

A CUDA wheel is not enough if cublas/cudnn/c10_cuda sit in ``torch/lib`` or
``nvidia/*/bin`` and Windows cannot load them. Prepend those directories so
``torch.cuda.is_available()`` can become true on an NVIDIA machine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_dir(path: Path) -> None:
    if not path.is_dir():
        return
    text = str(path)
    os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
    add = getattr(os, "add_dll_directory", None)
    if callable(add):
        try:
            add(text)
        except OSError:
            pass


def _bootstrap() -> None:
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    bases: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bases.append(Path(meipass))
    try:
        bases.append(Path(sys.executable).resolve().parent)
        bases.append(Path(sys.executable).resolve().parent / "_internal")
    except Exception:
        pass
    seen: set[str] = set()
    for base in bases:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        _prepend_dir(base)
        _prepend_dir(base / "torch" / "lib")
        _prepend_dir(base / "torch" / "bin")
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            for child in nvidia.iterdir():
                if child.is_dir():
                    _prepend_dir(child / "bin")
                    _prepend_dir(child / "lib")
                    _prepend_dir(child)


_bootstrap()
