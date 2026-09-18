"""PyInstaller runtime hook: put vendored ToonCrafter / lvdm on sys.path.

``_MEIPASS`` is already on ``sys.path``. Prepend the vendor copies so
``import ToonCrafter`` and ``import lvdm`` resolve when those packages live
under ``vendor/tooncrafter`` rather than (or in addition to) the bundle root.

Do not import ToonCrafter here — that would pull lvdm/torchvision before the
torchvision ops runtime hook registers ``torchvision::nms``.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _prepend(path: Path) -> None:
    if not path.is_dir():
        return
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def _bootstrap() -> None:
    bases: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bases.append(Path(meipass))
    try:
        exe_dir = Path(sys.executable).resolve().parent
        bases.append(exe_dir)
        bases.append(exe_dir / "_internal")
    except Exception:
        pass

    seen: set[str] = set()
    for base in bases:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        # Parents of the ToonCrafter package (``import ToonCrafter``).
        _prepend(base)
        _prepend(base / "vendor" / "tooncrafter")
        _prepend(base / "tooncrafter_animator" / "vendor" / "tooncrafter")
        # Parents of lvdm (``import lvdm``) — the ToonCrafter directory itself.
        _prepend(base / "ToonCrafter")
        _prepend(base / "vendor" / "tooncrafter" / "ToonCrafter")
        _prepend(base / "tooncrafter_animator" / "vendor" / "tooncrafter" / "ToonCrafter")


_bootstrap()
