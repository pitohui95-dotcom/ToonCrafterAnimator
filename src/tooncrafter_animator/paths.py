from __future__ import annotations

import os
import sys
from pathlib import Path

from tooncrafter_animator import APP_NAME


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_root() -> Path:
    """Directory that contains the executable (frozen) or the package (dev)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_root() -> Path:
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return install_root()
    return Path(__file__).resolve().parent


def asset_path(name: str) -> Path:
    return resource_root() / "assets" / name


def _add_unique(paths: list[Path], path: Path | None) -> None:
    if path is None:
        return
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved not in paths:
        paths.append(resolved)


def vendor_search_roots() -> list[Path]:
    """Bundle / install directories that may hold the vendored ToonCrafter tree."""
    roots: list[Path] = []
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            _add_unique(roots, Path(meipass))
        exe_dir = Path(sys.executable).resolve().parent
        _add_unique(roots, exe_dir)
        _add_unique(roots, exe_dir / "_internal")
    _add_unique(roots, resource_root())
    _add_unique(roots, Path(__file__).resolve().parent)
    return roots


def _looks_like_vendor_root(path: Path) -> bool:
    return (path / "ToonCrafter" / "__init__.py").is_file()


def _find_vendor_root() -> Path | None:
    """Locate the directory whose child is the ``ToonCrafter`` package."""
    rels = (
        Path("vendor") / "tooncrafter",
        Path("tooncrafter_animator") / "vendor" / "tooncrafter",
        Path("."),
    )
    for base in vendor_search_roots():
        for rel in rels:
            candidate = base / rel
            if _looks_like_vendor_root(candidate):
                try:
                    return candidate.resolve()
                except OSError:
                    return candidate
        if base.name == "ToonCrafter" and (base / "__init__.py").is_file():
            return base.parent
    return None


def vendor_root() -> Path:
    """Directory that contains the ``ToonCrafter`` package.

    Unfrozen: ``src/tooncrafter_animator/vendor/tooncrafter``.
    Frozen onedir: prefer ``_MEIPASS/vendor/tooncrafter``, then a top-level
    ``_MEIPASS/ToonCrafter`` copy (so ``import ToonCrafter`` works because
    PyInstaller always puts ``_MEIPASS`` on ``sys.path``), then the older
    ``_MEIPASS/tooncrafter_animator/vendor/tooncrafter`` dest.
    """
    found = _find_vendor_root()
    if found is not None:
        return found
    return resource_root() / "vendor" / "tooncrafter"


def vendor_sys_paths() -> list[str]:
    """Parents of ``ToonCrafter`` and of ``lvdm`` that must be on ``sys.path``.

    ``from ToonCrafter.utils.utils import …`` needs the parent of the
    ``ToonCrafter`` directory. ``import lvdm`` needs the ``ToonCrafter``
    directory itself (lvdm lives next to ``ToonCrafter/__init__.py``).
    """
    paths: list[str] = []

    def _add(path: Path) -> None:
        if path.is_dir():
            text = str(path)
            if text not in paths:
                paths.append(text)

    found = _find_vendor_root()
    if found is not None:
        _add(found)
        _add(found / "ToonCrafter")
    for base in vendor_search_roots():
        _add(base)
        _add(base / "vendor" / "tooncrafter")
        _add(base / "vendor" / "tooncrafter" / "ToonCrafter")
        _add(base / "tooncrafter_animator" / "vendor" / "tooncrafter")
        _add(base / "tooncrafter_animator" / "vendor" / "tooncrafter" / "ToonCrafter")
    return paths


def inference_config_path() -> Path:
    return vendor_root() / "ToonCrafter" / "configs" / "inference_512_v1.0.yaml"


def bundled_ffmpeg_dir() -> Path:
    return install_root() / "ffmpeg"


def user_data_dir() -> Path:
    """%LOCALAPPDATA%\\ToonCrafterAnimator on Windows; XDG on Linux."""
    override = os.environ.get("TOONCRAFTER_APPDATA")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / APP_NAME
        return Path.home() / "AppData" / "Local" / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def log_dir() -> Path:
    return user_data_dir() / "logs"
