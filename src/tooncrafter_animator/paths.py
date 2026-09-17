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


def vendor_root() -> Path:
    return resource_root() / "vendor" / "tooncrafter"


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
