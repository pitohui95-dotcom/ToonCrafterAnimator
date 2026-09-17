from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tooncrafter_animator.paths import settings_path, user_data_dir

log = logging.getLogger("tooncrafter")

SETTINGS_SCHEMA = 1


@dataclass
class AppSettings:
    schema_version: int = SETTINGS_SCHEMA
    checkpoint_folder: str = ""
    checkpoint_file: str = ""
    clip_file: str = ""
    use_openclip_cache: bool = False
    device: str = "cpu"
    precision: str = "fp32"
    ffmpeg_path: str = ""
    last_export_dir: str = ""
    last_project_dir: str = ""
    last_output_width: int = 512
    last_output_height: int = 320
    vram_strategy: str = "none"
    accepted_setup: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def default_settings() -> AppSettings:
    return AppSettings()


def load_settings(path: Path | None = None) -> AppSettings:
    path = path or settings_path()
    if not path.is_file():
        return default_settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        backup = path.with_suffix(".json.corrupt")
        try:
            path.replace(backup)
            log.warning("settings_corrupt_backed_up path=%s backup=%s error=%s", path, backup, exc)
        except OSError:
            log.warning("settings_corrupt_unreadable path=%s error=%s", path, exc)
        return default_settings()
    if not isinstance(raw, dict):
        return default_settings()
    version = int(raw.get("schema_version", 0) or 0)
    if version != SETTINGS_SCHEMA:
        log.warning("settings_schema_mismatch got=%s expected=%s; using defaults where needed", version, SETTINGS_SCHEMA)
    data = default_settings().to_json()
    data.update({k: v for k, v in raw.items() if k in data})
    data["schema_version"] = SETTINGS_SCHEMA
    return AppSettings(**data)


def save_settings(settings: AppSettings, path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings.to_json(), indent=2)
    fd, tmp_name = tempfile.mkstemp(prefix=".settings.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
