from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tooncrafter_animator.core.errors import ProjectError
from tooncrafter_animator.core.models import AspectMode, GenerationParams

PROJECT_SCHEMA = 1


@dataclass
class KeyframeRef:
    absolute: str = ""
    relative: str = ""

    def resolved(self, project_dir: Path) -> Path | None:
        if self.relative:
            candidate = (project_dir / self.relative).resolve()
            if candidate.is_file():
                return candidate
        if self.absolute:
            candidate = Path(self.absolute).expanduser()
            if candidate.is_file():
                return candidate
        return None


@dataclass
class ProjectFile:
    schema_version: int = PROJECT_SCHEMA
    start: KeyframeRef = field(default_factory=KeyframeRef)
    end: KeyframeRef = field(default_factory=KeyframeRef)
    generation: GenerationParams = field(default_factory=GenerationParams)
    export_dir: str = ""
    checkpoint_folder: str = ""
    checkpoint_file: str = ""
    clip_file: str = ""
    missing_keyframes: list[str] = field(default_factory=list)


def _rel_to(project_path: Path, image: Path | None) -> KeyframeRef:
    if image is None:
        return KeyframeRef()
    image = image.resolve()
    try:
        relative = os.path.relpath(image, project_path.parent)
    except ValueError:
        relative = ""
    return KeyframeRef(absolute=str(image), relative=relative)


def save_project(
    path: Path,
    start: Path | None,
    end: Path | None,
    generation: GenerationParams,
    export_dir: str = "",
    checkpoint_folder: str = "",
    checkpoint_file: str = "",
    clip_file: str = "",
) -> None:
    path = Path(path)
    payload = {
        "schema_version": PROJECT_SCHEMA,
        "start": asdict(_rel_to(path, start)),
        "end": asdict(_rel_to(path, end)),
        "generation": {
            "output_width": generation.output_width,
            "output_height": generation.output_height,
            "aspect": generation.aspect.value,
            "intermediates": generation.intermediates,
            "fps": generation.fps,
            "seed": generation.seed,
            "steps": generation.steps,
            "cfg_scale": generation.cfg_scale,
            "eta": generation.eta,
            "motion_stride": generation.motion_stride,
            "precision": generation.precision,
            "device": generation.device,
            "prompt": generation.prompt,
            "gen_height": generation.gen_height,
            "gen_width": generation.gen_width,
            "vram_strategy": generation.vram_strategy,
        },
        "export_dir": export_dir,
        "checkpoint_folder": checkpoint_folder,
        "checkpoint_file": checkpoint_file,
        "clip_file": clip_file,
    }
    text = json.dumps(payload, indent=2)
    fd, tmp = tempfile.mkstemp(prefix=".project.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_project(path: Path) -> ProjectFile:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Could not read project file: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProjectError("Project file is not a JSON object.")
    version = int(raw.get("schema_version", 0) or 0)
    if version != PROJECT_SCHEMA:
        raise ProjectError(f"Unsupported project schema {version} (expected {PROJECT_SCHEMA}).")
    start = KeyframeRef(**{k: raw.get("start", {}).get(k, "") for k in ("absolute", "relative")})
    end = KeyframeRef(**{k: raw.get("end", {}).get(k, "") for k in ("absolute", "relative")})
    g = raw.get("generation") or {}
    try:
        aspect = AspectMode(g.get("aspect", "preserve"))
    except ValueError as exc:
        raise ProjectError(f"Unknown aspect mode: {g.get('aspect')!r}") from exc
    generation = GenerationParams(
        output_width=int(g.get("output_width", 512)),
        output_height=int(g.get("output_height", 320)),
        aspect=aspect,
        intermediates=int(g.get("intermediates", 14)),
        fps=int(g.get("fps", 8)),
        seed=int(g.get("seed", 123)),
        steps=int(g.get("steps", 50)),
        cfg_scale=float(g.get("cfg_scale", 7.5)),
        eta=float(g.get("eta", 1.0)),
        motion_stride=int(g.get("motion_stride", 10)),
        precision=str(g.get("precision", "fp32")),
        device=str(g.get("device", "cpu")),
        prompt=str(g.get("prompt", "")),
        gen_height=int(g.get("gen_height", 320)),
        gen_width=int(g.get("gen_width", 512)),
        vram_strategy=str(g.get("vram_strategy", "none")),
    )
    project = ProjectFile(
        schema_version=version,
        start=start,
        end=end,
        generation=generation,
        export_dir=str(raw.get("export_dir", "")),
        checkpoint_folder=str(raw.get("checkpoint_folder", "")),
        checkpoint_file=str(raw.get("checkpoint_file", "")),
        clip_file=str(raw.get("clip_file", "")),
    )
    missing: list[str] = []
    if not start.resolved(path.parent):
        if start.absolute or start.relative:
            missing.append("start")
    if not end.resolved(path.parent):
        if end.absolute or end.relative:
            missing.append("end")
    project.missing_keyframes = missing
    return project
