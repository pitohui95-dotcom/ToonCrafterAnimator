from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

from tooncrafter_animator.core.models import CheckpointProbe

CHECKPOINT_GLOBS = ("*.ckpt", "*.pt", "*.pth", "*.bin", "*.safetensors")
SKIP_NAMES = {"sketch_encoder.ckpt"}

# Keys taken from LatentVisualDiffusion / the 512-interp UNet.
REQUIRED_MARKERS = (
    "model.diffusion_model",
    "first_stage_model",
    "image_proj_model",
    "embedder",
    "cond_stage_model",
)
STRIDE_MARKERS = ("fps_embedding", "framestride_embed")


def discover_checkpoints(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    found: list[Path] = []
    for pattern in CHECKPOINT_GLOBS:
        found.extend(folder.rglob(pattern))
    unique = []
    seen = set()
    for path in sorted(found):
        if path.name in SKIP_NAMES:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def probe_checkpoint(path: Path) -> CheckpointProbe:
    path = Path(path)
    if not path.is_file():
        return CheckpointProbe(path, False, "File does not exist.", 0, "unknown", "unknown")
    size = path.stat().st_size
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return _probe_safetensors(path, size)
    if suffix in {".ckpt", ".pt", ".pth", ".bin"}:
        return _probe_torch_archive(path, size)
    return CheckpointProbe(path, False, f"Unsupported extension {suffix}.", size, "unknown", "unknown")


def _probe_safetensors(path: Path, size: int) -> CheckpointProbe:
    try:
        header_len, keys, dtypes = _read_safetensors_header(path)
    except Exception as exc:  # noqa: BLE001
        return CheckpointProbe(path, False, f"Not a valid safetensors file: {exc}", size, "safetensors", "unknown")
    if header_len <= 0:
        return CheckpointProbe(path, False, "Empty safetensors header.", size, "safetensors", "unknown")
    return _evaluate_keys(path, size, "safetensors", keys, dtypes)


def _read_safetensors_header(path: Path) -> tuple[int, list[str], dict[str, str]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) < 8:
            raise ValueError("truncated header length")
        (n,) = struct.unpack("<Q", raw_len)
        if n <= 0 or n > 128 * 1024 * 1024:
            raise ValueError(f"implausible header length {n}")
        blob = handle.read(n)
        if len(blob) < n:
            raise ValueError("truncated header JSON")
    meta = json.loads(blob.decode("utf-8"))
    keys = [k for k in meta.keys() if k != "__metadata__"]
    dtypes = {k: str(meta[k].get("dtype", "")) for k in keys if isinstance(meta[k], dict)}
    return n, keys, dtypes


def _probe_torch_archive(path: Path, size: int) -> CheckpointProbe:
    if not zipfile.is_zipfile(path):
        return CheckpointProbe(
            path,
            False,
            "Not a zip-based PyTorch archive. Legacy pickle checkpoints cannot be validated without loading.",
            size,
            "ckpt",
            "unknown",
        )
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    if not any(name.endswith("data.pkl") for name in names):
        return CheckpointProbe(path, False, "Zip archive is missing data.pkl (not a PyTorch checkpoint).", size, "ckpt", "unknown")
    keys, dtypes = _torch_keys(path)
    if keys is None:
        return CheckpointProbe(
            path,
            False,
            "PyTorch archive structure looks valid, but key names cannot be read without PyTorch. Install torch to validate.",
            size,
            "ckpt",
            "unknown",
        )
    return _evaluate_keys(path, size, "ckpt", keys, dtypes)


def _torch_keys(path: Path) -> tuple[list[str] | None, dict[str, str]]:
    try:
        import torch
    except ImportError:
        return None, {}
    try:
        kwargs = {"map_location": "cpu", "weights_only": True, "mmap": True}
        try:
            obj = torch.load(str(path), **kwargs)
        except TypeError:
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        try:
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            return None, {}
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        return None, {}
    keys = [str(k) for k in obj.keys()]
    dtypes: dict[str, str] = {}
    for key, value in obj.items():
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            dtypes[str(key)] = str(dtype).replace("torch.", "")
    return keys, dtypes


def _dtype_family(dtypes: dict[str, str]) -> str:
    joined = " ".join(dtypes.values()).lower()
    has_f16 = any(tag in joined for tag in ("float16", "fp16", "f16", "half"))
    has_f32 = any(tag in joined for tag in ("float32", "fp32", "f32"))
    if has_f16 and has_f32:
        return "mixed"
    if has_f16:
        return "fp16"
    if has_f32:
        return "fp32"
    return "unknown"


def _evaluate_keys(
    path: Path,
    size: int,
    fmt: str,
    keys: list[str],
    dtypes: dict[str, str],
) -> CheckpointProbe:
    blob = "\n".join(keys)
    hits = [marker for marker in REQUIRED_MARKERS if marker in blob]
    stride_hits = [marker for marker in STRIDE_MARKERS if marker in blob]
    missing = [m for m in REQUIRED_MARKERS if m not in hits]
    family = _dtype_family(dtypes)
    if missing:
        return CheckpointProbe(
            path,
            False,
            "Not a ToonCrafter interpolation checkpoint (missing "
            + ", ".join(missing)
            + ").",
            size,
            fmt,
            family,
            keys_checked=len(keys),
            marker_hits=tuple(hits + stride_hits),
        )
    if not stride_hits:
        return CheckpointProbe(
            path,
            False,
            "Checkpoint is missing fps_embedding / framestride_embed (not the interp UNet).",
            size,
            fmt,
            family,
            keys_checked=len(keys),
            marker_hits=tuple(hits),
        )
    size_note = f"{size / (1024**3):.1f} GB" if size else "unknown size"
    return CheckpointProbe(
        path,
        True,
        f"Looks like a ToonCrafter interpolation checkpoint ({size_note}, {family or 'unknown'} weights).",
        size,
        fmt,
        family,
        keys_checked=len(keys),
        marker_hits=tuple(hits + stride_hits),
    )
