from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tooncrafter_animator.core.models import AspectMode

NATIVE_H, NATIVE_W = 320, 512


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def array_to_image(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def resize_min_edge_center_crop(image: Image.Image, size_hw: tuple[int, int]) -> Image.Image:
    """Match torchvision Resize(min(H,W)) + CenterCrop((H,W)) on a PIL RGB image.

    torchvision Resize(int) matches the shorter edge; interpolation is bilinear.
    CenterCrop pads if the image is smaller than the crop window (zeros).
    """
    target_h, target_w = size_hw
    short = min(target_h, target_w)
    width, height = image.size
    if min(width, height) == 0:
        raise ValueError("Image has a zero dimension.")
    scale = short / min(width, height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return _center_crop_or_pad(resized, target_w, target_h, fill=(0, 0, 0))


def _center_crop_or_pad(image: Image.Image, crop_w: int, crop_h: int, fill: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    if width < crop_w or height < crop_h:
        canvas = Image.new("RGB", (max(width, crop_w), max(height, crop_h)), fill)
        canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
        image = canvas
        width, height = image.size
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return image.crop((left, top, left + crop_w, top + crop_h))


def to_minus_one_one(hwc_uint8: np.ndarray) -> np.ndarray:
    """(x / 255? no) upstream uses tensor in [0,1] then (x - 0.5) * 2.

    Callers that already have [0,1] float should use that path. This helper
    accepts uint8 HWC and returns float32 CHW in [-1, 1].
    """
    x = hwc_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) * 2.0
    return np.transpose(x, (2, 0, 1))


def from_minus_one_one(chw: np.ndarray) -> np.ndarray:
    x = np.clip(chw.astype(np.float32), -1.0, 1.0)
    x = (x + 1.0) * 0.5
    x = np.transpose(x, (1, 2, 0))
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def sample_border_color(image: Image.Image | np.ndarray) -> tuple[int, int, int]:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return (0, 0, 0)
    top = arr[0, :, :3]
    bottom = arr[-1, :, :3]
    left = arr[:, 0, :3]
    right = arr[:, -1, :3]
    border = np.concatenate([top, bottom, left, right], axis=0)
    mean = border.mean(axis=0)
    return tuple(int(round(c)) for c in mean)  # type: ignore[return-value]


def fit_or_cover(
    frame: np.ndarray,
    width: int,
    height: int,
    mode: AspectMode,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Post-resize a generated frame to the user's output W×H."""
    image = array_to_image(frame)
    if width <= 0 or height <= 0:
        raise ValueError("Output size must be positive.")
    src_w, src_h = image.size
    if mode is AspectMode.CROP:
        scale = max(width / src_w, height / src_h)
        new_w = max(1, round(src_w * scale))
        new_h = max(1, round(src_h * scale))
        scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.asarray(_center_crop_or_pad(scaled, width, height, fill), dtype=np.uint8)
    # preserve: fit inside, pad
    scale = min(width / src_w, height / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(scaled, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def preprocess_keyframe(arr: np.ndarray, size_hw: tuple[int, int] = (NATIVE_H, NATIVE_W)) -> np.ndarray:
    """uint8 HWC → uint8 HWC at generation resolution (Resize+CenterCrop)."""
    image = array_to_image(arr)
    cropped = resize_min_edge_center_crop(image, size_hw)
    return np.asarray(cropped, dtype=np.uint8)
