from __future__ import annotations

import numpy as np
from PIL import Image

from tooncrafter_animator.core.images import (
    NATIVE_H,
    NATIVE_W,
    AspectMode,
    fit_or_cover,
    resize_min_edge_center_crop,
    to_minus_one_one,
)


def test_resize_center_crop_native_size() -> None:
    img = Image.new("RGB", (1920, 1080), (20, 40, 80))
    out = resize_min_edge_center_crop(img, (NATIVE_H, NATIVE_W))
    assert out.size == (NATIVE_W, NATIVE_H)


def test_minus_one_one_formula() -> None:
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[0, 0] = (255, 0, 128)
    chw = to_minus_one_one(arr)
    assert chw.shape == (3, 4, 4)
    # 1.0 -> (1-0.5)*2 = 1.0; 0 -> -1; 128/255 ≈ 0.502 → ~0.004
    assert abs(chw[0, 0, 0] - 1.0) < 1e-5
    assert abs(chw[1, 0, 0] + 1.0) < 1e-5


def test_preserve_fits_inside() -> None:
    frame = np.zeros((320, 512, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    out = fit_or_cover(frame, 400, 400, AspectMode.PRESERVE, fill=(1, 2, 3))
    assert out.shape == (400, 400, 3)
    # corners should be pad colour (native is wider than tall → letterbox)
    assert tuple(out[0, 0]) == (1, 2, 3)


def test_crop_covers() -> None:
    frame = np.zeros((320, 512, 3), dtype=np.uint8)
    frame[:, :] = (9, 9, 9)
    out = fit_or_cover(frame, 256, 256, AspectMode.CROP)
    assert out.shape == (256, 256, 3)
    assert (out == 9).all()
