from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tooncrafter_animator.inference.adapter import ensure_torchvision_ops


def test_torchvision_nms_operator_is_registered() -> None:
    """The inference stack must register torchvision::nms (used via torchvision.ops).

    Skip when torch/torchvision are not installed. Fail if they are present but
    the C++ op is missing — that is the packaged-EXE failure mode.
    """
    ensure_torchvision_ops()
    assert hasattr(torch.ops, "torchvision")
    assert hasattr(torch.ops.torchvision, "nms")
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1]])
    scores = torch.tensor([0.9, 0.8])
    keep = torch.ops.torchvision.nms(boxes, scores, 0.5)
    assert keep.numel() >= 1
    from torchvision.ops import nms as tv_nms

    keep2 = tv_nms(boxes, scores, 0.5)
    assert keep2.numel() >= 1
