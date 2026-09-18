from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tca_ci_build", ROOT / "packaging" / "ci_build.py")
assert _SPEC and _SPEC.loader
_ci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ci)


def test_headers_named_cudnn_do_not_count_as_cuda_natives(tmp_path: Path) -> None:
    fake = tmp_path / "onedir"
    (fake / "torch" / "include").mkdir(parents=True)
    (fake / "torch" / "include" / "cudnn.h").write_text("/* not a GPU pack */\n", encoding="utf-8")
    (fake / "torch" / "_cudnn.pyi").write_text("", encoding="utf-8")
    (fake / "FindCUDNN.cmake").write_text("", encoding="utf-8")
    hits = _ci.cuda_native_hits(list(fake.rglob("*")))
    assert hits == []
    assert _ci.cuda_runtime_complete(hits) is False
    with pytest.raises(SystemExit):
        _ci.verify_cuda_bundle(fake, require_installed_cuda_torch=False)


def test_real_cuda_dlls_pass_runtime_complete(tmp_path: Path) -> None:
    lib = tmp_path / "onedir" / "torch" / "lib"
    lib.mkdir(parents=True)
    for name in (
        "c10_cuda.dll",
        "torch_cuda.dll",
        "cublas64_12.dll",
        "cudnn64_9.dll",
        "cudart64_12.dll",
    ):
        (lib / name).write_bytes(b"MZ")
    hits = _ci.cuda_native_hits(list((tmp_path / "onedir").rglob("*")))
    assert {p.name for p in hits} >= {"c10_cuda.dll", "torch_cuda.dll", "cublas64_12.dll"}
    assert _ci.cuda_runtime_complete(hits) is True
    _ci.verify_cuda_bundle(tmp_path / "onedir", require_installed_cuda_torch=False)


def test_cpu_torch_dlls_alone_are_not_a_gpu_pack(tmp_path: Path) -> None:
    lib = tmp_path / "onedir" / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib / "c10.dll").write_bytes(b"MZ")
    (lib / "torch_cpu.dll").write_bytes(b"MZ")
    hits = _ci.cuda_native_hits(list((tmp_path / "onedir").rglob("*")))
    assert hits == []
    assert _ci.cuda_runtime_complete(hits) is False
