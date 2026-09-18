from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "tca_zip_release", ROOT / "packaging" / "zip_release.py"
)
assert _SPEC and _SPEC.loader
_zip_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_zip_release)
split_if_needed = _zip_release.split_if_needed
zip_onedir = _zip_release.zip_onedir


def test_zip_and_split_under_github_limit(tmp_path: Path) -> None:
    src = tmp_path / "onedir"
    src.mkdir()
    (src / "README.txt").write_text("CUDA PyTorch\n", encoding="utf-8")
    (src / "ToonCrafterAnimator.exe").write_bytes(b"MZ" + b"\x00" * 100)
    dest = tmp_path / "out" / "ToonCrafterAnimator-windows-x64-cuda.zip"
    zipped = zip_onedir(src, dest)
    assert zipped.is_file()
    parts = split_if_needed(zipped, max_bytes=10_000_000)
    assert parts == [zipped]


def test_split_over_limit_writes_parts_and_join_txt(tmp_path: Path) -> None:
    blob = tmp_path / "big.zip"
    blob.write_bytes(b"abcdefghij" * 50)
    parts = split_if_needed(blob, max_bytes=120)
    names = [p.name for p in parts]
    assert any(n.endswith(".part01") for n in names)
    assert any(n.endswith(".JOIN.txt") for n in names)
    join = next(p for p in parts if p.name.endswith(".JOIN.txt"))
    text = join.read_text(encoding="utf-8")
    assert "copy /b" in text
    assert "分卷" in text
