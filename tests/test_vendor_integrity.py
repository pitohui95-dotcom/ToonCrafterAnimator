from __future__ import annotations

from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "src" / "tooncrafter_animator" / "vendor" / "tooncrafter"
FORBIDDEN = ("from comfy", "import comfy", "hf_hub_download", "huggingface_hub")


def test_no_comfy_or_hub_in_vendor() -> None:
    hits = []
    for path in VENDOR.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in FORBIDDEN:
            if token in text:
                hits.append(f"{path.relative_to(VENDOR)}: {token}")
    assert hits == []


def test_provenance_lists_modified_utils() -> None:
    text = (VENDOR / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "utils.py" in text
    assert "Apache-2.0" in text
    assert "96024189ecb2bcc7014a439b3c8676108cc26738" in text


def test_licenses_present() -> None:
    assert (VENDOR / "LICENSE").is_file()
    assert (VENDOR / "ToonCrafter" / "LICENSE").is_file()
    yaml = VENDOR / "ToonCrafter" / "configs" / "inference_512_v1.0.yaml"
    assert yaml.is_file()
    raw = yaml.read_text(encoding="utf-8")
    assert "LatentVisualDiffusion" in raw
    assert "temporal_length: 16" in raw
    assert "uncond_type: 'empty_seq'" in raw
