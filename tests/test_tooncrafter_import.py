from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tooncrafter_animator.inference import adapter as adapter_mod
from tooncrafter_animator.inference.adapter import ensure_vendor_on_path
from tooncrafter_animator.paths import inference_config_path, vendor_root, vendor_sys_paths

ROOT = Path(__file__).resolve().parents[1]
VENDOR_SRC = ROOT / "src" / "tooncrafter_animator" / "vendor" / "tooncrafter" / "ToonCrafter"

# Vendor / adapter sites that do `from ToonCrafter` / `import ToonCrafter`.
TOONCRAFTER_IMPORT_SITES = (
    ROOT / "src" / "tooncrafter_animator" / "inference" / "adapter.py",
    VENDOR_SRC / "lvdm" / "basics.py",
    VENDOR_SRC / "lvdm" / "modules" / "encoders" / "condition.py",
    VENDOR_SRC / "lvdm" / "modules" / "networks" / "ae_modules.py",
    VENDOR_SRC / "lvdm" / "models" / "autoencoder.py",
    VENDOR_SRC / "lvdm" / "models" / "ddpm3d.py",
    ROOT / "tests" / "test_lvdm_instantiate.py",
)


def _purge_tooncrafter_modules() -> None:
    for name in list(sys.modules):
        if name == "ToonCrafter" or name.startswith("ToonCrafter.") or name == "lvdm" or name.startswith("lvdm."):
            del sys.modules[name]
    adapter_mod._PATH_READY = False
    importlib.invalidate_caches()


def test_import_sites_use_tooncrafter_package() -> None:
    for path in TOONCRAFTER_IMPORT_SITES:
        text = path.read_text(encoding="utf-8")
        assert "ToonCrafter" in text, path
        assert path.is_file()


def test_import_tooncrafter_without_checkpoint() -> None:
    """Adapter load path: import ToonCrafter + instantiate_from_config, no ckpt."""
    _purge_tooncrafter_modules()
    ensure_vendor_on_path()
    import ToonCrafter
    from ToonCrafter.utils.utils import instantiate_from_config, get_obj_from_str
    import lvdm

    assert Path(ToonCrafter.__file__).is_file()
    assert "vendor" in Path(ToonCrafter.__file__).as_posix()
    assert inference_config_path().is_file()
    assert "temporal_length: 16" in inference_config_path().read_text(encoding="utf-8")
    assert callable(instantiate_from_config)
    assert callable(get_obj_from_str)
    assert lvdm.__file__ is None or Path(lvdm.__file__).is_file() or hasattr(lvdm, "__path__")


def test_import_tooncrafter_in_clean_subprocess() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "from tooncrafter_animator.inference.adapter import ensure_vendor_on_path\n"
        "from tooncrafter_animator.paths import inference_config_path\n"
        "from pathlib import Path\n"
        "ensure_vendor_on_path()\n"
        "import ToonCrafter\n"
        "from ToonCrafter.utils.utils import instantiate_from_config\n"
        "import lvdm\n"
        "assert Path(ToonCrafter.__file__).is_file()\n"
        "assert inference_config_path().is_file()\n"
        "assert instantiate_from_config.__name__ == 'instantiate_from_config'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def _copy_pkg(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        VENDOR_SRC,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )


@pytest.mark.parametrize(
    "layout",
    (
        "vendor/tooncrafter/ToonCrafter",
        "tooncrafter_animator/vendor/tooncrafter/ToonCrafter",
        "ToonCrafter",
    ),
)
def test_frozen_meipass_layouts_resolve_tooncrafter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    """PyInstaller datas dests: vendor tree, legacy dest, or top-level package."""
    pkg = tmp_path / layout
    _copy_pkg(pkg)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    _purge_tooncrafter_modules()
    ensure_vendor_on_path()
    root = vendor_root()
    assert (root / "ToonCrafter" / "__init__.py").is_file()
    assert tmp_path in root.resolve().parents or root.resolve() == tmp_path.resolve()
    import ToonCrafter
    from ToonCrafter.utils.utils import instantiate_from_config

    got = Path(ToonCrafter.__file__).resolve()
    assert tmp_path.resolve() in got.parents
    assert callable(instantiate_from_config)
    assert inference_config_path().is_file()
    assert any(str(tmp_path) in p for p in vendor_sys_paths())
