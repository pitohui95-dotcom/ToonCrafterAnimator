from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="tooncrafter-animator")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Headless boot check: settings, ffmpeg, assets. Prints JSON and exits.",
    )
    parser.add_argument("--project", type=str, default="", help="Open a project JSON on launch.")
    args, unknown = parser.parse_known_args(argv)
    if unknown and not args.selftest:
        # Qt may receive -platform etc.
        pass

    if args.selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
        return _selftest()

    from tooncrafter_animator.ui.main_window import run_app

    return run_app(project_path=args.project or None)


def _selftest() -> int:
    report: dict[str, Any] = {"ok": False}
    try:
        from tooncrafter_animator import APP_DISPLAY_NAME, __version__
        from tooncrafter_animator.core import ffmpeg as ffmpeg_mod
        from tooncrafter_animator.hardware import list_devices, torch_available
        from tooncrafter_animator.inference.adapter import ensure_vendor_on_path
        from tooncrafter_animator.paths import asset_path, bundled_ffmpeg_dir, inference_config_path, user_data_dir
        from tooncrafter_animator.settings import load_settings

        settings = load_settings()
        ffmpeg = ffmpeg_mod.resolve_ffmpeg(settings.ffmpeg_path)
        icon = asset_path("icon.png")
        ensure_vendor_on_path()
        import ToonCrafter  # noqa: F401 — adapter load path; no checkpoint
        from ToonCrafter.utils.utils import instantiate_from_config

        yaml_path = inference_config_path()
        if not yaml_path.is_file():
            raise RuntimeError(f"ToonCrafter inference yaml missing: {yaml_path}")
        report.update(
            {
                "app": APP_DISPLAY_NAME,
                "version": __version__,
                "user_data_dir": str(user_data_dir()),
                "settings_ok": True,
                "ffmpeg": str(ffmpeg) if ffmpeg else None,
                "ffmpeg_ok": ffmpeg is not None,
                "icon_ok": icon.is_file(),
                "bundled_ffmpeg_dir": str(bundled_ffmpeg_dir()),
                "torch": torch_available(),
                "devices": [d.to_dict() for d in list_devices()],
                "checkpoint_folder": settings.checkpoint_folder,
                "has_checkpoint": bool(settings.checkpoint_file),
                "tooncrafter_ok": True,
                "tooncrafter_file": getattr(ToonCrafter, "__file__", None),
                "inference_yaml_ok": yaml_path.is_file(),
                "instantiate_from_config": instantiate_from_config.__name__,
            }
        )
        if torch_available():
            from tooncrafter_animator.hardware import torch_cuda_build
            from tooncrafter_animator.inference.adapter import ensure_torchvision_ops

            ensure_torchvision_ops()
            import torch

            report["torchvision_nms"] = bool(
                getattr(getattr(torch.ops, "torchvision", None), "nms", None) is not None
            )
            if not report["torchvision_nms"]:
                raise RuntimeError("torchvision::nms is not registered")
            report["torch_cuda_build"] = torch_cuda_build()
            report["torch_version_cuda"] = getattr(torch.version, "cuda", None)
            report["cuda_available"] = bool(torch.cuda.is_available())
        else:
            report["torchvision_nms"] = None
            report["torch_cuda_build"] = False
            report["torch_version_cuda"] = None
            report["cuda_available"] = False
        # Import UI offscreen to prove widgets construct.
        from PySide6.QtWidgets import QApplication

        from tooncrafter_animator.ui.main_window import MainWindow
        from tooncrafter_animator.ui.theme import apply_theme

        app = QApplication.instance() or QApplication(["tooncrafter-animator-selftest"])
        apply_theme(app)
        win = MainWindow(settings=settings, show_setup=False)
        win.resize(1280, 800)
        report["window"] = {"width": win.width(), "height": win.height(), "title": win.windowTitle()}
        win.close()
        report["ok"] = True
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 — selftest must always emit JSON
        report["ok"] = False
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
