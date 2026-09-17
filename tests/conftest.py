from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TOONCRAFTER_APPDATA", os.environ.get("TOONCRAFTER_APPDATA", ""))
