from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from tooncrafter_animator.paths import log_dir

LOG_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for key, value in record.__dict__.items():
            if key in logging.LogRecord("", 0, "", 0, "", (), None).__dict__:
                continue
            if key.startswith("_"):
                continue
            extras.append(f"{key}={value!r}")
        if extras:
            return base + " " + " ".join(extras)
        return base


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("tooncrafter")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    directory = log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = log_path or (directory / "animator.log")
    handler = logging.handlers.RotatingFileHandler(
        target, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(KeyValueFormatter(LOG_FORMAT))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(KeyValueFormatter(LOG_FORMAT))
    logger.addHandler(stream)
    logger.debug("logging_ready file=%s", target)
    return logger
