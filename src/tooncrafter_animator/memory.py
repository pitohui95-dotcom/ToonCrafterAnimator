from __future__ import annotations

import gc
import logging

log = logging.getLogger("tooncrafter")


def release_torch_memory() -> None:
    """Drop cached CUDA blocks after a job. Safe to call when torch is absent."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("cuda_empty_cache_failed error=%s", exc)
