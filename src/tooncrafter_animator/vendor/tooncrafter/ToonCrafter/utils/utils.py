# CHANGED from upstream ToonCrafter/utils/utils.py
# This file was slimmed for standalone inference: cv2, numpy image helpers, and
# torch.distributed process-group setup were removed because they are unused by
# the 512-interp path and would pull extra runtime dependencies. The functions
# kept below (count_params, instantiate_from_config, get_obj_from_str) are
# identical in behaviour to upstream.
#
# Upstream: https://github.com/ToonCrafter/ToonCrafter
# Also vendored via: https://github.com/AIGODLIKE/ComfyUI-ToonCrafter
# See ../PROVENANCE.md (Apache-2.0 §4(b) prominent notice).

import importlib


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
