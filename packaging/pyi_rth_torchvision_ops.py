"""PyInstaller runtime hook: register torchvision C++ ops before inference.

``torchvision::nms`` is provided by the native ``_C_stable`` / ``_C``
extension. Importing ``torchvision.ops`` loads it. This must run in the
frozen app before ToonCrafter/OpenCLIP touch torchvision transforms.
"""


def _register() -> None:
    try:
        import torchvision  # noqa: F401
        import torchvision.extension  # noqa: F401
        import torchvision.ops  # noqa: F401
        from torchvision.ops import nms  # noqa: F401
    except Exception:
        return


_register()
