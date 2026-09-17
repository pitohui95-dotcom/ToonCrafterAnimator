from __future__ import annotations


class AnimatorError(Exception):
    """Base class for user-facing failures."""


class CheckpointError(AnimatorError):
    pass


class InferenceCancelled(AnimatorError):
    pass


class MissingDependency(AnimatorError):
    pass


class ExportError(AnimatorError):
    pass


class ProjectError(AnimatorError):
    pass
