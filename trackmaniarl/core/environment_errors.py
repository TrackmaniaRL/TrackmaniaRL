"""Recoverable observation interruptions, distinct from transport failures."""


class EnvironmentResetError(RuntimeError):
    """A transient race change invalidated the initial observation."""


class EnvironmentPausedError(RuntimeError):
    """Collection must wait for external readiness and reset before continuing."""
