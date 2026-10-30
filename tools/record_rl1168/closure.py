"""Checks for the native supervisor receipt, distinct from campaign closure."""


def native_closure_clean(closure):
    """Require a normal driver exit and no forced pidfd signals or cleanup errors."""
    return (
        isinstance(closure, dict)
        and type(closure.get("returncode")) is int
        and closure["returncode"] == 0
        and all(
            isinstance(closure.get(key), list) and not closure[key]
            for key in ("remainingOwned", "cleanupErrors", "cleanupPidfdSignals")
        )
        and "supervisorError" in closure
        and closure["supervisorError"] is None
    )
