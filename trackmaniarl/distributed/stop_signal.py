"""One-way stop notification that cannot inherit a dead child's Python lock."""

from __future__ import annotations

from multiprocessing.context import BaseContext
from time import monotonic, sleep


class ProcessStopSignal:
    """A single parent writes a shared byte; children only read it.

    Unlike ``multiprocessing.Event``, reads and writes acquire no process-shared
    condition lock. A child dying inside a check therefore cannot prevent the
    launcher from notifying the remaining children or reaching its join timeout.
    The flag is monotonic and intentionally has no ``clear`` operation.
    """

    def __init__(self, context: BaseContext) -> None:
        self._flag = context.RawValue("b", 0)
        self._drain = context.RawValue("b", 0)

    def request_drain(self) -> None:
        """Finish already accepted work after the launcher has stopped the actor."""
        self._drain.value = 1

    def drain_requested(self) -> bool:
        return bool(self._drain.value)

    def set(self) -> None:
        self._flag.value = 1

    def is_set(self) -> bool:
        return bool(self._flag.value)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else monotonic() + max(0.0, timeout)
        while not self.is_set():
            remaining = 0.05 if deadline is None else min(0.05, deadline - monotonic())
            if remaining <= 0.0:
                return self.is_set()
            sleep(remaining)
        return True
