"""Small raw backend contract, independent of Torch and the graphical client."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

RawState = dict[str, Any]
RawResult = dict[str, Any]


class NativeBackend(Protocol):
    @property
    def capabilities(self) -> dict[str, Any]: ...

    def reset(self) -> RawState: ...

    def step(self, action: int, native_iterations: int = 5) -> RawResult: ...

    def read_state(self) -> RawState: ...

    def close(self) -> None: ...


class NativeExecutionDisabledError(RuntimeError):
    """Native reset requires explicit allow_native=True."""


class NativeLifecycleError(RuntimeError):
    """The caller must reset, or create a fresh environment after failure/close."""


class NativeObservationUnavailableError(ValueError):
    """No actual current feature vector is available; it must not be zero-filled."""


class NativeInconclusiveError(RuntimeError):
    """A raw result cannot be promoted to a terminal, truncation or replay row."""

    def __init__(self, raw_result: RawResult) -> None:
        self.raw_result = deepcopy(raw_result)
        super().__init__("native result is inconclusive; inspect raw_result and retained trace_dir")
