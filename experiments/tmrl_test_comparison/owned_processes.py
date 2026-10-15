"""Bounded closure checks for cached, creation-time-aware process identities."""

import math
import time
from collections.abc import Iterable

import psutil


def wait_for_owned_processes(
    processes: Iterable[psutil.Process], *, timeout: float
) -> list[psutil.Process]:
    """Return surviving original identities without waiting on reused PID handles.

    ``Process.is_running`` compares PID and creation time. Keep the original
    observed Process objects; rebuilding them from bare PIDs would lose ownership.
    AccessDenied is deliberately propagated: an uninspectable live identity must
    block handoff, rather than being interpreted as successful closure.
    """
    if not math.isfinite(timeout) or not 0 <= timeout <= 30:
        raise ValueError("Owned-process closure timeout must be within 0..30 seconds")
    pending = list(processes)
    deadline = time.monotonic() + timeout
    while True:
        surviving = []
        for process in pending:
            try:
                if process.is_running():
                    surviving.append(process)
            except psutil.NoSuchProcess:
                continue
        remaining = deadline - time.monotonic()
        if not surviving or remaining <= 0:
            return surviving
        pending = surviving
        time.sleep(min(0.1, remaining))
