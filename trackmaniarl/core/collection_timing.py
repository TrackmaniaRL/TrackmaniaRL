"""Move cyclic garbage collection out of races bounded by rollout resets."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def defer_rollout_gc(*, reset_at_rollout_end: bool) -> Iterator[None]:
    """Collect before the new race and restore the caller's GC state on every exit."""
    enabled = reset_at_rollout_end and gc.isenabled()
    if enabled:
        gc.collect()
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
