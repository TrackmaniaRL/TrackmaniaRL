"""Explicit clock domains for the experiment budget, separate from liveness."""

from __future__ import annotations

import math
import time

SUPPORTED_CLOCK_SOURCES = ("monotonic", "monotonic_raw")


def validate_clock_source(value: object) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_CLOCK_SOURCES:
        raise ValueError("supported explicit campaign clock source required")
    return value


def clock_seconds(source: str) -> float:
    """RAW avoids WSL's observed rate adjustment; never silently fall back."""
    validate_clock_source(source)
    if source == "monotonic_raw":
        clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", None)
        read_clock = getattr(time, "clock_gettime", None)
        if clock_id is None or not callable(read_clock):
            raise RuntimeError("CLOCK_MONOTONIC_RAW is unavailable on this host")
        value = read_clock(clock_id)
    else:
        value = time.monotonic()
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("campaign clock returned an invalid timestamp")
    return float(value)
