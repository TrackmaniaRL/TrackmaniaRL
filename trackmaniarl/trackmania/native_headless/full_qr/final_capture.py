"""Actual collector source witness; EOF alone never supplies final coverage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from struct import unpack_from

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os
from trackmaniarl.trackmania.native_headless.full_qr.contact_delivery import _snapshot
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import (
    NativeGoalContext,
    decode_goal_facts,
)


@dataclass(frozen=True, slots=True)
class FinalCollectorBoundary:
    native_t: int
    previous_native_t: int
    goal_native_t: int
    sequence: int
    source_result_sequence: int


def read_final_stream(path: Path, record_size: int, maximum: int) -> bytes:
    """Stable bounded native output, independent of a process memory reader."""
    descriptor, before = _snapshot(path, record_size, maximum)
    try:
        raw = linux_os.pread(descriptor, before.st_size, 0)
        after = os.fstat(descriptor)
        if len(raw) != before.st_size or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("final source stream changed during completed capture read")
        return raw
    finally:
        os.close(descriptor)


def decode_final_collector_boundary(
    raw: bytes, goal_facts_raw: bytes, *, context: NativeGoalContext
) -> FinalCollectorBoundary:
    """Validate the real completed native callback and its paired writer.

    This does not supply negative motor coverage or assert that future
    physics exists. Consumers must join the actual physical/contact streams.
    """
    if type(raw) is not bytes or len(raw) != 896 or raw[:8] != b"TMFN0001":
        raise ValueError("exact completed native collector witness required")
    meta = context.meta
    sequence = unpack_from("<Q", raw, 24)[0]
    clock, previous = unpack_from("<II", raw, 32)
    goals, event = unpack_from("<II", raw, 72)
    result_sequence = unpack_from("<Q", raw, 80)[0]
    mask, phase = unpack_from("<II", raw, 104)
    if (
        unpack_from("<4I", raw, 8) != (896, 1, context.pid, context.pid)
        or not 1 <= sequence <= 5000
        or not 2200 <= previous < clock <= 152200
        or clock % 10
        or clock != previous + 10
        or unpack_from("<4Q", raw, 40)
        != (meta["sim"], meta["controller"], meta["participant"], meta["vehicle"])
        or goals != 1
        or not 2200 <= event <= 152200
        or not 1 <= result_sequence <= 200000
        or unpack_from("<2Q", raw, 88) != (meta["module_slide"], meta["module_slide"] + 0x11834F2)
        or mask & 0x47FF != 0x47FF
        or phase != 1
        or raw[112:128] != bytes(16)
    ):
        raise ValueError("collector witness identity, physical clock or completion mismatch")
    result = raw[128:]
    if (
        result[:8] != b"TMRESULT"
        or unpack_from("<4I", result, 8) != (768, 1, context.pid, context.pid)
        or unpack_from("<Q", result, 24)[0] != result_sequence
        or unpack_from("<2Q", result, 40) != (meta["controller"], meta["sim"])
        or unpack_from("<Q", result, 80)[0] != meta["participant"]
        or unpack_from("<I", result, 128)[0] != clock
        or unpack_from("<I", result, 148)[0] != mask
        or unpack_from("<Q", result, 232)[0] != meta["module_slide"] + 0x11834F3
        or unpack_from("<I", result, 240)[0] != 13
        or unpack_from("<I", result, 248)[0] != 1
    ):
        raise ValueError("actual collector primary record does not match its witness")
    facts = decode_goal_facts(goal_facts_raw, context)
    endpoints = tuple(fact for fact in facts if fact.endpoint_requested)
    if (
        len(endpoints) != 1
        or endpoints[0].event_native_t != event
        or endpoints[0].callback_native_t > clock
        or endpoints[0].dispatch_native_t > clock
    ):
        raise ValueError("complete matching final Goal writer pair required")
    return FinalCollectorBoundary(clock, previous, event, sequence, result_sequence)
