"""Native classifier-derived Goal facts, distinct from client UI qualification."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from struct import unpack_from

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os
from trackmaniarl.trackmania.native_headless.full_qr.contact_delivery import _snapshot
from trackmaniarl.trackmania.native_headless.full_qr.waypoint_capture import decode_writer_pairs


@dataclass(frozen=True, slots=True)
class NativeGoalContext:
    pid: int
    meta: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class NativeWaypointFact:
    callback_native_t: int
    event_native_t: int
    waypoint_id: int
    endpoint_requested: bool
    entry_name: str
    entry_bytes: bytes
    dispatch_native_t: int = 0


@dataclass(frozen=True, slots=True)
class NativeGoalBoundary:
    native_t: int
    pairs: int
    goals: int
    last_goal_native_t: int | None
    goal_active: bool
    # No inference from nativeGoal to client UI is made by this decoder.
    client_ui_finish_proven: bool = False


def decode_goal_facts(raw: bytes, context: NativeGoalContext) -> tuple[NativeWaypointFact, ...]:
    if type(raw) is not bytes or len(raw) % 1536 or len(raw) > 10000 * 768:
        raise ValueError("complete bounded nativeGoal entry/store pairs required")
    if not raw:
        return ()
    rows = [raw[i : i + 768] for i in range(0, len(raw), 768)]
    base, meta = context.meta["module_slide"], context.meta
    for index, row in enumerate(rows):
        if (
            row[:8] != b"TMGF0001"
            or unpack_from("<4I", row, 8) != (768, 1, context.pid, context.pid)
            or unpack_from("<Q", row, 24)[0] != index + 1
            or unpack_from("<I", row, 32)[0] != index % 2
            or unpack_from("<3Q", row, 88) != (meta["participant"], meta["sim"], meta["controller"])
            or unpack_from("<Q", row, 752)[0] != base
            or row[760:768] != bytes(8)
        ):
            raise ValueError("nativeGoal fact identity/sequence mismatch")
    writer = b"".join(row[240:752] for row in rows)
    writes = decode_writer_pairs(writer, pid=context.pid, meta=meta)
    result = []
    for entry, store, write in zip(rows[::2], rows[1::2], writes, strict=True):
        manager, table_owner, table, map_pointer = unpack_from("<4Q", entry, 40)
        waypoint, endpoint, event, dispatch = unpack_from("<4I", entry, 72)
        selected = unpack_from("<Q", entry, 112)[0]
        embedded = entry[240:752]
        if (
            entry[36:240] != store[36:240]
            or any(not 4096 <= x < 1 << 47 for x in (manager, table_owner, table, map_pointer))
            or selected != table + waypoint * 0x78
            or manager != unpack_from("<Q", embedded, 64 + 8 * 8)[0]
            or unpack_from("<I", entry, 36)[0] != write.native_clock_at_callback
            or (waypoint, bool(endpoint), event, dispatch)
            != (
                write.waypoint_id,
                write.endpoint_write_requested,
                write.raw_waypoint_time,
                write.dispatch_frame_time,
            )
            or endpoint not in (0, 1)
            or not 2200 <= event <= 162200
        ):
            raise ValueError("nativeGoal map/writer binding mismatch")
        try:
            name = entry[120:128].split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("nativeGoal map entry name must be literal ASCII") from error
        if not name or (endpoint and (name != "Goal" or not entry[120 + 0x51] & 1)):
            raise ValueError("nativeGoal endpoint requires the actual Goal map entry")
        result.append(
            NativeWaypointFact(
                write.native_clock_at_callback,
                event,
                waypoint,
                bool(endpoint),
                name,
                entry[120:240],
                dispatch,
            )
        )
    return tuple(result)


def decode_goal_boundary(
    receipt: bytes, facts: tuple[NativeWaypointFact, ...], context: NativeGoalContext
) -> NativeGoalBoundary:
    if type(receipt) is not bytes or len(receipt) != 96 or receipt[:8] != b"TMGR0001":
        raise ValueError("exact nativeGoal reader receipt required")
    meta = context.meta
    sequence, clock, reserved = unpack_from("<QII", receipt, 24)
    pairs, last_goal, goals = unpack_from("<QII", receipt, 40)
    if (
        unpack_from("<4I", receipt, 8) != (96, 1, context.pid, context.pid)
        or not 1 <= sequence <= 16200
        or clock != 2190 + (sequence - 1) * 10
        or clock > 162200
        or reserved
        or receipt[88:96] != bytes(8)
        or unpack_from("<4Q", receipt, 56)
        != (meta["participant"], meta["vehicle"], meta["sim"], meta["module_slide"])
        or pairs != len(facts)
        or pairs > 5000
        or any(fact.callback_native_t > clock for fact in facts)
    ):
        raise ValueError("nativeGoal reader identity/clock/prefix mismatch")
    endpoints = tuple(fact for fact in facts if fact.endpoint_requested)
    expected = endpoints[-1].event_native_t if endpoints else 0
    if goals != len(endpoints) or last_goal != expected:
        raise ValueError("nativeGoal reader endpoint prefix mismatch")
    # A callback can precede the actual physical timestamp. Do not end the
    # episode at callback time or because its file has become nonempty.
    return NativeGoalBoundary(
        clock, pairs, goals, last_goal or None, bool(goals and last_goal <= clock)
    )


class HeldNativeGoalReader:
    """Read two stable owned streams while the matching native actor is held."""

    def __init__(self, receipts: Path, facts: Path, context: NativeGoalContext) -> None:
        self.receipts, self.facts, self.context = receipts, facts, context
        self._files: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.waypoints: tuple[NativeWaypointFact, ...] = ()

    def read(self, native_t: int) -> NativeGoalBoundary:
        receipt_fd, before = _snapshot(self.receipts, 96, 16200)
        fact_fd = None
        try:
            fact_fd, facts_before = _snapshot(self.facts, 768, 10000)
            files = (
                (before.st_dev, before.st_ino),
                (facts_before.st_dev, facts_before.st_ino),
            )
            if self._files is not None and files != self._files:
                raise ValueError("nativeGoal stream identity changed")
            if (
                type(native_t) is not int
                or not 2190 <= native_t <= 162200
                or native_t % 10
                or before.st_size != ((native_t - 2190) // 10 + 1) * 96
            ):
                raise ValueError("nativeGoal receipt stream does not match held boundary")
            raw = linux_os.pread(receipt_fd, 96, before.st_size - 96)
            facts_raw = linux_os.pread(fact_fd, facts_before.st_size, 0)
            waypoints = decode_goal_facts(facts_raw, self.context)
            boundary = decode_goal_boundary(raw, waypoints, self.context)
            for fd, original in ((receipt_fd, before), (fact_fd, facts_before)):
                after = os.fstat(fd)
                if (after.st_size, after.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
                    raise ValueError("nativeGoal stream changed while actor should be held")
        finally:
            os.close(receipt_fd)
            if fact_fd is not None:
                os.close(fact_fd)
        self._files = files
        self.waypoints = waypoints
        return boundary
