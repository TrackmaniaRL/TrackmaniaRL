"""Decode diagnostic writer evidence; this never authorizes an RL terminal."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class WaypointWriteEvidence:
    native_clock_at_callback: int
    raw_waypoint_time: int
    dispatch_frame_time: int
    waypoint_id: int
    endpoint_write_requested: bool
    event_class: int
    selectors: tuple[int, int]
    race_base_words: tuple[int, int]
    physical_crossing_proven: bool = False


def decode_writer_pairs(
    raw: bytes, *, pid: int, meta: Mapping[str, int]
) -> tuple[WaypointWriteEvidence, ...]:
    """Require actual event caller and both native stores; reject API-only writes."""
    if not raw or len(raw) % 1024 or len(raw) > 10000 * 512:
        raise ValueError("complete entry/store record pairs required")
    records = [raw[i : i + 512] for i in range(0, len(raw), 512)]
    base = meta["module_slide"]
    player = meta["participant"]

    def u(data: bytes, offset: int) -> int:
        _decoded_scalar: int = struct.unpack_from("<I", data, offset)[0]
        return _decoded_scalar

    def q(data: bytes, offset: int) -> int:
        _decoded_scalar: int = struct.unpack_from("<Q", data, offset)[0]
        return _decoded_scalar

    for index, record in enumerate(records, 1):
        kind = (index - 1) % 2
        expected_mask = 1231 if kind == 0 else 799
        if (
            record[:8] != b"TMWP0001"
            or struct.unpack_from("<4I", record, 8) != (512, 1, pid, pid)
            or q(record, 24) != index
            or u(record, 32) != kind
            or u(record, 36) != (0x1216411 if kind == 0 else 0x12167D6)
            or q(record, 40) != meta["sim"]
            or q(record, 48) != meta["controller"]
            or u(record, 56) < 2200
            or u(record, 56) > 152200
            or u(record, 56) % 10
            or u(record, 60) != expected_mask
            or (q(record, 248), q(record, 256)) != (player, player)
        ):
            raise ValueError("writer record identity, clock or ownership mismatch")
    result = []
    for entry, store in zip(records[::2], records[1::2], strict=True):
        regs = struct.unpack_from("<23Q", entry, 64)
        raw_time = regs[12]
        frame_time = regs[14]
        selectors = struct.unpack_from("<2I", entry, 264)
        race_base = struct.unpack_from("<2I", entry, 272)
        event = entry[416:464]
        # These two actual return addresses distinguish the witnessed dispatcher
        # from the scripting API and from a guessed function entrypoint.
        if (
            q(entry, 368) != base + 0x11FBA27
            or q(entry, 400) != base + 0x120AC95
            or q(entry, 480) != 0x120AC95
            or q(entry, 360) != regs[10]
            or q(entry, 392) != q(entry, 488)
            or q(entry, 464) < 4096
            or regs[9] != player
            or regs[0] > 65535
            or regs[1] not in (0, 1)
            or raw_time >= 2**32
            or frame_time >= 2**32
            or u(event, 0) != 0
            or u(event, 4) != frame_time
            or u(event, 40) != raw_time
            or u(entry, 56) != u(store, 56)
            or entry[264:280] != store[264:280]
            or u(store, 292) != raw_time
            or u(store, 296) != regs[1]
            or u(store, 288) != regs[0]
        ):
            raise ValueError("event caller, argument or entry/store binding mismatch")
        for offset in (416, 448):
            if (
                u(store, offset + 4) != selectors[(offset - 416) // 32]
                or u(store, offset + 12) != 1
                or u(store, offset + 16) != raw_time
                or u(store, offset + 20) & 65535 != regs[0]
            ):
                raise ValueError("first waypoint native destination mismatch")
        result.append(
            WaypointWriteEvidence(
                native_clock_at_callback=u(entry, 56),
                raw_waypoint_time=raw_time,
                dispatch_frame_time=frame_time,
                waypoint_id=regs[0],
                endpoint_write_requested=bool(regs[1]),
                event_class=u(event, 24),
                selectors=selectors,
                race_base_words=race_base,
            )
        )
    return tuple(result)
