"""Bind native waypoint classification to its actual map entry and writer.

This evidence describes the native endpoint decision. It does not supply the
historical client's UISequence or authorize a training terminal.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass

from trackmaniarl.trackmania.native_headless.full_qr.waypoint_capture import decode_writer_pairs


@dataclass(frozen=True)
class EndpointClassificationContext:
    pid: int
    birth: int
    meta: Mapping[str, int]


@dataclass(frozen=True)
class NativeEndpointClassification:
    native_time: int
    waypoint_id: int
    map_entry_name: str
    classification: int
    raw_waypoint_time: int
    client_ui_finish_proven: bool = False


def decode_endpoint_classification(
    raw: bytes,
    writer_raw: bytes,
    context: EndpointClassificationContext,
) -> NativeEndpointClassification:
    """Decode the witnessed single entry/return pair for the pinned map.

    The original classifier must return3 for an entry whose endpoint bit is set;
    the actual caller must subsequently request an endpoint write for the same
    player, manager and waypoint. Empty evidence is never a negative finish.
    """
    pid, birth, meta = context.pid, context.birth, context.meta
    if len(raw) != 2048 or len(writer_raw) != 1024 or birth <= 0:
        raise ValueError("one complete classification and writer pair required")
    before, after = raw[:1024], raw[1024:]
    base = meta["module_slide"]
    writes = decode_writer_pairs(writer_raw, pid=pid, meta=meta)

    def q(data: bytes, offset: int) -> int:
        _decoded_scalar: int = struct.unpack_from("<Q", data, offset)[0]
        return _decoded_scalar

    def u(data: bytes, offset: int) -> int:
        _decoded_scalar: int = struct.unpack_from("<I", data, offset)[0]
        return _decoded_scalar

    for index, record in enumerate((before, after)):
        if (
            record[:8] != b"TMFC0001"
            or struct.unpack_from("<3I", record, 8) != (1024, 1, pid)
            or u(record, 20) <= 1
            or q(record, 24) != index + 1
            or q(record, 32) != base + 0x12154A0
            or q(record, 40) != base + 0x12157BB
            or q(record, 48) & 15 != 1 << index
            or struct.unpack_from("<2I", record, 56) != (4, 5)
            or u(record, 76) != index
            or q(record, 80 + 16 * 8) != base + (0x12154A0 if index == 0 else 0x12157BB)
            or q(record, 848) != birth
        ):
            raise ValueError("classifier identity, breakpoint or sequence mismatch")
    native_time = u(before, 72)
    if (
        native_time != u(after, 72)
        or native_time != writes[0].native_clock_at_callback
        or before[20:24] != after[20:24]
        or not 2200 <= native_time <= 152200
        or native_time % 10
        or q(before, 304) != base + 0x11FB9F8
        or q(after, 304 + 0x98) != base + 0x11FB9F8
        or q(before, 80 + 19 * 8) != q(after, 80 + 19 * 8) + 0x98
    ):
        raise ValueError("classifier caller, stack or native clock mismatch")

    manager = q(before, 80 + 14 * 8)
    waypoint_id = q(before, 80 + 12 * 8)
    entry_address = q(before, 608)
    table = q(before, 592)
    if (
        manager < 4096
        or manager != q(before, 576)
        or manager != q(writer_raw, 64 + 8 * 8)
        or q(before, 80 + 13 * 8) != meta["participant"]
        or waypoint_id != writes[0].waypoint_id
        or waypoint_id > 65535
        or any(q(before, offset) < 4096 for offset in (584, 592, 600, 776))
        or entry_address != table + waypoint_id * 0x78
        or not before[648 + 0x51] & 1
        or not writes[0].endpoint_write_requested
        or q(after, 80 + 10 * 8) != 3
    ):
        raise ValueError("native endpoint, map entry or writer binding mismatch")
    try:
        name = before[648:656].split(b"\0", 1)[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("pinned map entry name is not ASCII") from exc
    if not name:
        raise ValueError("pinned map entry name is missing")
    return NativeEndpointClassification(
        native_time=native_time,
        waypoint_id=waypoint_id,
        map_entry_name=name,
        classification=3,
        raw_waypoint_time=writes[0].raw_waypoint_time,
    )
