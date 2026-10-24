"""Strict offline contact provenance, without assigning physical finish timing."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass

from trackmaniarl.trackmania.native_headless.full_qr.waypoint_capture import decode_writer_pairs


@dataclass(frozen=True)
class ContactCaptureContext:
    pid: int
    meta: Mapping[str, int]


@dataclass(frozen=True)
class ContactWaypointChain:
    contact_count: int
    contacts_before_conversion: int
    callback_native_time: int
    visual_state_stamp: int
    selected_raw_time: int
    dispatch_frame_time: int
    physical_crossing_proven: bool = False


def decode_contact_chain(
    raw: bytes, writer_raw: bytes, context: ContactCaptureContext
) -> ContactWaypointChain:
    """Accept the first witnessed contact88 -> event32 -> event48 -> writer chain.

    Exact raw time propagation and the native minimum selection are verified.
    The meaning of that timestamp and its relationship to client UI are separate.
    """
    if not raw or len(raw) % 1024 or len(raw) > 10000 * 1024:
        raise ValueError("complete bounded contact records required")
    pid, meta = context.pid, context.meta

    def u(data: bytes, offset: int) -> int:
        _decoded_scalar: int = struct.unpack_from("<I", data, offset)[0]
        return _decoded_scalar

    def q(data: bytes, offset: int) -> int:
        _decoded_scalar: int = struct.unpack_from("<Q", data, offset)[0]
        return _decoded_scalar

    contacts = []
    conversions = []
    for sequence, offset in enumerate(range(0, len(raw), 1024), 1):
        record = raw[offset : offset + 1024]
        kind = u(record, 32)
        if (
            record[:8] != b"TMEV0001"
            or struct.unpack_from("<4I", record, 8) != (1024, 1, pid, pid)
            or q(record, 24) != sequence
            or kind not in (0, 1)
            or u(record, 36) != (0xCBC6BC, 0x120B15F)[kind]
            or u(record, 40) < 2190
            or u(record, 40) > 162200
            or u(record, 40) % 10
            or struct.unpack_from("<4Q", record, 48)
            != (meta["sim"], meta["controller"], meta["participant"], meta["vehicle"])
            or u(record, 268) != (127, 47)[kind]
        ):
            raise ValueError("contact identity or ownership mismatch")
        if kind == 0:
            if (
                u(record, 448) != 0xA020000
                or q(record, 456) != meta["vehicle"]
                or u(record, 464) != 0x327D000
                or u(record, 468) != u(record, 544)
                or q(record, 472) != q(record, 480)
                or u(record, 44) != u(record, 412)
            ):
                raise ValueError("contact body metadata or timestamp mismatch")
            contacts.append(record)
        else:
            conversions.append((record, tuple(contacts)))
    if len(conversions) != 1:
        raise ValueError("exactly one first-waypoint conversion required")
    converted, preceding = conversions[0]
    source = converted[336:368]
    dest = converted[368:416]
    matching = [
        contact
        for contact in preceding
        if q(contact, 480) == q(source, 0)
        and contact[448:464] == source[8:24]
        and u(contact, 40) == u(converted, 40)
    ]
    if not matching:
        raise ValueError("no preceding own contact for conversion")
    selected = u(source, 24)
    contact = min(matching, key=lambda value: u(value, 44))
    if (
        selected != u(contact, 44)
        or q(source, 16) != meta["vehicle"]
        or u(dest, 0) != 0
        or dest[8:24] != source[8:24]
        or dest[24:32] != contact[464:472]
        or q(dest, 32) != q(source, 0)
        or u(dest, 40) != selected
        or converted[496:544] != contact[496:544]
    ):
        raise ValueError("contact minimum or event conversion mismatch")
    (writer,) = decode_writer_pairs(writer_raw, pid=pid, meta=meta)
    if (
        writer.raw_waypoint_time != selected
        or writer.native_clock_at_callback != u(converted, 40)
        or writer.dispatch_frame_time != u(dest, 4)
    ):
        raise ValueError("event conversion and waypoint writer mismatch")
    return ContactWaypointChain(
        contact_count=len(contacts),
        contacts_before_conversion=len(matching),
        callback_native_time=u(converted, 40),
        visual_state_stamp=u(converted, 312),
        selected_raw_time=selected,
        dispatch_frame_time=writer.dispatch_frame_time,
    )
