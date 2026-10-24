"""Strict measured coefficient stream, separate from legacy getter defaults."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from struct import unpack_from

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCaptureContext

RECORD_SIZE = 128


@dataclass(frozen=True, slots=True)
class MeasuredAdherence:
    native_t: int
    sequence: int
    participant: int
    vehicle: int
    arena: int
    pool: int
    entity: int
    coefficient: float
    reference: bytes


def decode_adherence(raw: bytes, context: VehicleCaptureContext) -> MeasuredAdherence:
    if type(raw) is not bytes or len(raw) != RECORD_SIZE or raw[:8] != b"TMAD0001":
        raise ValueError("exact TMAD0001 measured adherence record required")
    if unpack_from("<4I", raw, 8) != (RECORD_SIZE, 1, context.pid, context.pid):
        raise ValueError("adherence header/owner mismatch")
    sequence, clock, delta = unpack_from("<QII", raw, 24)
    if (
        not 1 <= sequence <= 16200
        or clock != context.native_t
        or clock != 2190 + (sequence - 1) * 10
        or delta != (0 if sequence == 1 else 10)
        or unpack_from("<II", raw, 120) != (clock, 0)
    ):
        raise ValueError("adherence sequence/clock mismatch")
    participant, vehicle, arena, pool = unpack_from("<4Q", raw, 40)
    if (participant, vehicle) != (context.participant, context.vehicle) or not all(
        4096 <= address < 1 << 47 for address in (participant, vehicle, arena, pool)
    ):
        raise ValueError("adherence current actor/arena binding mismatch")
    entity, slot = unpack_from("<II", raw, 72)
    kind, generation, physical = unpack_from("<IIQ", raw, 80)
    if (
        entity & 0xFF000000 != 0x02000000
        or slot >= 65536
        or kind != 0x0A020000
        or generation == 0xFFFFFFFF
        or physical != vehicle
    ):
        raise ValueError("adherence actual entity/physics reference mismatch")
    coefficient = unpack_from("<f", raw, 96)[0]
    if not isfinite(coefficient):
        raise ValueError("nonfinite measured adherence")
    if unpack_from("<3I", raw, 100) != (0x121C7A0, 0x1195D70, 0xA48730):
        raise ValueError("adherence pinned source route mismatch")
    if unpack_from("<Q", raw, 112)[0] not in (0, arena):
        raise ValueError("adherence temporary context belongs to another arena")
    return MeasuredAdherence(
        clock, sequence, participant, vehicle, arena, pool, entity, coefficient, raw[80:96]
    )


class HeldAdherenceReader:
    """Read only while the owning session holds its actor at the matching IPC boundary."""

    def __init__(self, path: Path, context: VehicleCaptureContext) -> None:
        self.path = path
        self.context = context

    def read(self, native_t: int) -> MeasuredAdherence:
        descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or not RECORD_SIZE <= before.st_size <= 16200 * RECORD_SIZE
                or before.st_size % RECORD_SIZE
            ):
                raise ValueError("incomplete or unbounded adherence stream")
            raw = linux_os.pread(descriptor, RECORD_SIZE, before.st_size - RECORD_SIZE)
            after = os.fstat(descriptor)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("adherence stream changed while actor should be held")
        finally:
            os.close(descriptor)
        captured = decode_adherence(raw, replace(self.context, native_t=native_t))
        if captured.sequence * RECORD_SIZE != before.st_size:
            raise ValueError("adherence stream sequence/size mismatch")
        return captured
