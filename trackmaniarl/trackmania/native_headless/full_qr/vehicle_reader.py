"""Read the last complete observer record while its owned actor is held at IPC."""

from __future__ import annotations

import os
import stat
from dataclasses import replace
from pathlib import Path

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import (
    RECORD_SIZE,
    VehicleCapture,
    VehicleCaptureContext,
    decode_vehicle_capture,
)


class HeldVehicleReader:
    """Capture reader, not a process owner or an independent live authorization.

    The caller must hold its own actor after the matching reader acknowledgement.
    Context comes from that session's verified metadata and current boundary.
    """

    def __init__(self, path: Path, context: VehicleCaptureContext) -> None:
        self.path = path
        self.context = context

    def read(self, native_t: int) -> VehicleCapture:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size < RECORD_SIZE
                or before.st_size > 16200 * RECORD_SIZE
                or before.st_size % RECORD_SIZE
            ):
                raise ValueError("incomplete or unbounded vehicle capture stream")
            raw = linux_os.pread(descriptor, RECORD_SIZE, before.st_size - RECORD_SIZE)
            after = os.fstat(descriptor)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("vehicle capture stream changed while actor should be held")
        finally:
            os.close(descriptor)
        captured = decode_vehicle_capture(raw, replace(self.context, native_t=native_t))
        if captured.sequence * RECORD_SIZE != before.st_size:
            raise ValueError("vehicle capture stream sequence/size mismatch")
        return captured


class HeldVehicleStream:
    """Read every fresh ten-ms vehicle source at an exact held boundary."""

    def __init__(self, path: Path, context: VehicleCaptureContext) -> None:
        self.path, self.context = path, context
        self._sequence = 0
        self._file: tuple[int, int] | None = None

    def take(self, native_t: int) -> tuple[VehicleCapture, ...]:
        if type(native_t) is not int or not 2190 <= native_t <= 162200 or native_t % 10:
            raise ValueError("actual ten-ms vehicle boundary required")
        target = (native_t - 2190) // 10 + 1
        descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            identity = (before.st_dev, before.st_ino)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != target * RECORD_SIZE
                or not self._sequence < target <= 16200
                or (self._file is not None and self._file != identity)
            ):
                raise ValueError("fresh complete matching vehicle stream required")
            captures = []
            for index in range(self._sequence, target):
                clock = 2190 + index * 10
                raw = linux_os.pread(descriptor, RECORD_SIZE, index * RECORD_SIZE)
                capture = decode_vehicle_capture(raw, replace(self.context, native_t=clock))
                if capture.sequence != index + 1:
                    raise ValueError("vehicle stream sequence gap")
                captures.append(capture)
            after = os.fstat(descriptor)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("vehicle stream changed while actor should be held")
        finally:
            os.close(descriptor)
        self._sequence, self._file = target, identity
        return tuple(captures)
