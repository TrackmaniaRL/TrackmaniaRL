"""Decode the explicit synthetic native-converter fixture, never collisions."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class RumbleFixtureContext:
    pid: int
    module_base: int
    native_time: int


@dataclass(frozen=True)
class NativeRumbleFixture:
    amplitude: float
    frequency: float
    setting: float
    input_clock: int
    input_frame_delta_ms: int
    queue_expiry: int | None
    motor_channels: tuple[float, float]
    physical_collision_proven: bool = False


def decode_rumble_fixture(
    raw: bytes, context: RumbleFixtureContext
) -> tuple[NativeRumbleFixture, ...]:
    """Validate all100 pinned fixture cases and preserve their input-clock domain.

    The queue stores floats before device conversion and can contain values>1.
    No motor word or historical collision boolean can be inferred here.
    """
    if len(raw) != 100 * 512:
        raise ValueError("complete100-case synthetic converter fixture required")
    result = []
    settings = (0.0, 0.5, 1.0, 2.0)
    amplitudes = (-0.01, 0.0, 0.1, 0.5, 1.0)
    frequencies = (0.0, 3.5, 36.75, 70.0, 100.0)
    for index in range(100):
        row = raw[index * 512 : (index + 1) * 512]
        size, version, pid, sequence, native_time, synthetic = struct.unpack_from("<6I", row, 8)
        amplitude, frequency, setting = struct.unpack_from("<3f", row, 32)
        duration, input_clock, frame_delta = struct.unpack_from("<3I", row, 44)
        queue_count, expiry, first, second = struct.unpack_from("<II2f", row, 64)
        setting_index, within_setting = divmod(index, 25)
        amplitude_index, frequency_index = divmod(within_setting, 5)
        expected = struct.pack(
            "<3f",
            amplitudes[amplitude_index],
            frequencies[frequency_index],
            settings[setting_index],
        )
        if (
            row[:8] != b"TMRS0001"
            or (size, version, pid, sequence, native_time, synthetic)
            != (512, 1, context.pid, index + 1, context.native_time, 1)
            or struct.unpack_from("<Q", row, 56)[0] != context.module_base + 0x1356080
            or row[32:44] != expected
            or duration != 80
            or frame_delta != 1000
            or struct.unpack_from("<2f", row, 272) != (3.5, 70.0)
            or struct.unpack_from("<I", row, 280)[0] != 1
            or row[80:272] != bytes(192)
            or row[284:] != bytes(228)
        ):
            raise ValueError("synthetic fixture identity, domain or native-call proof mismatch")
        active = amplitude > 0 and setting > 0
        if (
            queue_count != int(active)
            or (active and expiry != (input_clock + frame_delta) % 2**32)
            or (not active and (expiry != 0 or first != 0 or second != 0))
            or not all(math.isfinite(v) and v >= 0 for v in (first, second))
        ):
            raise ValueError("native fixture queue or input-clock binding mismatch")
        result.append(
            NativeRumbleFixture(
                amplitude=amplitude,
                frequency=frequency,
                setting=setting,
                input_clock=input_clock,
                input_frame_delta_ms=frame_delta,
                queue_expiry=expiry if active else None,
                motor_channels=(first, second),
            )
        )
    return tuple(result)
