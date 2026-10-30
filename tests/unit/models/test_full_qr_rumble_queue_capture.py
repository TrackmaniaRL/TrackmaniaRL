"""Validate invented converter records without a native observation claim."""

import struct
from itertools import product

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.rumble_queue_capture import (
    RumbleFixtureContext,
    decode_rumble_fixture,
)


def _evidence() -> tuple[bytes, RumbleFixtureContext]:
    context = RumbleFixtureContext(pid=123, module_base=0x10000000, native_time=2200)
    rows = []
    for index, (setting, amplitude, frequency) in enumerate(
        product((0.0, 0.5, 1.0, 2.0), (-0.01, 0.0, 0.1, 0.5, 1.0), (0.0, 3.5, 36.75, 70.0, 100.0))
    ):
        row = bytearray(512)
        row[:8] = b"TMRS0001"
        struct.pack_into("<6I", row, 8, 512, 1, context.pid, index + 1, context.native_time, 1)
        struct.pack_into("<3f", row, 32, amplitude, frequency, setting)
        struct.pack_into("<3I", row, 44, 80, 90000, 1000)
        struct.pack_into("<Q", row, 56, context.module_base + 0x1356080)
        active = amplitude > 0 and setting > 0
        channels = (
            (10.0 if amplitude == 1 and setting == 2 else 2.0, 3.0 if frequency == 36.75 else 0.0)
            if active
            else (0.0, 0.0)
        )
        struct.pack_into("<II2f", row, 64, int(active), 91000 if active else 0, *channels)
        struct.pack_into("<2fI", row, 272, 3.5, 70.0, 1)
        rows.append(row)
    return b"".join(rows), context


def test_synthetic_converter_fixture_preserves_its_own_clock_domain() -> None:
    raw, context = _evidence()
    values = decode_rumble_fixture(raw, context)
    active = [v for v in values if v.queue_expiry is not None]
    assert len(active) == 45
    assert all(v.input_frame_delta_ms == 1000 for v in values)
    assert all(v.queue_expiry == v.input_clock + v.input_frame_delta_ms for v in active)
    assert values[0].input_clock != context.native_time
    assert all(v.physical_collision_proven is False for v in values)


def test_queue_values_are_not_prematurely_clamped_to_device_range() -> None:
    raw, context = _evidence()
    values = decode_rumble_fixture(raw, context)
    assert max(max(v.motor_channels) for v in values) == 10.0
    assert sum(min(v.motor_channels) > 0 for v in values) == 9
    assert all(v.queue_expiry is None for v in values if v.amplitude <= 0 or v.setting <= 0)


@pytest.mark.parametrize(
    ("offset", "format", "value"),
    [
        (16, "I", 1),  # unrelated process
        (20, "I", 2),  # missing first call
        (28, "I", 0),  # trying to relabel synthetic input as gameplay
        (56, "Q", 0),  # unbound native converter
        (272, "f", 0),  # incorrect channel blending constants
        (280, "I", 0),  # own receiver/clock preservation proof missing
        (35 * 512 + 68, "I", 1),  # queue expiry detached from input clock
    ],
)
def test_unbound_or_mislabelled_converter_capture_is_rejected(
    offset: int, format: str, value: int
) -> None:
    raw, context = _evidence()
    changed = bytearray(raw)
    struct.pack_into("<" + format, changed, offset, value)
    with pytest.raises(ValueError, match="mismatch"):
        decode_rumble_fixture(bytes(changed), context)


def test_partial_native_converter_fixture_is_rejected() -> None:
    raw, context = _evidence()
    with pytest.raises(ValueError, match="complete100-case"):
        decode_rumble_fixture(raw[:-1], context)
