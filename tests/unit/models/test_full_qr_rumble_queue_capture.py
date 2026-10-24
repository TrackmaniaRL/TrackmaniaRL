"""Replay actual converter calls with explicitly synthetic amplitudes."""

import hashlib
import json
import struct
from pathlib import Path

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.rumble_queue_capture import (
    RumbleFixtureContext,
    decode_rumble_fixture,
)


def _evidence() -> tuple[bytes, RumbleFixtureContext]:
    directory = Path(__file__).parents[2] / "fixtures/full_qr/rumble_converter_v1"
    saved = json.loads((directory / "context.json").read_text())
    raw = (directory / "converter.bin").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == saved["sha256"]
    context = RumbleFixtureContext(
        **{key: saved[key] for key in ("pid", "module_base", "native_time")}
    )
    return raw, context


def test_native_converter_fixture_preserves_its_own_clock_domain() -> None:
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
