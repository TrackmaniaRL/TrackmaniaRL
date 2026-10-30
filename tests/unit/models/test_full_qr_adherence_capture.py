"""Synthetic adherence stream, boundary joins and explicit corruptions."""

import os
from dataclasses import replace
from pathlib import Path
from struct import pack_into

import pytest

from tests.unit.models.test_full_qr_vehicle_capture import _record
from trackmaniarl.trackmania.native_headless.full_qr.adherence_capture import (
    HeldAdherenceReader,
    decode_adherence,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    GRAPH_FIELDS,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import (
    VehicleCaptureContext,
    decode_vehicle_capture,
)


def fixture() -> tuple[dict, bytes, bytes, VehicleCaptureContext]:
    vehicle, original_context = _record()
    context = replace(original_context, native_t=2200)
    vehicle_bytes = bytearray(vehicle)
    pack_into("<QII", vehicle_bytes, 24, 2, context.native_t, 10)
    pack_into("<I", vehicle_bytes, 220, context.native_t)
    pack_into("<4I", vehicle_bytes, 1248, *([context.native_t] * 4))
    raw = bytearray(128)
    raw[:8] = b"TMAD0001"
    pack_into("<4I", raw, 8, 128, 1, context.pid, context.pid)
    pack_into("<QII", raw, 24, 2, context.native_t, 10)
    pack_into("<4Q", raw, 40, context.participant, context.vehicle, 0x30000, 0x40000)
    pack_into("<4I", raw, 72, 0x02000001, 0, 0x0A020000, 1)
    pack_into("<Qf", raw, 88, context.vehicle, 1.0)
    pack_into("<3I", raw, 100, 0x121C7A0, 0x1195D70, 0xA48730)
    pack_into("<QII", raw, 112, 0x30000, context.native_t, 0)
    return {}, bytes(raw), bytes(vehicle_bytes), context


def test_synthetic_boundary_preserves_coefficient_and_all_other_channels() -> None:
    _, raw, vehicle_raw, context = fixture()
    adherence = decode_adherence(raw, context)
    vehicle = decode_vehicle_capture(vehicle_raw, context)
    frame = NativeGraphFrame.from_measured_capture(vehicle, adherence)
    legacy = NativeGraphFrame.from_capture(vehicle)
    assert frame == legacy  # The synthetic coefficient matches the default.
    assert frame.channels[GRAPH_FIELDS.index(29)] == adherence.coefficient
    assert adherence.native_t == 2200
    assert adherence.sequence == 2


def test_measured_coefficient_does_not_use_the_null_getter_default() -> None:
    _, raw, vehicle_raw, context = fixture()
    adherence = replace(decode_adherence(raw, context), coefficient=0.625)
    # Explicit unit mutation, not telemetry evidence.
    vehicle = decode_vehicle_capture(vehicle_raw, context)
    frame = NativeGraphFrame.from_measured_capture(vehicle, adherence)
    assert frame.channels[GRAPH_FIELDS.index(29)] == 0.625
    for index, value in enumerate(NativeGraphFrame.from_capture(vehicle).channels):
        if GRAPH_FIELDS[index] != 29:
            assert frame.channels[index] == value


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (16, "<I", 99999),  # foreign actor
        (24, "<Q", 3),  # missing interval
        (32, "<I", 152190),  # stale boundary
        (48, "<Q", 4096),  # foreign car
        (72, "<I", 0x0FF00000),  # missing entity
        (80, "<I", 0x032E2000),  # script wrapper is not the physics reference
        (84, "<I", 0xFFFFFFFF),  # invalid generation
        (88, "<Q", 4096),  # wrong resolved physics
        (96, "<f", float("nan")),
        (104, "<I", 0),  # changed source
        (112, "<Q", 4096),  # foreign temporary context
        (120, "<I", 152190),  # source clock changed
    ],
)
def test_corrupted_measurement_is_refused(offset: int, fmt: str, value: int | float) -> None:
    _, raw, _, context = fixture()
    changed = bytearray(raw)
    pack_into(fmt, changed, offset, value)
    with pytest.raises(ValueError, match="adherence"):
        decode_adherence(bytes(changed), context)


def test_join_refuses_stale_measurement_and_foreign_vehicle_context() -> None:
    _, raw, vehicle_raw, context = fixture()
    adherence = decode_adherence(raw, context)
    vehicle = decode_vehicle_capture(vehicle_raw, context)
    with pytest.raises(ValueError, match="boundary mismatch"):
        NativeGraphFrame.from_measured_capture(vehicle, replace(adherence, native_t=152150))
    with pytest.raises(ValueError, match="another adherence arena"):
        NativeGraphFrame.from_measured_capture(replace(vehicle, global_owner=4096), adherence)


@pytest.mark.skipif(not hasattr(os, "pread"), reason="native held reader requires Linux pread")
def test_held_reader_rejects_missing_prefix_and_partial_stream(tmp_path: Path) -> None:
    _, raw, _, context = fixture()
    path = tmp_path / "adherence.bin"
    path.write_bytes(raw)
    reader = HeldAdherenceReader(path, context)
    with pytest.raises(ValueError, match="sequence/size"):
        reader.read(context.native_t)
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        reader.read(context.native_t)
