"""Transactional incremental reader faults; these do not qualify physics."""

import os
from pathlib import Path
from struct import pack_into

import pytest

from tests.unit.models.test_full_qr_vehicle_capture import _record
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_reader import HeldVehicleStream

pytestmark = pytest.mark.skipif(not hasattr(os, "pread"), reason="Linux native capture reader")


def following(raw: bytes) -> bytes:
    second = bytearray(raw)
    pack_into("<QII", second, 24, 2, 2200, 10)
    pack_into("<I", second, 208 + 12, 2200)
    pack_into("<4I", second, 1248, *(2200,) * 4)
    return bytes(second)


def test_partial_snapshot_does_not_advance_the_stream_cursor(tmp_path: Path) -> None:
    raw, context = _record()
    path = tmp_path / "vehicle.bin"
    path.write_bytes(raw[:-1])
    reader = HeldVehicleStream(path, context)
    with pytest.raises(ValueError, match="complete matching"):
        reader.take(2190)
    path.write_bytes(raw)
    assert [capture.sequence for capture in reader.take(2190)] == [1]


def test_repeated_boundary_rejected_without_losing_next_record(tmp_path: Path) -> None:
    raw, context = _record()
    path = tmp_path / "vehicle.bin"
    path.write_bytes(raw)
    reader = HeldVehicleStream(path, context)
    reader.take(2190)
    with pytest.raises(ValueError, match="fresh complete"):
        reader.take(2190)
    path.write_bytes(raw + following(raw))
    assert [capture.native_t for capture in reader.take(2200)] == [2200]


def test_replaced_stream_cannot_reuse_a_previous_cursor(tmp_path: Path) -> None:
    raw, context = _record()
    path = tmp_path / "vehicle.bin"
    path.write_bytes(raw)
    reader = HeldVehicleStream(path, context)
    reader.take(2190)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(raw + following(raw))
    replacement.replace(path)
    with pytest.raises(ValueError, match="fresh complete"):
        reader.take(2200)


def test_missing_intermediate_record_cannot_publish_a_later_capture(tmp_path: Path) -> None:
    raw, context = _record()
    path = tmp_path / "vehicle.bin"
    path.write_bytes(raw + raw)
    reader = HeldVehicleStream(path, context)
    with pytest.raises(ValueError, match=r"clock|context|sequence"):
        reader.take(2200)
    path.write_bytes(raw + following(raw))
    assert [capture.sequence for capture in reader.take(2200)] == [1, 2]
