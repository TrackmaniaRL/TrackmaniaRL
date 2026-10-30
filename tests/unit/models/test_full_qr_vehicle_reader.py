"""Linux capture stream tests; no engine process is launched."""

import os
from pathlib import Path

import pytest

from tests.unit.models.test_full_qr_vehicle_capture import _record
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_reader import HeldVehicleReader

pytestmark = pytest.mark.skipif(not hasattr(os, "pread"), reason="Linux native capture reader")


def test_held_reader_rejects_stale_clock_and_partial_tail(tmp_path: Path) -> None:
    raw, context = _record()
    path = tmp_path / "vehicle.bin"
    path.write_bytes(raw)
    reader = HeldVehicleReader(path, context)
    assert reader.read(2190).skidding_count == 3
    with pytest.raises(ValueError, match="context"):
        reader.read(2200)
    path.write_bytes(raw + b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        reader.read(2190)


def test_held_reader_rejects_replayed_sequence_and_symlink(tmp_path: Path) -> None:
    raw, context = _record()
    path = tmp_path / "vehicle.bin"
    path.write_bytes(raw + raw)
    with pytest.raises(ValueError, match="sequence/size"):
        HeldVehicleReader(path, context).read(2190)
    link = tmp_path / "redirect.bin"
    link.symlink_to(path)
    with pytest.raises(OSError, match="symbolic links"):
        HeldVehicleReader(link, context).read(2190)
