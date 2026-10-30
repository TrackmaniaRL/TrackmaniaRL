"""Synthetic per-wheel material cases; native getter parity is not claimed."""

from dataclasses import replace
from struct import pack_into

import pytest

from tests.unit.models.test_full_qr_vehicle_capture import _record
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import decode_vehicle_capture


@pytest.mark.parametrize(
    ("flags", "materials", "expected"),
    [
        ((0, 0, 0, 0), (1, 2, 3, 4), (80, 80, 80, 80)),
        ((2, 2, 2, 2), (1, 2, 3, 4), (13, 13, 13, 13)),
        ((4, 4, 4, 4), (1, 2, 3, 4), (1, 2, 3, 4)),
        ((6, 4, 0, 2), (7, 8, 9, 10), (13, 8, 80, 13)),
        ((4, 0, 2, 4), (80, 1, 0, 0), (80, 80, 13, 0)),
    ],
)
def test_synthetic_material_selection(
    flags: tuple[int, ...], materials: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    raw, context = _record()
    raw = bytearray(raw)
    for wheel, (flag, material) in enumerate(zip(flags, materials, strict=True)):
        pack_into("<I", raw, 208 + 0xD0 + wheel * 0x2C, flag)
        raw[208 + 0xB8 + wheel * 0x2C] = material
    assert decode_vehicle_capture(bytes(raw), context).ground_contact_materials == expected


def test_incomplete_material_source_is_never_published() -> None:
    raw, context = _record()
    capture = decode_vehicle_capture(raw, context)
    with pytest.raises(ValueError, match="complete actual visual"):
        _ = replace(capture, visual_state=capture.visual_state[:-1]).ground_contact_materials


def test_unknown_contact_material_does_not_fall_back_to_no_contact() -> None:
    raw, context = _record()
    raw = bytearray(raw)
    pack_into("<I", raw, 208 + 0xD0, 4)
    raw[208 + 0xB8] = 81
    with pytest.raises(ValueError, match="outside registered domain"):
        _ = decode_vehicle_capture(bytes(raw), context).ground_contact_materials
