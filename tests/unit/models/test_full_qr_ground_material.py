"""Actual150s material captures checked by60008 original native getter calls."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from struct import unpack_from

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import (
    VehicleCapture,
    VehicleCaptureContext,
    decode_vehicle_capture,
)

ROOT = Path(__file__).parents[2] / "fixtures/full_qr/ground_material430"
PROOF = json.loads((ROOT / "context.json").read_text())


def captured(index: int) -> VehicleCapture:
    row = PROOF["selected_literal_captures"][index]
    data = (ROOT / row["name"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["sha256"]
    context = PROOF["context"]
    return decode_vehicle_capture(
        data,
        VehicleCaptureContext(
            PROOF["pid"],
            context["participant"],
            context["vehicle"],
            context["module_base"],
            row["native_t"],
        ),
    )


@pytest.mark.parametrize("index", range(len(PROOF["selected_literal_captures"])))
def test_literal_material_combinations_match_actual_original_getters(index: int) -> None:
    actual = captured(index)
    expected = PROOF["selected_literal_captures"][index]["original_getter_results"]
    assert actual.ground_contact_materials == tuple(expected)


def test_incomplete_material_source_is_never_published() -> None:
    actual = captured(0)
    with pytest.raises(ValueError, match="complete actual visual"):
        _ = replace(actual, visual_state=actual.visual_state[:-1]).ground_contact_materials


def test_unknown_contact_material_does_not_fall_back_to_no_contact() -> None:
    for index in range(len(PROOF["selected_literal_captures"])):
        actual = captured(index)
        for wheel in range(4):
            flags = unpack_from("<I", actual.visual_state, 0xD0 + wheel * 0x2C)[0]
            if flags & 4 and not flags & 2:
                changed = bytearray(actual.visual_state)
                changed[0xB8 + wheel * 0x2C] = 81
                with pytest.raises(ValueError, match="outside registered domain"):
                    _ = replace(actual, visual_state=bytes(changed)).ground_contact_materials
                return
    pytest.fail("actual corpus must contain a ground-contact material")
