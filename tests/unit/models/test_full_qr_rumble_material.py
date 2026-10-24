"""Regression cases from original Windows helper on actual native wheel sources."""

import hashlib
import json
from pathlib import Path

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.rumble_material import (
    RumbleMaterial,
    select_rumble_material,
)

ROOT = Path(__file__).parents[2] / "fixtures/full_qr/rumble_material439"
DATA = (ROOT / "cases.json").read_bytes()
assert hashlib.sha256(DATA).hexdigest() == (ROOT / "sha256.txt").read_text().strip()
CASES = json.loads(DATA)["cases"]


@pytest.mark.parametrize("case", CASES)
def test_actual_bound_sources_match_original_windows_helper(case: dict) -> None:
    expected = RumbleMaterial(**case["expected"])
    actual = select_rumble_material(
        bytes.fromhex(case["visual"]),
        tuple(bytes.fromhex(model) for model in case["wheel_models"]),
    )
    assert actual == expected
    for original in case["original_outputs"]:
        assert (actual.material, actual.front_contacts, actual.back_contacts) == (
            original["material"],
            original["front"],
            original["back"],
        )


def test_missing_wheel_binding_is_rejected() -> None:
    case = CASES[0]
    with pytest.raises(ValueError, match="four independently bound"):
        select_rumble_material(bytes.fromhex(case["visual"]), ())


def test_nonfinite_wheel_geometry_is_rejected() -> None:
    case = CASES[0]
    models = [bytes.fromhex(model) for model in case["wheel_models"]]
    changed = bytearray(models[0])
    changed[0x18:0x1C] = bytes.fromhex("0000c07f")
    models[0] = bytes(changed)
    with pytest.raises(ValueError, match="finite actual wheel"):
        select_rumble_material(bytes.fromhex(case["visual"]), tuple(models))
