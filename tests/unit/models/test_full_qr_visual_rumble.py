"""Literal physical endpoint captures and observed interpolated visual values."""

import hashlib
import json
from pathlib import Path
from struct import pack, unpack_from

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.gameplay_rumble import (
    interpolate_visual_rumble,
)
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import RumbleState

ROOT = Path(__file__).parents[2] / "fixtures/full_qr/interpolation461"
DATA = (ROOT / "cases.json").read_bytes()
assert hashlib.sha256(DATA).hexdigest() == (ROOT / "sha256.txt").read_text().strip()
CASES = json.loads(DATA)["cases"]


@pytest.mark.parametrize("case", CASES)
def test_literal_observed_endpoints_reproduce_visual_bits(case: dict) -> None:
    previous = RumbleState(*unpack_from("<ff", bytes.fromhex(case["previous_source_hex"]), 0x2B0))
    current = RumbleState(*unpack_from("<ff", bytes.fromhex(case["current_source_hex"]), 0x2B0))
    actual = interpolate_visual_rumble(previous, current, case["weight"])
    expected = bytes.fromhex(case["visual_source_hex"])[0x32C:0x334]
    assert pack("<ff", actual.frequency, actual.amplitude) == expected


def test_missing_physical_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="two actual typed"):
        interpolate_visual_rumble(None, RumbleState(1.0, 0.5), 0.5)


def test_unbounded_weight_cannot_extrapolate_a_motor_signal() -> None:
    with pytest.raises(ValueError, match="bounded physical"):
        interpolate_visual_rumble(RumbleState(1.0, 0.5), RumbleState(1.0, 0.4), 1.1)
