"""Actual Windows visual inputs and original frequency helper results."""

import hashlib
import json
from pathlib import Path
from struct import pack

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.gameplay_rumble import unity_gameplay_pulse

ROOT = Path(__file__).parents[2] / "fixtures/full_qr/gameplay456"
DATA = (ROOT / "cases.json").read_bytes()
assert hashlib.sha256(DATA).hexdigest() == (ROOT / "sha256.txt").read_text().strip()
CASES = json.loads(DATA)["cases"]


@pytest.mark.parametrize("case", CASES)
def test_actual_visual_channels_match_original_windows_leaf(case: dict) -> None:
    pulse = unity_gameplay_pulse(
        case["amp"], case["freq"], 1.0, case["input_clock"], case["minimum_duration"]
    )
    assert pulse is not None
    assert pack("<ff", pulse.first, pulse.second).hex() == case["original_hex"]


def test_unmeasured_intensity_cannot_silently_use_unity_conversion() -> None:
    with pytest.raises(ValueError, match="only measured unity"):
        unity_gameplay_pulse(0.8, 8.0, 0.5, 100, 7)


def test_nonfinite_visual_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite actual"):
        unity_gameplay_pulse(float("nan"), 8.0, 1.0, 100, 7)
