"""End-to-end actual Windows queue bytes to observed SDK motor bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.gameplay_rumble import decode_device_queue
from trackmaniarl.trackmania.native_headless.full_qr.motor_feedback import queued_motor_words
from trackmaniarl.trackmania.native_headless.full_qr.rumble_queue import evaluate_rumble_queue

ROOT = Path(__file__).parents[2] / "fixtures/full_qr/device_queue461"
DATA = (ROOT / "cases.json").read_bytes()
assert hashlib.sha256(DATA).hexdigest() == (ROOT / "sha256.txt").read_text().strip()
CASES = json.loads(DATA)["cases"]


@pytest.mark.parametrize("case", CASES)
def test_actual_queue_matches_both_observed_sdk_motors(case: dict) -> None:
    pulses = decode_device_queue(bytes.fromhex(case["queue_hex"]))
    evaluation = evaluate_rumble_queue(pulses, case["input_clock"])
    words = queued_motor_words(*evaluation.channels)
    assert tuple(word >> 8 for word in words) == (case["expected_large"], case["expected_small"])


def test_partial_queue_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete actual"):
        decode_device_queue(bytes.fromhex(CASES[0]["queue_hex"])[:-1])


def test_corrupt_count_cannot_be_treated_as_empty_queue() -> None:
    raw = bytearray.fromhex(CASES[0]["queue_hex"])
    raw[:4] = (17).to_bytes(4, "little")
    with pytest.raises(ValueError, match="exceeds source capacity"):
        decode_device_queue(bytes(raw))
