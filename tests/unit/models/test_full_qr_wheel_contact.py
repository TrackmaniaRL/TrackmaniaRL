"""Actual contact records must retain provenance and never become fake rewards."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.wheel_contact_capture import (
    RECORD_BYTES,
    WheelContactContext,
    decode_wheel_contact,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/wheel_contacts_v1"


def evidence() -> tuple[bytes, WheelContactContext, list[dict]]:
    raw = (FIXTURE / "contacts.bin").read_bytes()
    saved = json.loads((FIXTURE / "context.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == saved["sha256"]
    return raw, WheelContactContext(**saved["context"]), saved["expected"]


def test_real_native_ground_and_side_contacts_preserve_literal_fields() -> None:
    raw, context, expected = evidence()
    results = [
        decode_wheel_contact(raw[i : i + RECORD_BYTES], context)
        for i in range(0, len(raw), RECORD_BYTES)
    ]
    assert json.loads(json.dumps([asdict(result) for result in results])) == expected
    assert {result.material for result in results} == {9, 16}
    assert any(abs(result.normal[1]) < 0.1 for result in results)
    assert any(result.normal[1] > 0.99 for result in results)
    assert all(not hasattr(result, "collision_detected") for result in results)


@pytest.mark.parametrize(
    ("offset", "format", "value"),
    [
        (16, "I", 1),
        (20, "I", 1),
        (32, "Q", 0),
        (48, "Q", 2),
        (56, "I", 0),
        (72, "I", 6071),
        (76, "I", 1),
        (80 + 13 * 8, "Q", 0),
        (80 + 14 * 8, "Q", 0),
        (304, "Q", 0),
        (848, "Q", 1),
        (928, "I", 2),
        (1288, "Q", 0),
    ],
)
def test_foreign_or_unavailable_contact_cannot_enter_source_adapter(
    offset: int, format: str, value: int
) -> None:
    raw, context, _ = evidence()
    damaged = bytearray(raw[:RECORD_BYTES])
    pack_into("<" + format, damaged, offset, value)
    with pytest.raises(ValueError, match="wheel contact"):
        decode_wheel_contact(bytes(damaged), context)


@pytest.mark.parametrize("offset", [0x1C, 0x28, 0x40])
def test_nonfinite_contact_is_rejected_without_substitution(offset: int) -> None:
    raw, context, _ = evidence()
    damaged = bytearray(raw[:RECORD_BYTES])
    pack_into("<f", damaged, 936 + offset, float("nan"))
    with pytest.raises(ValueError, match="finite literal"):
        decode_wheel_contact(bytes(damaged), context)


def test_other_record_version_and_out_of_range_wheel_are_rejected() -> None:
    raw, context, _ = evidence()
    with pytest.raises(ValueError, match="complete extended"):
        decode_wheel_contact(raw[:1024], context)
    damaged = bytearray(raw[:RECORD_BYTES])
    pack_into("<Q", damaged, 80 + 12 * 8, context.wheel_count)
    with pytest.raises(ValueError, match="pointer/index"):
        decode_wheel_contact(bytes(damaged), context)
