"""Actual post-solver arguments, including a non-wheel impact and future times."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.prepared_contact_capture import (
    PreparedContactContext,
    decode_prepared_contact,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/prepared_contacts_v1"


def evidence() -> tuple[bytes, PreparedContactContext, list[dict]]:
    raw = (FIXTURE / "contacts.bin").read_bytes()
    saved = json.loads((FIXTURE / "context.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == saved["sha256"]
    context = saved["context"]
    for key in ("normalization_header", "normalization_thresholds"):
        context[key] = bytes.fromhex(context[key])
    context["wheel_models"] = tuple(context["wheel_models"])
    return raw, PreparedContactContext(**context), saved["expected"]


def test_actual_prepared_vectors_keep_interpolated_time_and_nonwheel_sentinel() -> None:
    raw, context, expected = evidence()
    decoded = [
        decode_prepared_contact(raw[i : i + 2048], context) for i in range(0, len(raw), 2048)
    ]
    assert json.loads(json.dumps([asdict(x) for x in decoded])) == expected
    assert {x.wheel_index for x in decoded} == {-1, 0, 1, 2, 3}
    assert all(x.event_time > x.native_time for x in decoded)
    body = next(x for x in decoded if x.wheel_index == -1)
    assert body.wheel_category is None
    assert (
        max(
            abs(sum(a * b for a, b in zip(x.normal, x.contact_velocity, strict=True)))
            for x in decoded
        )
        > 77
    )
    assert all(
        not hasattr(x, "contact_flags") and not hasattr(x, "collision_detected") for x in decoded
    )


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (16, "I", 1),
        (20, "I", 1),
        (32, "Q", 0),
        (48, "Q", 2),
        (72, "I", 26801),
        (76, "I", 1),
        (304, "Q", 0),
        (848, "Q", 1),
        (928, "I", 128),
        (1288, "Q", 0),
        (1600, "Q", 0),
        (1608, "Q", 0),
        (1640, "f", 0.0),
        (1684, "f", float("nan")),
        (1732, "i", -2),
        (1740, "I", 0),
    ],
)
def test_foreign_corrupt_or_incomplete_prepared_source_is_rejected(
    offset: int, fmt: str, value: int | float
) -> None:
    raw, context, _ = evidence()
    changed = bytearray(raw[:2048])
    pack_into("<" + fmt, changed, offset, value)
    with pytest.raises(ValueError, match="prepared"):
        decode_prepared_contact(bytes(changed), context)


def test_delivery_scratch_register_cannot_be_promoted_to_flags() -> None:
    raw, context, _ = evidence()
    original = decode_prepared_contact(raw[:2048], context)
    changed = bytearray(raw[:2048])
    pack_into("<I", changed, 1728, 0xFFFFFFFF)
    pack_into("<Q", changed, 80 + 8 * 8, 0xFFFFFFFF)
    assert decode_prepared_contact(bytes(changed), context) == original


def test_coherently_relabelled_wheel_still_requires_actual_argument_register() -> None:
    raw, context, _ = evidence()
    original = decode_prepared_contact(raw[:2048], context)
    target = (original.wheel_index + 1) % len(context.wheel_models)
    changed = bytearray(raw[:2048])
    pack_into("<I", changed, 1732, target)
    pack_into("<Q", changed, 1448, context.vehicle + 0x388 + target * 8)
    pack_into("<Q", changed, 1456, context.wheel_models[target])
    with pytest.raises(ValueError, match="provenance mismatch"):
        decode_prepared_contact(bytes(changed), context)


def test_wrapping_wall_clock_cannot_alias_actual_race_event_time() -> None:
    raw, context, _ = evidence()
    original = decode_prepared_contact(raw[:2048], context)
    changed = bytearray(raw[:2048])
    pack_into("<Q", changed, 80 + 14 * 8, original.event_time + 2**32)
    with pytest.raises(ValueError, match="event time unavailable"):
        decode_prepared_contact(bytes(changed), context)
