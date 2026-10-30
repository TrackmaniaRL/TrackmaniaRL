"""Synthetic wheel and normalization-owner reads retain bindings."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from struct import pack, pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.wheel_contact_capture import (
    RECORD_BYTES,
    WheelContactContext,
    WheelContactSourceContext,
    decode_wheel_contact_source,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/wheel_contact_sources_v1"


def evidence() -> tuple[bytes, WheelContactSourceContext, dict]:
    raw = (FIXTURE / "contacts.bin").read_bytes()
    saved = json.loads((FIXTURE / "context.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == saved["sha256"]
    context = WheelContactSourceContext(
        WheelContactContext(**saved["context"]),
        tuple(saved["wheel_models"]),
        saved["vehicle_model"],
        saved["normalization_owner"],
        bytes.fromhex(saved["normalization_owner_header_hex"]),
    )
    return raw, context, saved


def test_synthetic_all_four_wheels_and_owner_thresholds() -> None:
    raw, context, saved = evidence()
    results = [
        decode_wheel_contact_source(raw[i : i + RECORD_BYTES], context)
        for i in range(0, len(raw), RECORD_BYTES)
    ]
    assert len(results) == saved["records"] == 23
    assert {r.contact.wheel_index for r in results} == {0, 1, 2, 3}
    assert {r.contact.material for r in results} == {9, 16}
    assert any(r.wheel_model % 8 == 4 for r in results)
    for result in results:
        assert result.wheel_category == saved["expected_categories"][result.contact.wheel_index]
        assert (
            pack(
                "<4f", result.wheel_high, result.wheel_low, result.main_low, result.main_high
            ).hex()
            == saved["expected_thresholds_hex"]
        )
        assert not hasattr(result, "collision_detected")


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (1440, "Q", 0),
        (928, "I", 10),
        (928, "I", 58),
        (928, "I", 106),
        (1448, "Q", 0),
        (1456, "Q", 0),
        (1480, "I", 3),
        (1520, "Q", 0),
        (1600, "Q", 0),
        (1608, "Q", 0),
        (1640, "f", float("nan")),
        (1644, "f", 0.0),
        (1648, "f", 150.0),
        (1652, "f", float("inf")),
    ],
)
def test_missing_foreign_or_invalid_source_is_rejected(
    offset: int, fmt: str, value: int | float
) -> None:
    raw, context, _ = evidence()
    damaged = bytearray(raw[:RECORD_BYTES])
    pack_into("<" + fmt, damaged, offset, value)
    with pytest.raises(ValueError, match=r"provenance|thresholds"):
        decode_wheel_contact_source(bytes(damaged), context)


def test_independently_bound_context_cannot_be_replaced_by_record_values() -> None:
    raw, context, _ = evidence()
    for changed in (
        replace(context, wheel_models=context.wheel_models[:-1]),
        replace(context, wheel_models=tuple(p + 1 for p in context.wheel_models)),
        replace(context, normalization_owner=context.vehicle_model),
        replace(context, normalization_owner_header=bytes(32)),
    ):
        with pytest.raises(ValueError, match=r"sources required|provenance"):
            decode_wheel_contact_source(raw[:RECORD_BYTES], changed)
