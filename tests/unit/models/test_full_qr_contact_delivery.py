"""Synthetic source streams and faults; no claimed motor qualification."""

import hashlib
import json
import os
from pathlib import Path
from struct import pack_into, unpack_from
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.contact_delivery import (
    HeldContactDeliveryReader,
    decode_contact_receipt,
)
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact_capture import (
    PreparedImpactContext,
)


def fixture() -> tuple[dict[str, Any], PreparedImpactContext, bytes, bytes]:
    root = Path(__file__).parents[2] / "fixtures/full_qr/contact_delivery361"
    data = json.loads((root / "context.json").read_text())
    for entry in data["files"]:
        assert hashlib.sha256((root / entry["name"]).read_bytes()).hexdigest() == entry["sha256"]
    values = dict(data["context"])
    for name in ("normalization_header", "normalization_thresholds"):
        values[name] = bytes.fromhex(values[name])
    values["wheel_models"] = tuple(values["wheel_models"])
    values["wheel_headers"] = tuple(bytes.fromhex(x) for x in values["wheel_headers"])
    return (
        data,
        PreparedImpactContext(**values),
        (root / "receipts.bin").read_bytes(),
        (root / "contacts.bin").read_bytes(),
    )


def test_synthetic_reader_receipts_preserve_counts_and_timestamps() -> None:
    data, context, receipts, _ = fixture()
    first = decode_contact_receipt(receipts[:96], context)
    initial = decode_contact_receipt(receipts[96:192], context)
    final = decode_contact_receipt(receipts[-96:], context)
    assert (first.native_t, first.contacts, first.last_callback, first.last_event) == (
        2190,
        0,
        0,
        0,
    )
    assert (initial.native_t, initial.contacts) == (2200, 4)
    assert (initial.last_callback, initial.last_event) == (2190, 2200)
    assert (final.native_t, final.contacts) == (data["final_native_t"], data["contacts"])


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (16, "<I", 1),
        (24, "<Q", 1),
        (32, "<I", 2190),
        (36, "<I", 1),
        (40, "<Q", 100001),
        (48, "<I", 99999),
        (52, "<I", 0),
        (56, "<Q", 1),
        (64, "<Q", 1),
        (72, "<Q", 1),
        (80, "<Q", 1),
        (88, "<Q", 0),
    ],
)
def test_wrong_actor_clock_and_delivered_count_are_refused(
    offset: int, fmt: str, value: int
) -> None:
    _, context, receipts, _ = fixture()
    raw = bytearray(receipts[-96:])
    pack_into(fmt, raw, offset, value)
    with pytest.raises(ValueError, match="contact"):
        decode_contact_receipt(bytes(raw), context)


@pytest.mark.skipif(not hasattr(os, "pread"), reason="held native reader uses Linux pread")
def test_two_stage_delivery_consumes_every_intermediate_boundary(tmp_path: Path) -> None:
    data, context, receipts, contacts = fixture()
    receipt_path, contact_path = tmp_path / "receipts.bin", tmp_path / "contacts.bin"
    receipt_path.write_bytes(receipts[:192])
    contact_path.write_bytes(contacts[: 4 * 512])
    reader = HeldContactDeliveryReader(receipt_path, contact_path, context)
    initial = reader.take(2200)
    assert [x.receipt.native_t for x in initial] == [2190, 2200]
    assert [len(x.contacts) for x in initial] == [0, 4]
    # Append synthetic later bytes between held boundaries.
    with receipt_path.open("ab") as stream:
        stream.write(receipts[192:])
    with contact_path.open("ab") as stream:
        stream.write(contacts[4 * 512 :])
    later = reader.take(data["final_native_t"])
    assert sum(len(x.contacts) for x in later) == data["contacts"] - 4
    assert len(later) == len(receipts) // 96 - 2
    assert all(x.event_time <= batch.receipt.native_t for batch in later for x in batch.contacts)
    with pytest.raises(ValueError, match="fresh held"):
        reader.take(data["final_native_t"])


@pytest.mark.skipif(not hasattr(os, "pread"), reason="held native reader uses Linux pread")
def test_absent_promised_contact_fails_without_consuming_partial_state(tmp_path: Path) -> None:
    data, context, receipts, contacts = fixture()
    receipt_path, contact_path = tmp_path / "receipts.bin", tmp_path / "contacts.bin"
    receipt_path.write_bytes(receipts)
    contact_path.write_bytes(contacts[:-512])
    reader = HeldContactDeliveryReader(receipt_path, contact_path, context)
    with pytest.raises(ValueError, match="promises records absent"):
        reader.take(data["final_native_t"])
    assert reader._receipt_sequence == 0
    assert reader._contacts == 0


@pytest.mark.skipif(not hasattr(os, "pread"), reason="held native reader uses Linux pread")
def test_receipt_cannot_hide_extra_contact_bytes(tmp_path: Path) -> None:
    _, context, receipts, contacts = fixture()
    receipt_path, contact_path = tmp_path / "receipts.bin", tmp_path / "contacts.bin"
    receipt_path.write_bytes(receipts[:192])
    contact_path.write_bytes(contacts[: 5 * 512])
    reader = HeldContactDeliveryReader(receipt_path, contact_path, context)
    assert unpack_from("<Q", receipts, 96 + 40)[0] == 4
    with pytest.raises(ValueError, match="uncredited contact"):
        reader.take(2200)
