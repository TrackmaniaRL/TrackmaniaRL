"""Literal actual nativeGoal records; no synthetic UI equivalence."""

import hashlib
import json
import os
from pathlib import Path
from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import (
    HeldNativeGoalReader,
    NativeGoalContext,
    decode_goal_boundary,
    decode_goal_facts,
)


def fixture() -> tuple[NativeGoalContext, dict[str, bytes]]:
    root = Path(__file__).parents[2] / "fixtures/full_qr/goal_delivery380"
    data = json.loads((root / "context.json").read_text())
    files = {entry["name"]: (root / entry["name"]).read_bytes() for entry in data["files"]}
    for entry in data["files"]:
        assert hashlib.sha256(files[entry["name"]]).hexdigest() == entry["sha256"]
    return NativeGoalContext(data["pid"], data["meta"]), files


def test_actual_goal_waits_for_event_time_after_its_callback() -> None:
    context, files = fixture()
    facts = decode_goal_facts(files["facts.bin"], context)
    assert len(facts) == 1
    assert facts[0].entry_name == "Goal"
    assert facts[0].callback_native_t == 36490
    assert facts[0].event_native_t == 36505
    pending = decode_goal_boundary(files["pending.bin"], facts, context)
    active = decode_goal_boundary(files["active.bin"], facts, context)
    assert pending.native_t == 36500
    assert pending.goal_active is False
    assert active.native_t == 36550
    assert active.goal_active is True
    assert active.client_ui_finish_proven is False


def test_actual_initial_receipt_has_no_native_goal_facts() -> None:
    context, files = fixture()
    boundary = decode_goal_boundary(files["initial.bin"], (), context)
    assert boundary.native_t == 2200
    assert boundary.pairs == 0
    assert boundary.goals == 0
    assert boundary.last_goal_native_t is None
    assert boundary.goal_active is False
    assert boundary.client_ui_finish_proven is False


def test_payload_word_at24_is_not_the_classifier_result() -> None:
    context, files = fixture()
    # Actual377 failed an incorrect new guard comparing this payload to3.
    # The actual caller's cmpEAX3/seteR9 produces the endpoint argument.
    raw = files["facts.bin"]
    assert raw[240 + 416 + 24 : 240 + 416 + 28] == bytes.fromhex("00d02703")
    assert decode_goal_facts(raw, context)[0].endpoint_requested is True


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (16, "<I", 1),
        (24, "<Q", 2),
        (88, "<Q", 1),
        (96, "<Q", 1),
        (104, "<Q", 1),
        (112, "<Q", 1),
        (752, "<Q", 1),
        (240 + 368, "<Q", 1),  # Wrong native writer caller.
    ],
)
def test_wrong_goal_identity_and_provenance_are_refused(offset: int, fmt: str, value: int) -> None:
    context, files = fixture()
    raw = bytearray(files["facts.bin"])
    pack_into(fmt, raw, offset, value)
    with pytest.raises(ValueError, match=r"nativeGoal|writer|event caller"):
        decode_goal_facts(bytes(raw), context)


def test_goal_requires_the_same_actual_map_entry_and_endpoint_bit() -> None:
    context, files = fixture()
    raw = bytearray(files["facts.bin"])
    for base in (0, 768):
        raw[base + 120 + 0x51] &= 0xFE
    with pytest.raises(ValueError, match="actual Goal map entry"):
        decode_goal_facts(bytes(raw), context)


def test_missing_committed_goal_pair_is_not_negative_goal_coverage() -> None:
    context, files = fixture()
    with pytest.raises(ValueError, match="prefix mismatch"):
        decode_goal_boundary(files["active.bin"], (), context)
    with pytest.raises(ValueError, match="complete bounded"):
        decode_goal_facts(files["facts.bin"][:768], context)


@pytest.mark.skipif(not hasattr(os, "pread"), reason="held goal reader requires Linux pread")
def test_actual_held_goal_stream_waits_through_future_event(tmp_path: Path) -> None:
    context, files = fixture()
    receipts, facts = tmp_path / "receipts.bin", tmp_path / "facts.bin"
    receipts.write_bytes(files["receipts.bin"][:192])
    facts.write_bytes(b"")
    reader = HeldNativeGoalReader(receipts, facts, context)
    assert reader.read(2200).goal_active is False
    pending_size = ((36500 - 2190) // 10 + 1) * 96
    with receipts.open("ab") as stream:
        stream.write(files["receipts.bin"][192:pending_size])
    with facts.open("ab") as stream:
        stream.write(files["facts.bin"])
    assert reader.read(36500).goal_active is False
    with receipts.open("ab") as stream:
        stream.write(files["receipts.bin"][pending_size:])
    assert reader.read(36550).goal_active is True


@pytest.mark.skipif(not hasattr(os, "pread"), reason="held goal reader requires Linux pread")
def test_a_future_file_or_missing_goal_data_cannot_supply_a_boundary(tmp_path: Path) -> None:
    context, files = fixture()
    receipts, facts = tmp_path / "receipts.bin", tmp_path / "facts.bin"
    receipts.write_bytes(files["receipts.bin"])
    facts.write_bytes(files["facts.bin"])
    reader = HeldNativeGoalReader(receipts, facts, context)
    with pytest.raises(ValueError, match="does not match held"):
        reader.read(36500)
    facts.write_bytes(b"")
    with pytest.raises(ValueError, match="prefix mismatch"):
        reader.read(36550)
