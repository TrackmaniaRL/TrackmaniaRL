"""Source and retained-ledger clock tests; never execute an engine or game."""

from __future__ import annotations

import copy
import struct

import pytest

from scripts.headless_native_clock import (
    WINDOWS,
    ClockProofError,
    audit_reference,
    check_unit_windows,
    controlled_limit,
    elf_offset,
)


def windows() -> dict[str, bytes]:
    return {name: bytes.fromhex(expected) for name, (_, expected) in WINDOWS.items()}


def test_unit_is_bound_to_actual_constant_operand_and_simulation_dt_call() -> None:
    proof = check_unit_windows(windows(), bytes.fromhex("6f12833a"))
    assert proof["nominal_scale"] == {"numerator": 1, "denominator": 1000}
    assert proof["nominal_seconds_per_interval"] == 0.01
    assert proof["binary32_scale_value"] == 0.0010000000474974513
    assert proof["dt10_binary32_product_nearest_even"] != 0.01


@pytest.mark.parametrize("name", list(WINDOWS))
def test_changed_instruction_or_linkage_is_not_accepted(name: str) -> None:
    spans = windows()
    raw = bytearray(spans[name])
    raw[-1] ^= 1
    spans[name] = bytes(raw)
    with pytest.raises(ClockProofError, match="instruction guard"):
        check_unit_windows(spans, bytes.fromhex("6f12833a"))


def test_a_guessed_scale_or_incomplete_call_chain_is_rejected() -> None:
    with pytest.raises(ClockProofError, match="scale differs"):
        check_unit_windows(windows(), struct.pack("<f", 0.01))
    spans = windows()
    spans.pop("dt_component_call")
    with pytest.raises(ClockProofError, match="all native"):
        check_unit_windows(spans, bytes.fromhex("6f12833a"))


def image() -> bytes:
    raw = bytearray(0x300)
    raw[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", raw, 32, 64)
    struct.pack_into("<HH", raw, 54, 56, 1)
    struct.pack_into("<IIQQQQQQ", raw, 64, 1, 5, 0x100, 0x1000, 0, 0x100, 0x200, 0x1000)
    return bytes(raw)


def test_elf_mapping_uses_file_backed_range_without_assuming_va_equals_offset() -> None:
    assert elf_offset(image(), 0x1020, 4) == 0x120
    with pytest.raises(ClockProofError, match="file-backed"):
        elf_offset(image(), 0x1100, 4)
    with pytest.raises(ClockProofError, match="ELF64"):
        elf_offset(bytes(0x300), 0x1020, 4)


def test_ambiguous_elf_segments_cannot_select_a_convenient_byte_window() -> None:
    raw = bytearray(image())
    struct.pack_into("<H", raw, 56, 2)
    raw[120:176] = raw[64:120]
    with pytest.raises(ClockProofError, match="exactly one"):
        elf_offset(bytes(raw), 0x1020, 4)


def fixture() -> tuple[dict, list[dict]]:
    reader = bytearray(256)
    struct.pack_into("<I", reader, 116, 4152)
    post = bytearray(256)
    post[:8] = b"TMPOST03"
    for offset, value in (
        (8, 256),
        (12, 3),
        (16, 50),
        (20, 50),
        (112, 2210),
        (116, 2210),
        (136, 10),
        (140, 10),
        (144, 2210),
        (148, 75),
        (212, 10),
    ):
        struct.pack_into("<I", post, offset, value)
    result = {
        "authorization": {"nativeEventHex": [post.hex()]},
        "actualIterations": 1,
        "observation": {"nativeT": 2210},
        "terminated": True,
        "truncated": False,
        "outcome": {
            "qualifiedBy": "env002.full-original-validator+TMRESULT",
            "computedResult": {"NbCheckpoints": 1, "NbRespawns": 0, "Time": 17715, "Score": 0},
            "validationAcceptance": False,
        },
    }
    report = {
        "reset": {"state": {"nativeT": 2200, "currentRecordIndex": 152, "readerHex": reader.hex()}},
        "rows": 1,
        "actual_intervals": 1,
        "last_result": result,
    }
    return report, [{"decision": 0, "result": copy.deepcopy(result)}]


def test_source_range_follows_tape_guard_and_is_not_a_windows_observation_range() -> None:
    assert controlled_limit(2200, 152, 4152) == 42190
    report, rows = fixture()
    audit = audit_reference(report, rows)
    assert audit["actual_post_intervals"] == 1
    assert audit["controlled_maximum_from_reader_guard"] == 42190
    assert audit["validation_acceptance"] is False


@pytest.mark.parametrize("offset", [112, 116, 136, 140, 144])
def test_raw_clock_or_actual_integration_dt_disagreement_is_not_rounded(offset: int) -> None:
    report, rows = fixture()
    raw = bytearray.fromhex(rows[0]["result"]["authorization"]["nativeEventHex"][0])
    struct.pack_into("<I", raw, offset, 11)
    rows[0]["result"]["authorization"]["nativeEventHex"] = [raw.hex()]
    with pytest.raises(ClockProofError, match="counter/frame_dt"):
        audit_reference(report, rows)


def test_missing_reference_rows_and_nonstandard_reset_do_not_expand_scope() -> None:
    report, _ = fixture()
    with pytest.raises(ClockProofError, match="complete retained"):
        audit_reference(report, [])
    with pytest.raises(ClockProofError, match="exact V7"):
        controlled_limit(2210, 153, 4152)
