"""Pure phase/content/float-bit tests; no native engine or game execution."""

from __future__ import annotations

import struct
from typing import Any, cast

import pytest

from scripts.headless_replay_windows_tapes import ObservedSession
from scripts.input_consumer_native_replay import (
    anchor_index,
    close_case,
    compare,
    phase_plan,
    word_bytes,
)


def pose(y_word: int = 0x3F800000) -> dict[str, Any]:
    return {"raw_body_bits_u32": [0x3F800000, 0, 0, 0, 0, y_word, 0, 0, 0, 0]}


def post(row: dict[str, Any], native_t: int = 2260) -> bytes:
    value = bytearray(256)
    value[152:192] = word_bytes(row["raw_body_bits_u32"])
    struct.pack_into("<I", value, 112, native_t)
    return bytes(value)


def test_anchor_uses_unique_physical_content_without_clock_offset_guessing() -> None:
    actual = pose()
    different = pose(0x3F800001)
    neutral = [{"postHex": post(different).hex()}, {"postHex": post(actual).hex()}]
    assert anchor_index([actual], neutral) == 1
    neutral.append({"postHex": post(actual, 9000).hex()})
    with pytest.raises(ValueError, match="exactly one"):
        anchor_index([actual], neutral)


def test_phase_contracts_have_no_invented_endpoint_or_previous_input() -> None:
    a = phase_plan("A-current-to-next", 199, 2250)
    b = phase_plan("B-previous-to-next", 199, 2250)
    c = phase_plan("C-current-to-same", 199, 2250)
    assert len(a["pairs"]) == 198
    assert a["pairs"][0] == (0, 1)
    assert a["pairs"][-1] == (197, 198)
    assert b["start_pose_index"] == 1
    assert b["pairs"][0] == (0, 2)
    assert len(b["pairs"]) == 197
    assert c["start_native_t"] == 2240
    assert c["pairs"][0] == (0, 0)
    assert c["pairs"][-1] == (198, 198)
    with pytest.raises(ValueError, match="explicit A/B/C"):
        phase_plan("best-fitting-offset", 199, 2250)


def test_vertical_one_ulp_is_not_hidden_by_float_tolerance_or_quaternion_fields() -> None:
    wanted = pose()
    exact = compare(post(wanted), wanted)
    assert exact["physics6_exact"] is True
    changed = compare(post(pose(0x3F800001)), wanted)
    assert changed["physics6_exact"] is False
    assert changed["Y_difference"] == 2**-23
    assert changed["VY_difference"] == 0
    quaternion_only = pose()
    quaternion_only["raw_body_bits_u32"][1] = 1
    changed_q = compare(post(quaternion_only), wanted)
    assert changed_q["physics6_exact"] is True
    assert changed_q["physical10_exact"] is False


@pytest.mark.parametrize("invalid", [[True], [-1], [0x100000000], [0.0]])
def test_uint32_evidence_cannot_be_coerced_or_rounded(invalid: list[Any]) -> None:
    with pytest.raises(ValueError, match="literal uint32"):
        word_bytes(invalid)


def test_adjacent_actual_steering_values_retain_their_exact_wire_bits() -> None:
    assert word_bytes([0xBE2952A5]).hex() == "a55229be"
    assert word_bytes([0xBE2952A5]) != word_bytes([0xBE2952A4])


def test_close_failure_preserves_original_error_and_cleanup_diagnostics() -> None:
    class BrokenClose:
        def __init__(self) -> None:
            self.cleanup: dict[str, Any] = {
                "remainingOwned": [123],
                "error": "supervisor proof missing",
            }

        def close(self, *, error: bool) -> None:
            assert error is True
            raise OSError("actual close failure")

    broken = BrokenClose()
    case: dict[str, Any] = {"completed": False}
    with pytest.raises(OSError, match="actual close failure"):
        close_case(cast(ObservedSession, broken), case)
    assert case["cleanup"] == broken.cleanup


def test_missing_or_nonempty_zero_owned_cleanup_proof_fails() -> None:
    class Closed:
        def __init__(self) -> None:
            self.cleanup: dict[str, Any] = {}

        def close(self, *, error: bool) -> None:
            assert error is False

    for proof in ({}, {"remainingOwned": [123]}):
        closed = Closed()
        closed.cleanup = proof
        case: dict[str, Any] = {"completed": True}
        with pytest.raises(RuntimeError, match="zero-owned cleanup proof"):
            close_case(cast(ObservedSession, closed), case)
        assert case["cleanup"] == proof
