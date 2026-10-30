"""Strict clock diagnostics use synthetic retained bytes; no live runtimes."""

from __future__ import annotations

import copy
import hashlib
import struct
from typing import Any

import pytest

from scripts.headless_clock_alignment import AlignmentError
from scripts.headless_clock_r5 import (
    ClockEvidenceError,
    analyze_pair,
    content_join,
    inspect_case,
    inspect_windows,
)
from scripts.headless_telemetry006 import COLLECTOR_VERSION, FIELD_SCHEMA, PROTOCOL


def _raw(tick: int, *, gas: int = 0) -> list[float]:
    values = [0.0] * 33
    values[3] = tick * 10
    values[4:10] = [tick / 4, 10.0, 20.0, tick / 8, 0.0, tick / 16]
    values[31] = gas
    return list(struct.unpack("<33f", struct.pack("<33f", *values)))


def _packet(raw: list[float], sequence: int) -> dict[str, Any]:
    binary = struct.pack("<33f", *raw)
    return {
        "raw33": raw,
        "raw33_le_hex": binary.hex(),
        "fresh_metadata": {
            "payload_sha256": hashlib.sha256(binary).hexdigest(),
            "host_request_perf_counter_ns": sequence * 10000,
            "host_received_perf_counter_ns": sequence * 10000 + 100,
            "producer": {
                "protocol": PROTOCOL,
                "collector_version": COLLECTOR_VERSION,
                "field_schema": FIELD_SCHEMA,
                "encoding": "little-endian-float32",
                "status": "ok",
                "field_count": 33,
                "payload_bytes": 132,
                "producer_clock": "Openplanet.Time.Now/raw",
                "nonce": "1" * 32,
                "sequence": sequence,
                "producer_begin_Time_Now_raw": sequence,
                "producer_end_Time_Now_raw": sequence,
                "actual_map_uid": "fixed-test-map",
                "simulation_phase_qualified": False,
                "applied_input_acknowledgement": False,
                "atomic_snapshot_claimed": False,
            },
        },
    }


def _case(*, missing: int | None = None, duplicate: bool = False) -> dict[str, Any]:
    initial = _raw(1)
    current = initial
    sequence = 1
    rows = []
    for decision in range(2):
        before = current
        packets = []
        applied = sequence * 10000 + 200
        for tick in range(2 + decision * 5, 7 + decision * 5):
            if tick == missing:
                continue
            sequence += 1
            current = _raw(tick)
            packets.append(_packet(current, sequence))
            if duplicate and tick == 3:
                changed = copy.deepcopy(current)
                changed[30] = struct.unpack("<f", struct.pack("<I", 1))[0]
                sequence += 1
                packets.append(_packet(changed, sequence))
        rows.append(
            {
                "action": 36,
                "before_raw33": before,
                "after_raw33": current,
                "issued_gbs": [0, 0, 0],
                "apply_start_host_ns": applied,
                "apply_finished_host_ns": applied + 10,
                "packets": packets,
            }
        )
    return {
        "completed": True,
        "initial_raw33": initial,
        "initial_packet": _packet(initial, 1),
        "rows": rows,
    }


def _audit(*, missing: int | None = None, stationary: bool = False) -> dict[str, Any]:
    samples = []
    for tick in range(1, 802):
        if tick == missing:
            continue
        raw = _raw(tick)
        if stationary:
            raw[4:10] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        samples.append(
            {
                "race_ms": tick * 10,
                "raw33": raw,
                "physics6_hex": struct.pack("<6f", *raw[4:10]).hex(),
            }
        )
    return {
        "cases": {
            "any-name": {
                "complete_tick_evidence": missing is None,
                "ambiguous_physics_ticks": [],
                "ambiguous_sgb_ticks": [],
                "missing_race_ticks": [] if missing is None else [missing * 10],
                "samples": samples,
            }
        }
    }


def _native(audit: dict[str, Any]) -> dict[str, Any]:
    rows = []
    windows = {
        sample["race_ms"]: sample["raw33"] for sample in audit["cases"]["any-name"]["samples"]
    }
    for tick in range(1, 802):
        raw = windows.get(tick * 10, _raw(tick))
        post = bytearray(256)
        post[:8] = b"TMPOST03"
        struct.pack_into("<I", post, 112, tick * 10 + 2190)
        struct.pack_into("<6f", post, 168, *raw[4:10])
        rows.append({"native_t": tick * 10 + 2190, "postHex": post.hex()})
    return {"passed": True, "rows": rows}


def test_complete_case_requires_every_tick_and_retained_exact_packet() -> None:
    result = inspect_case(_case(), [36, 36], minimum_span_ms=100)
    assert result["complete_tick_evidence"]
    assert result["distinct_race_ticks"] == 11
    assert result["clock_alignment_qualified"] is False
    assert result["input_consumption_acknowledgement_qualified"] is False


def test_missing_tick_is_retained_and_never_interpolated() -> None:
    result = inspect_case(_case(missing=3), [36, 36], minimum_span_ms=100)
    assert result["complete_tick_evidence"] is False
    assert result["missing_race_ticks"] == [30]
    assert len(result["samples"]) == 10


def test_one_ulp_sgb_ambiguity_is_rejected_without_rounding() -> None:
    result = inspect_case(_case(duplicate=True), [36, 36], minimum_span_ms=100)
    assert result["complete_tick_evidence"] is False
    assert result["ambiguous_sgb_ticks"] == [30]
    assert result["ambiguous_physics_ticks"] == []


def test_packet_hash_failure_never_gets_capture_qualification() -> None:
    case = _case()
    case["initial_packet"]["fresh_metadata"]["payload_sha256"] = "0" * 64
    result = inspect_windows(
        {"passed": True, "cases": {"anything": case}}, {"anything": [36, 36]}, minimum_span_ms=100
    )
    assert result["capture_passed"] is True
    assert result["complete_tick_evidence"] is False
    assert "SHA256" in result["cases"]["anything"]["error"]


def test_observation_only_join_retains_hole_and_never_qualifies_full_clock() -> None:
    audit = _audit(missing=400)
    result = content_join(audit, _native(audit), case="any-name")
    assert result["observed_samples_affine_relation_verified"] is True
    assert result["exact_retained_matches"] == 800
    assert result["windows_missing_race_ticks"] == [4000]
    assert result["clock_alignment_qualified"] is False
    assert result["continuous_interval_qualified"] is False
    assert result["input_waveform_qualified"] is False
    assert result["paired_training_qualified"] is False
    assert "relative_clock_alignment_qualified" not in result["fit_and_heldout_evidence"]
    with pytest.raises(ClockEvidenceError, match="complete Windows"):
        analyze_pair(audit, _native(audit), case="any-name")


def test_stationary_repeated_states_cannot_prove_clock_alignment() -> None:
    audit = _audit(stationary=True)
    with pytest.raises(AlignmentError, match="unique exact"):
        content_join(audit, _native(audit), case="any-name")


def test_native_gap_is_rejected_even_for_observation_only_join() -> None:
    audit = _audit(missing=400)
    native = _native(audit)
    del native["rows"][500]
    with pytest.raises(ClockEvidenceError, match="every actual native"):
        content_join(audit, native, case="any-name")


def test_matching_subset_does_not_hide_one_retained_mismatching_state() -> None:
    audit = _audit(missing=400)
    native = _native(audit)
    post = bytearray.fromhex(native["rows"][500]["postHex"])
    struct.pack_into("<f", post, 168, 999.0)
    native["rows"][500]["postHex"] = post.hex()
    result = content_join(audit, native, case="any-name")
    assert result["observed_samples_affine_relation_verified"] is False
    assert result["exact_retained_matches"] == 799
    assert result["unmatched_retained_windows_ticks"] == [5010]
