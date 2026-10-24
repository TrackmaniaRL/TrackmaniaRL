"""Literal 007 analysis tests; synthetic records and no runtime operations."""

from __future__ import annotations

import copy
import struct
from typing import Any

import pytest

from scripts.input_consumer_analysis import (
    ConsumerEvidenceError,
    consumer_records,
    content_matches,
    observed_r13_relation,
    ulp_distance,
)


def _record(index: int = 0) -> dict[str, Any]:
    words = list(struct.unpack("<10I", struct.pack("<10f", 1, 0, 0, 0, index, 10, 20, 1, 2, 3)))
    return {
        "sequence": index,
        "Time_Now_raw": 100 + index,
        "R13D_step_argument_raw": 10000 + index * 10,
        "actual_control_bits_u32": [0, 0, 0, 0],
        "vehicle_observed_input_bits_u32": [0, 0, 0],
        "participant_cached_input_bits_u32": [0, 0, 0],
        "raw_body_bits_u32": words,
    }


def test_one_ulp_difference_is_preserved_instead_of_rounded() -> None:
    row = _record()
    row["actual_control_bits_u32"] = [0x3F800000, 0, 0xBE2952A5, 0xBE2952A5]
    row["vehicle_observed_input_bits_u32"] = [0xBE2952A6, 0x3F800000, 0]
    row["participant_cached_input_bits_u32"] = [0xBE2952A6, 0x3F800000, 0]
    result = consumer_records([row])[0]
    assert result["raw_equals_filtered_steer"] is True
    assert result["actual_equals_vehicle_sgb"] is False
    assert result["raw_vehicle_steer_ulp"] == 1
    assert result["raw_cache_steer_ulp"] == 1
    assert result["vehicle_equals_cache_sgb"] is True
    assert ulp_distance(0xBE2952A5, 0xBE2952A6) == 1


def test_native_braking_boolean_is_encoded_explicitly_without_replacing_source_words() -> None:
    row = _record()
    row["actual_control_bits_u32"][1] = 0x3F800000
    row["vehicle_observed_input_bits_u32"][2] = 1
    row["participant_cached_input_bits_u32"][2] = 1
    result = consumer_records([row])[0]
    assert result["actual_equals_vehicle_sgb"] is True
    assert result["vehicle_observed_bits_u32"][2] == 1
    assert result["vehicle_sgb_bits_u32"][2] == 0x3F800000


def test_two_api_inputs_at_one_unique_physical_time_remain_ambiguous() -> None:
    records = consumer_records([_record()])
    base = {
        "index": 0,
        "race_ms": 20,
        "physics6_hex": records[0]["physics6_hex"],
        "sgb_bits_u32": [0xBE2952A5, 0x3F800000, 0],
        "producer_metadata": None,
    }
    different = {**base, "index": 1, "sgb_bits_u32": [0xBE2952A6, 0x3F800000, 0]}
    result = content_matches(records, [base, different])
    assert result["matched_unique_consumer_states"] == 1
    assert result["ambiguous_API_input_matches"] == 1
    assert len(result["matches"][0]["API_input_comparisons"]) == 2
    assert result["same_record_atomic_observation_claimed"] is False


def test_repeated_consumer_physics_cannot_be_selected_as_clock_anchor() -> None:
    first, second = _record(), _record(1)
    second["raw_body_bits_u32"] = copy.deepcopy(first["raw_body_bits_u32"])
    records = consumer_records([first, second])
    api = [
        {
            "index": 0,
            "race_ms": 20,
            "physics6_hex": records[0]["physics6_hex"],
            "sgb_bits_u32": [0, 0, 0],
            "producer_metadata": None,
        }
    ]
    result = content_matches(records, api)
    assert result["matched_unique_consumer_states"] == 0
    assert len(result["excluded_records"]) == 2


def test_r13_fit_and_heldout_relation_does_not_claim_simulation_clock_or_ack() -> None:
    rows = [
        {"R13D_raw": 10000 + i * 10, "race_ms": 20 + i * 10, "moving_nonzero_velocity": True}
        for i in range(8)
    ]
    result = observed_r13_relation(rows)
    assert result["observed_content_relation_consistent"] is True
    assert result["fit_matches"] == result["heldout_matches"] == 4
    assert result["simulation_clock_qualified"] is False
    assert result["input_consumption_acknowledged"] is False
    rows[3]["race_ms"] += 1
    assert observed_r13_relation(rows)["observed_content_relation_consistent"] is False


def test_negative_timestamp_and_nonfinite_control_are_rejected() -> None:
    row = _record()
    row["Time_Now_raw"] = -1
    with pytest.raises(ConsumerEvidenceError, match="monotonic"):
        consumer_records([row])
    row = _record()
    row["actual_control_bits_u32"][2] = 0x7FC00000
    with pytest.raises(ConsumerEvidenceError, match="nonfinite"):
        consumer_records([row])
