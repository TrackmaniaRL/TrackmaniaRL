"""Offline literal 007/API analysis; content joins do not acknowledge commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from scripts.headless_clock_r5 import _packet as validated_fresh_packet
from scripts.headless_windows_physical_probe import EXECUTABLE_SHA256, OPENPLANET_SHA256
from scripts.input_consumer_probe import verify_capture, verify_restoration


class ConsumerEvidenceError(ValueError):
    """The retained records do not meet the literal diagnostic contract."""


def _words(value: Any, length: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in value)
    ):
        raise ConsumerEvidenceError("literal uint32 source words required")
    return tuple(value)


def _float(word: int) -> float:
    value = float(struct.unpack("<f", struct.pack("<I", word))[0])
    if not math.isfinite(value):
        raise ConsumerEvidenceError("nonfinite control value cannot receive an ULP comparison")
    return value


def ulp_distance(first: int, second: int) -> int:
    """Distance in ordered finite binary32 words; no rounding or tolerance."""
    _float(first), _float(second)
    a = (~first & 0xFFFFFFFF) if first & 0x80000000 else first ^ 0x80000000
    b = (~second & 0xFFFFFFFF) if second & 0x80000000 else second ^ 0x80000000
    return abs(a - b)


def _observed(words: tuple[int, ...]) -> tuple[int, ...]:
    # Native observed/cache braking is an integer boolean. Explicitly encode
    # that boolean as Connect's float32 field, retaining original words too.
    if words[2] not in (0, 1):
        raise ConsumerEvidenceError("observed braking word must be the literal integer0 or1")
    return words[0], words[1], 0x3F800000 if words[2] else 0


def consumer_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    previous = -1
    for index, row in enumerate(rows):
        actual = _words(row["actual_control_bits_u32"], 4)
        vehicle = _words(row["vehicle_observed_input_bits_u32"], 3)
        cached = _words(row["participant_cached_input_bits_u32"], 3)
        body = _words(row["raw_body_bits_u32"], 10)
        if (
            type(row["sequence"]) is not int
            or row["sequence"] != index
            or type(row["Time_Now_raw"]) is not int
            or row["Time_Now_raw"] < 0
            or row["Time_Now_raw"] < previous
        ):
            raise ConsumerEvidenceError(
                "literal acquisition order and monotonic host metadata required"
            )
        previous = row["Time_Now_raw"]
        step = row["R13D_step_argument_raw"]
        if type(step) is not int or not 0 <= step <= 0xFFFFFFFF:
            raise ConsumerEvidenceError("literal R13D uint32 required")
        sgbs = {
            "actual": (actual[2], actual[0], actual[1]),
            "vehicle": _observed(vehicle),
            "cache": _observed(cached),
        }
        for sgb in sgbs.values():
            for word in sgb:
                _float(word)
        result.append(
            {
                "sequence": index,
                "Time_Now_raw": previous,
                "R13D_raw": step,
                "physics6_hex": struct.pack("<6I", *body[4:10]).hex(),
                "physics6": list(struct.unpack("<6f", struct.pack("<6I", *body[4:10]))),
                "actual_control_bits_u32": list(actual),
                "vehicle_observed_bits_u32": list(vehicle),
                "participant_cached_bits_u32": list(cached),
                "actual_sgb_bits_u32": list(sgbs["actual"]),
                "vehicle_sgb_bits_u32": list(sgbs["vehicle"]),
                "cache_sgb_bits_u32": list(sgbs["cache"]),
                "actual_sgb_values": [_float(word) for word in sgbs["actual"]],
                "filtered_steer_value": _float(actual[3]),
                "actual_equals_vehicle_sgb": sgbs["actual"] == sgbs["vehicle"],
                "actual_equals_cache_sgb": sgbs["actual"] == sgbs["cache"],
                "vehicle_equals_cache_sgb": sgbs["vehicle"] == sgbs["cache"],
                "raw_equals_filtered_steer": actual[2] == actual[3],
                "raw_vehicle_steer_ulp": ulp_distance(actual[2], vehicle[0]),
                "raw_cache_steer_ulp": ulp_distance(actual[2], cached[0]),
                "vehicle_cache_steer_ulp": ulp_distance(vehicle[0], cached[0]),
            }
        )
    return result


def _api_packet(  # noqa: PLR0913 -- independent raw bytes, source identity and producer metadata
    raw: Any, *, source: str, index: int, metadata: Any = None
) -> dict[str, Any]:
    if (
        not isinstance(raw, list)
        or len(raw) != 33
        or not all(type(value) in (int, float) and math.isfinite(value) for value in raw)
    ):
        raise ConsumerEvidenceError("full finite actual API raw33 required")
    packet = struct.pack("<33f", *raw)
    if raw[3] < 0 or int(raw[3]) != raw[3]:
        raise ConsumerEvidenceError("actual integral API race milliseconds required")
    return {
        "source": source,
        "index": index,
        "race_ms": int(raw[3]),
        "physics6_hex": packet[16:40].hex(),
        "sgb_bits_u32": list(struct.unpack_from("<3I", packet, 120)),
        "producer_metadata": metadata,
    }


def content_matches(records: list[dict[str, Any]], api: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["physics6_hex"] for record in records)
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in api:
        by_signature[row["physics6_hex"]].append(row)
    matches, excluded = [], []
    for record in records:
        candidates = by_signature.get(record["physics6_hex"], [])
        times = {candidate["race_ms"] for candidate in candidates}
        reason = None
        if counts[record["physics6_hex"]] != 1:
            reason = "repeated-consumer-physical-signature"
        elif not candidates:
            reason = "no-retained-API-content-match"
        elif len(times) != 1:
            reason = "API-content-occurs-at-multiple-race-times"
        if reason is not None:
            excluded.append(
                {
                    "sequence": record["sequence"],
                    "reason": reason,
                    "candidate_race_times": sorted(times),
                }
            )
            continue
        patterns = sorted({tuple(candidate["sgb_bits_u32"]) for candidate in candidates})
        comparisons = []
        for pattern in patterns:
            comparisons.append(
                {
                    "API_sgb_bits_u32": list(pattern),
                    "API_sgb_values": [_float(word) for word in pattern],
                    "actual_equals_API_sgb": tuple(record["actual_sgb_bits_u32"]) == pattern,
                    "vehicle_equals_API_sgb": tuple(record["vehicle_sgb_bits_u32"]) == pattern,
                    "cache_equals_API_sgb": tuple(record["cache_sgb_bits_u32"]) == pattern,
                    "actual_API_steer_ulp": ulp_distance(
                        record["actual_sgb_bits_u32"][0], pattern[0]
                    ),
                    "vehicle_API_steer_ulp": ulp_distance(
                        record["vehicle_sgb_bits_u32"][0], pattern[0]
                    ),
                }
            )
        matches.append(
            {
                "consumer_sequence": record["sequence"],
                "R13D_raw": record["R13D_raw"],
                "race_ms": next(iter(times)),
                "consumer_Time_Now_raw": record["Time_Now_raw"],
                "physics6_hex": record["physics6_hex"],
                "moving_nonzero_velocity": any(value != 0 for value in record["physics6"][3:]),
                "API_packet_indices": [candidate["index"] for candidate in candidates],
                "API_input_ambiguous": len(patterns) != 1,
                "API_input_comparisons": comparisons,
                "fresh_producer_intervals": [
                    candidate["producer_metadata"]
                    for candidate in candidates
                    if candidate["producer_metadata"] is not None
                ],
            }
        )
    return {
        "matched_unique_consumer_states": len(matches),
        "matches": matches,
        "excluded_records": excluded,
        "ambiguous_API_input_matches": sum(match["API_input_ambiguous"] for match in matches),
        "same_record_atomic_observation_claimed": False,
    }


def observed_r13_relation(matches: list[dict[str, Any]]) -> dict[str, Any]:
    moving = [row for row in matches if row["moving_nonzero_velocity"]]
    report: dict[str, Any] = {
        "observed_content_relation_consistent": False,
        "simulation_clock_qualified": False,
        "input_consumption_acknowledged": False,
        "moving_unique_matches": len(moving),
        "literal_R13D_minus_API_race_histogram": dict(
            Counter(row["R13D_raw"] - row["race_ms"] for row in moving)
        ),
    }
    if len(moving) < 6:
        report["reason"] = "three fit and three held-out moving exact matches required"
        return report
    if any(
        b["R13D_raw"] <= a["R13D_raw"] or b["race_ms"] <= a["race_ms"] for a, b in pairwise(moving)
    ):
        report["reason"] = "R13D or content-matched race time is not strictly monotonic"
        return report
    fit, heldout = moving[::2], moving[1::2]
    first, last = fit[0], fit[-1]
    slope = Fraction(last["race_ms"] - first["race_ms"], last["R13D_raw"] - first["R13D_raw"])
    intercept = Fraction(first["race_ms"]) - slope * first["R13D_raw"]
    accepted = slope > 0 and all(
        slope * row["R13D_raw"] + intercept == row["race_ms"] for row in moving
    )
    report.update(
        observed_content_relation_consistent=accepted,
        fit_matches=len(fit),
        heldout_matches=len(heldout),
        slope={"numerator": slope.numerator, "denominator": slope.denominator},
        intercept={"numerator": intercept.numerator, "denominator": intercept.denominator},
        observed_race_span_ms=moving[-1]["race_ms"] - moving[0]["race_ms"],
        relationship="API race ms = slope * literal R13D + intercept, only listed content matches",
    )
    return report


def analyze(
    consumer: dict[str, Any], telemetry: dict[str, Any], diagnostic: dict[str, Any]
) -> dict[str, Any]:
    verify_restoration(consumer, consumer["nonce"])
    verify_capture(consumer)
    if diagnostic.get("passed") is not True or diagnostic.get("controller_closed") is not True:
        raise ConsumerEvidenceError("completed diagnostic and controller cleanup required")
    if diagnostic.get("original_code_restored_verified") is not True:
        raise ConsumerEvidenceError("actual hook restoration required")
    if diagnostic.get("actual_binary_sha256") != {
        "Trackmania.exe": EXECUTABLE_SHA256,
        "Openplanet.dll": OPENPLANET_SHA256,
    }:
        raise ConsumerEvidenceError("actual pinned binary identity required")
    if not any(
        message.get("report") == consumer for message in diagnostic["probe_protocol_messages"]
    ):
        raise ConsumerEvidenceError("diagnostic and consumer report do not bind the same result")
    records = consumer_records(consumer["records"])
    stream = [_api_packet(telemetry["initial_raw33"], source="initial", index=-1)]
    stream.extend(
        _api_packet(row["raw33"], source="original-stream", index=index)
        for index, row in enumerate(telemetry["packets"])
    )
    fresh = []
    for index, row in enumerate(telemetry.get("fresh006", [])):
        checked = validated_fresh_packet(
            {
                "raw33": row["raw33"],
                "raw33_le_hex": row["raw_payload_hex"],
                "fresh_metadata": row["metadata"],
            }
        )
        fresh.append(
            _api_packet(
                row["raw33"],
                source=row["boundary"],
                index=index,
                metadata={
                    "boundary": row["boundary"],
                    "producer_begin": checked["producer_begin"],
                    "producer_end": checked["producer_end"],
                    "host_request_ns": checked["host_request_ns"],
                    "host_received_ns": checked["host_received_ns"],
                },
            )
        )
    stream_join, fresh_join = content_matches(records, stream), content_matches(records, fresh)
    direct = {}
    for flag in (
        "actual_equals_vehicle_sgb",
        "actual_equals_cache_sgb",
        "vehicle_equals_cache_sgb",
        "raw_equals_filtered_steer",
    ):
        direct[flag] = {
            "equal": sum(row[flag] for row in records),
            "different": sum(not row[flag] for row in records),
        }
    for flag in ("raw_vehicle_steer_ulp", "raw_cache_steer_ulp", "vehicle_cache_steer_ulp"):
        direct[flag] = dict(Counter(row[flag] for row in records))
    per_channel = {}
    for left, right in (("actual", "vehicle"), ("actual", "cache"), ("vehicle", "cache")):
        per_channel[left + "_vs_" + right] = {
            channel: [
                row["sequence"]
                for row in records
                if row[left + "_sgb_bits_u32"][index] != row[right + "_sgb_bits_u32"][index]
            ]
            for index, channel in enumerate(("steer", "gas", "brake"))
        }
    transitions = {}
    for source in ("actual", "vehicle", "cache"):
        field = source + "_sgb_bits_u32"
        transitions[source] = [
            {
                "sequence": current["sequence"],
                "R13D_raw": current["R13D_raw"],
                "Time_Now_raw": current["Time_Now_raw"],
                "previous_sgb_bits_u32": previous[field],
                "new_sgb_bits_u32": current[field],
            }
            for previous, current in pairwise(records)
            if previous[field] != current[field]
        ]
    phase_rows = []
    for phase in telemetry["phases"]:
        start, end = phase["apply_start_host_ns"], phase["apply_finished_host_ns"]
        if type(start) is not int or type(end) is not int or end < start:
            raise ConsumerEvidenceError("actual command host interval required")
        boundaries = [
            row
            for row in telemetry.get("fresh006", [])
            if row["boundary"] in ("before_" + phase["name"], "after_" + phase["name"])
        ]
        before_boundary = next(
            (row for row in boundaries if row["boundary"] == "before_" + phase["name"]), None
        )
        phase_rows.append(
            {
                "name": phase["name"],
                "issued_gbs": phase["issued_gbs"],
                "before_observed_race_ms": phase["before_raw33"][3],
                "after_observed_race_ms": phase["after_raw33"][3],
                "apply_start_host_ns": start,
                "apply_finished_host_ns": end,
                "controller_call_host_ns": end - start,
                "fresh_boundaries": boundaries,
                "fresh_before_receive_to_apply_ns": None
                if before_boundary is None
                else start - before_boundary["metadata"]["host_received_perf_counter_ns"],
                "fresh_before_race_minus_phase_before_race_ms": None
                if before_boundary is None
                else before_boundary["raw33"][3] - phase["before_raw33"][3],
                "issued_command_consumption_acknowledged": False,
            }
        )
    return {
        "schema": "tmrl.input-consumer-literal-analysis/v1",
        "analysis_completed": True,
        "consumer_records": len(records),
        "original_API_packets": len(stream),
        "fresh006_boundary_samples": len(fresh),
        "same_callback_comparisons": direct,
        "same_callback_mismatch_sequences_by_channel": per_channel,
        "same_callback_control_transitions": transitions,
        "API_steer_word_counts": {
            name: dict(Counter(f"0x{row['sgb_bits_u32'][0]:08x}" for row in source))
            for name, source in (("original_stream", stream), ("fresh006", fresh))
        },
        "steer_word_counts": {
            source: dict(Counter(f"0x{row[field][index]:08x}" for row in records))
            for source, field, index in (
                ("raw", "actual_control_bits_u32", 2),
                ("filtered", "actual_control_bits_u32", 3),
                ("vehicle", "vehicle_observed_bits_u32", 0),
                ("cache", "participant_cached_bits_u32", 0),
            )
        },
        "literal_R13D_deltas": dict(
            Counter(b["R13D_raw"] - a["R13D_raw"] for a, b in pairwise(records))
        ),
        "callback_Time_Now_raw_deltas": dict(
            Counter(b["Time_Now_raw"] - a["Time_Now_raw"] for a, b in pairwise(records))
        ),
        "records": records,
        "original_stream_content_join": stream_join,
        "fresh006_content_join": fresh_join,
        "original_stream_R13D_relation": observed_r13_relation(stream_join["matches"]),
        "fresh006_R13D_relation": observed_r13_relation(fresh_join["matches"]),
        "phases_and_command_timestamps": phase_rows,
        "consumer_scope": consumer["phase"],
        "braking_comparison": (
            "raw boolean uint32 separately retained; encoded as float32 only for SGB comparison"
        ),
        "clock_alignment_qualified": False,
        "issued_command_ACK_qualified": False,
        "whole_step_phase_qualified": False,
        "paired_training_qualified": False,
        "atomic_cross_thread_snapshot_qualified": False,
        "scope": (
            "one pinned map/reset/run; literal callback fields "
            "and unique observation-content joins only"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = {
        "consumer": args.directory / "input-consumer007.json",
        "telemetry": args.directory / "original-connect.json",
        "diagnostic": args.directory / "diagnostic.json",
    }
    report = analyze(
        *(json.loads(paths[key].read_bytes()) for key in ("consumer", "telemetry", "diagnostic"))
    )
    report["sources"] = {
        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in paths.items()
    }
    report["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
