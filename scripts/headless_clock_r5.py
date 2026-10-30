"""Offline strict r5 clock/telemetry qualification; never launches a runtime.

All missing ticks and conflicting SGB values remain failures. Capture success,
sample-clock qualification and input-consumption acknowledgement are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from scripts.headless_clock_alignment import AlignmentError, Sample, align_clocks
from scripts.headless_telemetry006 import COLLECTOR_VERSION, FIELD_SCHEMA, PROTOCOL


class ClockEvidenceError(ValueError):
    """The retained data cannot support the requested strict qualification."""


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ClockEvidenceError(name + " must be a literal nonnegative integer")
    return value


def _bytes(raw: Any) -> bytes:
    if not isinstance(raw, list) or len(raw) != 33:
        raise ClockEvidenceError("a full raw33 observation is required")
    if not all(type(value) in (int, float) and math.isfinite(value) for value in raw):
        raise ClockEvidenceError("raw33 must contain finite actual numeric values")
    return struct.pack("<33f", *raw)


def _packet(source: dict[str, Any]) -> dict[str, Any]:
    packet = _bytes(source["raw33"])
    if bytes.fromhex(source["raw33_le_hex"]) != packet:
        raise ClockEvidenceError("raw33 values and exact retained binary disagree")
    metadata = source["fresh_metadata"]
    producer = metadata["producer"]
    expected = {
        "protocol": PROTOCOL,
        "collector_version": COLLECTOR_VERSION,
        "field_schema": FIELD_SCHEMA,
        "encoding": "little-endian-float32",
        "status": "ok",
        "field_count": 33,
        "payload_bytes": 132,
        "producer_clock": "Openplanet.Time.Now/raw",
    }
    if any(producer.get(key) != value for key, value in expected.items()):
        raise ClockEvidenceError("fresh006 producer header contract mismatch")
    for flag in (
        "simulation_phase_qualified",
        "applied_input_acknowledgement",
        "atomic_snapshot_claimed",
    ):
        if producer.get(flag) is not False:
            raise ClockEvidenceError("unqualified producer phase/ACK flags required")
    nonce = producer.get("nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 32
        or any(c not in "0123456789abcdef" for c in nonce)
    ):
        raise ClockEvidenceError("fresh006 exact connection nonce required")
    sequence = _integer(producer["sequence"], "sequence")
    if not sequence:
        raise ClockEvidenceError("sample sequence must be positive")
    begin = _integer(producer["producer_begin_Time_Now_raw"], "producer begin")
    end = _integer(producer["producer_end_Time_Now_raw"], "producer end")
    request = _integer(metadata["host_request_perf_counter_ns"], "host request")
    received = _integer(metadata["host_received_perf_counter_ns"], "host receive")
    if end < begin or received < request:
        raise ClockEvidenceError("producer/host interval regressed")
    if metadata["payload_sha256"] != hashlib.sha256(packet).hexdigest():
        raise ClockEvidenceError("fresh006 payload SHA256 mismatch")
    race = source["raw33"][3]
    if int(race) != race or race < 0:
        raise ClockEvidenceError("literal integral race milliseconds required")
    steer, gas, brake = struct.unpack_from("<3f", packet, 120)
    if not -1 <= steer <= 1 or gas not in (0, 1) or brake not in (0, 1):
        raise ClockEvidenceError("observed SGB is outside the exact native input domain")
    if not producer.get("actual_map_uid"):
        raise ClockEvidenceError("actual producer map UID required")
    return {
        "race_ms": int(race),
        "physics6_hex": packet[16:40].hex(),
        "sgb_hex": packet[120:132].hex(),
        "packet_sha256": metadata["payload_sha256"],
        "raw33": source["raw33"],
        "nonce": nonce,
        "sequence": sequence,
        "producer_begin": begin,
        "producer_end": end,
        "host_request_ns": request,
        "host_received_ns": received,
        "map_uid": producer["actual_map_uid"],
    }


def inspect_case(  # noqa: PLR0913 -- explicit qualification thresholds, never hidden defaults
    source: dict[str, Any],
    actions: list[int],
    *,
    tick_ms: int = 10,
    minimum_span_ms: int = 8000,
    maximum_span_ms: int = 10000,
    maximum_gas_ms: int = 1500,
) -> dict[str, Any]:
    if not actions or any(type(action) is not int or action not in (36, 39) for action in actions):
        raise ClockEvidenceError("this bounded plan supports only neutral36 and straight-gas39")
    if tick_ms != 10 or not 50 <= minimum_span_ms <= maximum_span_ms <= 10000:
        raise ClockEvidenceError("explicit ten-ms grid and bounded 50..10000ms span required")
    if not 0 < maximum_gas_ms <= 2000:
        raise ClockEvidenceError("explicit observed gas limit of at most2000ms required")
    rows = source["rows"]
    if len(rows) != len(actions):
        raise ClockEvidenceError("completed decision count differs from retained planned tape")
    first = source["initial_packet"]
    if _bytes(first["raw33"]) != _bytes(source["initial_raw33"]):
        raise ClockEvidenceError("initial observation does not bind its exact fresh packet")
    packets = [_packet(first)]
    prior_raw = first["raw33"]
    for index, (row, action) in enumerate(zip(rows, actions, strict=True)):
        if row["action"] != action or _bytes(row["before_raw33"]) != _bytes(prior_raw):
            raise ClockEvidenceError(
                f"decision{index} breaks planned action/observation continuity"
            )
        if struct.pack("<3f", *row["issued_gbs"]) != struct.pack("<3f", float(action == 39), 0, 0):
            raise ClockEvidenceError("issued GBS does not match the restricted straight-input plan")
        applied = _integer(row["apply_start_host_ns"], "issued control begin")
        finished = _integer(row["apply_finished_host_ns"], "issued control end")
        if finished < applied or applied < packets[-1]["host_received_ns"]:
            raise ClockEvidenceError("issued controls precede their retained observation")
        if not row.get("packets"):
            raise ClockEvidenceError(
                "each decision requires every acquired packet, including its endpoint"
            )
        packets.extend(_packet(packet) for packet in row["packets"])
        if _bytes(row["after_raw33"]) != _bytes(packets[-1]["raw33"]):
            raise ClockEvidenceError("decision endpoint does not bind the last acquired packet")
        prior_raw = row["after_raw33"]
    problems: list[str] = []
    by_time: dict[int, list[dict[str, Any]]] = {}
    previous = None
    for packet in packets:
        if packet["race_ms"] % tick_ms:
            problems.append("race-time-off-ten-ms-grid")
        if previous is not None:
            if packet["race_ms"] < previous["race_ms"]:
                problems.append("race-time-regression")
            if (
                packet["nonce"] != previous["nonce"]
                or packet["sequence"] != previous["sequence"] + 1
            ):
                problems.append("fresh-request-nonce-or-sequence-gap")
            if packet["host_request_ns"] < previous["host_received_ns"]:
                problems.append("host-acquisition-order-regression")
            if packet["producer_begin"] < previous["producer_end"]:
                problems.append("producer-clock-regression")
        previous = packet
        by_time.setdefault(packet["race_ms"], []).append(packet)
    times = sorted(by_time)
    missing = [
        tick for tick in range(times[0], times[-1] + tick_ms, tick_ms) if tick not in by_time
    ]
    ambiguous_sgb = [
        tick for tick, samples in by_time.items() if len({s["sgb_hex"] for s in samples}) != 1
    ]
    ambiguous_physics = [
        tick for tick, samples in by_time.items() if len({s["physics6_hex"] for s in samples}) != 1
    ]
    if missing:
        problems.append("missing-physical-or-input-ticks")
    if ambiguous_sgb:
        problems.append("ambiguous-observed-SGB-at-one-race-time")
    if ambiguous_physics:
        problems.append("ambiguous-physics6-at-one-race-time")
    if len({packet["map_uid"] for packet in packets}) != 1:
        problems.append("actual-map-changed")
    span = times[-1] - times[0]
    if not minimum_span_ms <= span <= maximum_span_ms:
        problems.append("observed-span-outside-explicit-bounds")
    samples = [by_time[tick][0] for tick in times]
    signature_counts = Counter(sample["physics6_hex"] for sample in samples)
    gas_ticks = [
        sample["race_ms"]
        for sample in samples
        if struct.unpack("<3f", bytes.fromhex(sample["sgb_hex"]))[1] == 1
    ]
    gas_span = max(gas_ticks) - min(gas_ticks) + tick_ms if gas_ticks else 0
    if gas_span > maximum_gas_ms:
        problems.append("observed-gas-span-exceeds-bound")
    if 39 not in actions and gas_ticks:
        problems.append("gas-observed-during-neutral-only-plan")
    if 39 in actions and not gas_ticks:
        problems.append("no-gas-observed-in-gas-pulse-plan")
    return {
        "complete_tick_evidence": not problems,
        "problems": sorted(set(problems)),
        "capture_completed": source.get("completed") is True,
        "planned_decisions": len(actions),
        "nominal_span_ms": 50 * len(actions),
        "first_race_ms": times[0],
        "last_race_ms": times[-1],
        "observed_span_ms": span,
        "acquired_packets": len(packets),
        "distinct_race_ticks": len(times),
        "duplicate_race_observations": len(packets) - len(times),
        "missing_race_ticks": missing,
        "ambiguous_sgb_ticks": ambiguous_sgb,
        "ambiguous_physics_ticks": ambiguous_physics,
        "observed_gas_span_ms": gas_span,
        "unique_physics_signatures": sum(count == 1 for count in signature_counts.values()),
        "repeated_physics_signatures": sum(count > 1 for count in signature_counts.values()),
        "constant_or_repeated_physics_proves_clock": False,
        "input_consumption_acknowledgement_qualified": False,
        "clock_alignment_qualified": False,
        "samples": samples,
    }


def inspect_windows(data: dict[str, Any], tapes: dict[str, Any], **limits: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "tmrl.strict-long-clock-windows/v5",
        "capture_passed": data.get("passed") is True,
        "complete_tick_evidence": False,
        "clock_alignment_qualified": False,
        "input_consumption_acknowledgement_qualified": False,
        "cases": {},
        "unavailable_replies_retained": len(data.get("fresh_unavailable_replies", [])),
    }
    for name, actions in tapes.items():
        try:
            result = inspect_case(data["cases"][name], actions, **limits)
            if result["capture_completed"] is not True:
                result["complete_tick_evidence"] = False
                result["problems"].append("case-capture-not-completed")
            report["cases"][name] = result
        except (ClockEvidenceError, KeyError, TypeError, ValueError, OverflowError) as error:
            report["cases"][name] = {"complete_tick_evidence": False, "error": str(error)}
    report["complete_tick_evidence"] = (
        bool(tapes)
        and report["capture_passed"]
        and all(case["complete_tick_evidence"] for case in report["cases"].values())
    )
    return report


def _native_samples(data: dict[str, Any], name: str) -> list[Sample]:
    if "cases" in data:
        case = data["cases"][name]
        rows = case["rows"]
        if not rows:
            raise ClockEvidenceError("native replay contains no measured intervals")
        initial = case["initial_comparison"]
        result = [
            Sample(
                _integer(rows[0]["evidence"]["beforeNativeT"], "native initial"),
                struct.unpack("<6f", bytes.fromhex(initial["native_physics6_hex"])),
            )
        ]
        for row in rows:
            evidence = row["evidence"]
            post = bytes.fromhex(evidence["postHex"])
            if len(post) != 256 or post[:8] != b"TMPOST03":
                raise ClockEvidenceError("literal native POST256 required")
            native_t = struct.unpack_from("<I", post, 112)[0]
            if native_t != evidence["afterNativeT"] or native_t != evidence["beforeNativeT"] + 10:
                raise ClockEvidenceError("native POST time and interval evidence disagree")
            result.append(Sample(native_t, struct.unpack_from("<6f", post, 168)))
        return result
    result = []
    for row in data["rows"]:
        native_t = _integer(row["native_t"], "native counter")
        if "postHex" in row:
            post = bytes.fromhex(row["postHex"])
            if (
                len(post) != 256
                or post[:8] != b"TMPOST03"
                or struct.unpack_from("<I", post, 112)[0] != native_t
            ):
                raise ClockEvidenceError("native raw POST/time binding differs")
            physics = struct.unpack_from("<6f", post, 168)
        else:
            raise ClockEvidenceError("long clock qualification requires raw native postHex")
        result.append(Sample(native_t, physics))
    return result


def analyze_pair(windows: dict[str, Any], native: dict[str, Any], *, case: str) -> dict[str, Any]:
    source = windows["cases"][case]
    report: dict[str, Any] = {
        "schema": "tmrl.strict-long-clock-pair/v5",
        "clock_alignment_qualified": False,
        "input_consumption_acknowledgement_qualified": False,
        "paired_training_qualified": False,
        "case": case,
    }
    if not source.get("complete_tick_evidence"):
        raise ClockEvidenceError(
            "complete Windows tick evidence required before long clock matching"
        )
    ns = _native_samples(native, case)
    if any(b.time != a.time + 10 for a, b in pairwise(ns)):
        raise ClockEvidenceError("native tick gap/repetition; every actual POST is required")
    ws = [Sample(row["race_ms"], tuple(row["raw33"][4:10])) for row in source["samples"]]
    fitted = align_clocks(ns, ws)
    slope, intercept = (
        Fraction(fitted[key]["numerator"], fitted[key]["denominator"])
        for key in ("slope", "intercept")
    )
    by_time = {sample.time: sample for sample in ns}
    comparisons = []
    for sample in ws:
        native_t = (sample.time - intercept) / slope
        if native_t.denominator != 1 or int(native_t) not in by_time:
            raise ClockEvidenceError("one Windows tick has no actual aligned native POST")
        measured = by_time[int(native_t)]
        comparisons.append(
            {
                "race_ms": sample.time,
                "native_t": int(native_t),
                "exact_physics6": sample.signature == measured.signature,
            }
        )
    anchor_span = fitted["matches"][-1]["windows_race_ms"] - fitted["matches"][0]["windows_race_ms"]
    report.update(
        alignment=fitted,
        comparisons=comparisons,
        distinct_match_span_ms=anchor_span,
        exact_matched_samples=sum(row["exact_physics6"] for row in comparisons),
        measured_windows_samples=len(ws),
    )
    report["clock_alignment_qualified"] = (
        bool(comparisons)
        and anchor_span >= 5000
        and all(row["exact_physics6"] for row in comparisons)
    )
    if not report["clock_alignment_qualified"]:
        report["reason"] = (
            "unique exact anchors must span at least5000ms and every observed tick must match"
        )
    report["scope"] = "only supplied complete map/reset/input trace; no extrapolation or input ACK"
    return report


def content_join(windows: dict[str, Any], native: dict[str, Any], *, case: str) -> dict[str, Any]:
    """Match retained observations without qualifying absent intervals or inputs."""
    source = windows["cases"][case]
    if source.get("ambiguous_physics_ticks"):
        raise ClockEvidenceError("ambiguous Windows physics at one race time forbids content join")
    ns = _native_samples(native, case)
    if not ns or any(b.time != a.time + 10 for a, b in pairwise(ns)):
        raise ClockEvidenceError("content join requires every actual native POST, without gaps")
    ws = []
    for row in source["samples"]:
        raw = _bytes(row["raw33"])
        if raw[16:40].hex() != row["physics6_hex"] or row["raw33"][3] != row["race_ms"]:
            raise ClockEvidenceError("retained Windows audit sample binding differs")
        ws.append(Sample(_integer(row["race_ms"], "Windows race time"), tuple(row["raw33"][4:10])))
    fitted = align_clocks(ns, ws)
    slope, intercept = (
        Fraction(fitted[key]["numerator"], fitted[key]["denominator"])
        for key in ("slope", "intercept")
    )
    by_time = {sample.time: sample for sample in ns}
    comparisons = []
    for sample in ws:
        counter = (sample.time - intercept) / slope
        measured = by_time.get(int(counter)) if counter.denominator == 1 else None
        comparisons.append(
            {
                "windows_race_ms": sample.time,
                "native_counter": int(counter) if counter.denominator == 1 else None,
                "actual_native_POST_present": measured is not None,
                "exact_physics6": measured is not None and sample.signature == measured.signature,
                "windows_physics6_hex": sample.signature.hex(),
                "native_physics6_hex": None if measured is None else measured.signature.hex(),
            }
        )
    span = fitted["matches"][-1]["windows_race_ms"] - fitted["matches"][0]["windows_race_ms"]
    fit_evidence = dict(fitted)
    fit_evidence.pop("relative_clock_alignment_qualified", None)
    return {
        "schema": "tmrl.partial-observation-clock-content-join/v5",
        "diagnostic_only": True,
        "observed_samples_affine_relation_verified": bool(comparisons)
        and span >= 5000
        and all(row["exact_physics6"] for row in comparisons),
        "clock_alignment_qualified": False,
        "continuous_interval_qualified": False,
        "input_waveform_qualified": False,
        "input_consumption_acknowledgement_qualified": False,
        "paired_training_qualified": False,
        "case": case,
        "windows_full_tick_evidence": source.get("complete_tick_evidence") is True,
        "windows_missing_race_ticks": source.get("missing_race_ticks", []),
        "windows_ambiguous_sgb_ticks": source.get("ambiguous_sgb_ticks", []),
        "native_actual_POST_samples": len(ns),
        "retained_windows_samples": len(ws),
        "exact_retained_matches": sum(row["exact_physics6"] for row in comparisons),
        "unique_anchor_span_ms": span,
        "fit_and_heldout_evidence": fit_evidence,
        "comparisons": comparisons,
        "unmatched_retained_windows_ticks": [
            row["windows_race_ms"] for row in comparisons if not row["exact_physics6"]
        ],
        "scope": (
            "only listed retained physical observations; "
            "absent ticks, intervals, inputs and ACK remain unqualified"
        ),
        "extrapolation_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    windows = sub.add_parser("inspect-windows")
    windows.add_argument("windows", type=Path)
    windows.add_argument("tapes", type=Path)
    windows.add_argument("output", type=Path)
    windows.add_argument("--minimum-span-ms", type=int, default=8000)
    windows.add_argument("--maximum-span-ms", type=int, default=10000)
    for name in ("analyze-pair", "content-join"):
        pair = sub.add_parser(name)
        pair.add_argument("windows_audit", type=Path)
        pair.add_argument("native", type=Path)
        pair.add_argument("output", type=Path)
        pair.add_argument("--case", required=True)
    args = parser.parse_args()
    sources = {
        key: {"path": str(value), "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}
        for key, value in vars(args).items()
        if isinstance(value, Path) and key != "output"
    }
    try:
        if args.command == "inspect-windows":
            report = inspect_windows(
                json.loads(args.windows.read_bytes()),
                json.loads(args.tapes.read_bytes()),
                minimum_span_ms=args.minimum_span_ms,
                maximum_span_ms=args.maximum_span_ms,
            )
            passed = report["complete_tick_evidence"]
        else:
            operation = content_join if args.command == "content-join" else analyze_pair
            report = operation(
                json.loads(args.windows_audit.read_bytes()),
                json.loads(args.native.read_bytes()),
                case=args.case,
            )
            passed = (
                report["observed_samples_affine_relation_verified"]
                if args.command == "content-join"
                else report["clock_alignment_qualified"]
            )
    except (ClockEvidenceError, AlignmentError, KeyError, ValueError, TypeError) as error:
        report = {
            "clock_alignment_qualified": False,
            "complete_tick_evidence": False,
            "error": str(error),
        }
        passed = False
    report["sources"] = sources
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
