"""Exact content join of actual physical-copy words and independently read API data."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def analyze(physical: dict[str, Any], stream: dict[str, Any]) -> dict[str, Any]:
    if (
        physical.get("schema") != "tmrl.headless-physical-copy-probe/v5"
        or physical.get("original_64_bytes_restored") is not True
        or physical.get("destination_samples_are_PRE_copy") is not True
    ):
        raise ValueError("actual restored PRE-copy probe report required")
    observations = defaultdict(set)
    for packet in stream["packets"]:
        raw = packet["raw33"]
        observations[struct.pack("<6f", *raw[4:10])].add(raw[3])
    rows = []
    for record in physical["records"]:
        signature = struct.pack("<6I", *record["raw_body_bits_u32"][4:10])
        clocks = sorted(observations.get(signature, ()))
        rows.append(
            {
                "sequence": record["sequence"],
                "Time_Now_raw": record["Time_Now_raw"],
                "physical6_hex": signature.hex(),
                "exact_matching_API_race_times_ms": clocks,
                "raw_equals_PRE_cache": record["raw_body_bits_u32"]
                == record["destination_PRE_copy_bits_u32"],
            }
        )
    return {
        "schema": "tmrl.physical-api-content-join/v1",
        "records": len(rows),
        "unique_API_clock_matches": sum(
            len(row["exact_matching_API_race_times_ms"]) == 1 for row in rows
        ),
        "ambiguous_API_clock_matches": sum(
            len(row["exact_matching_API_race_times_ms"]) > 1 for row in rows
        ),
        "missing_API_matches": sum(not row["exact_matching_API_race_times_ms"] for row in rows),
        "raw_equals_PRE_cache_count": sum(row["raw_equals_PRE_cache"] for row in rows),
        "atomic_or_same_call_API_read_claimed": False,
        "simulation_clock_or_applied_input_ACK_claimed": False,
        "limitations": (
            "Exact content join only; no timestamp interpolation or missing-sample substitution."
        ),
        "rows": rows,
    }


def main() -> None:
    physical_path, stream_path, output = map(Path, sys.argv[1:4])
    report = analyze(json.loads(physical_path.read_bytes()), json.loads(stream_path.read_bytes()))
    report["sources"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (physical_path, stream_path)
    ]
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
    print(
        json.dumps({key: value for key, value in report.items() if key not in ("rows", "sources")})
    )


if __name__ == "__main__":
    main()
