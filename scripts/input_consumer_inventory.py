"""Bounded displacement candidates in an offline PE; never accesses a process.

Matches are raw bytes, not decoded instructions. Validate each instruction and
base-register identity before treating a candidate as a participant field access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from scripts.headless_windows_native_inventory import PE


def candidates(  # noqa: PLR0913 - independent bounded static-scan controls
    image: PE, displacement: int, *, limit: int, start_rva: int = 0, end_rva: int = 0x100000000
) -> dict[str, Any]:
    if not 0 <= displacement <= 0x7FFFFFFF or not 1 <= limit <= 256:
        raise ValueError("bounded positive displacement and limit required")
    if not 0 <= start_rva < end_rva <= 0x100000000:
        raise ValueError("valid finite RVA interval required")
    needle = struct.pack("<I", displacement)
    rows: list[dict[str, Any]] = []
    total = 0
    for section in image.sections:
        if not section["flags"] & 0x20000000:
            continue
        begin = max(0, start_rva - section["rva"])
        end = min(section["file_size"], end_rva - section["rva"])
        if begin >= end:
            continue
        raw = image.raw[section["file_offset"] + begin : section["file_offset"] + end]
        offset = raw.find(needle)
        while offset >= 0:
            total += 1
            if len(rows) < limit:
                rva = section["rva"] + begin + offset
                start = max(0, offset - 8)
                rows.append(
                    {
                        "displacement_byte_rva": hex(rva),
                        "window_rva": hex(section["rva"] + begin + start),
                        "window_hex": raw[start : offset + 12].hex(),
                        "pdata_function": image.function(rva),
                    }
                )
            offset = raw.find(needle, offset + 1)
    return {
        "displacement": hex(displacement),
        "start_rva": hex(start_rva),
        "end_rva": hex(end_rva),
        "raw_byte_matches": total,
        "retained_candidates": rows,
        "truncated": total > limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--displacement", type=lambda value: int(value, 0), action="append")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--start-rva", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--end-rva", type=lambda value: int(value, 0), default=0x100000000)
    args = parser.parse_args()
    image = PE(args.binary)
    report = {
        "schema": "tmrl.windows-input-consumer-candidates/v1",
        "sha256": hashlib.sha256(image.raw).hexdigest(),
        "binary_bytes": len(image.raw),
        "image_base": hex(image.base),
        "executed": False,
        "live_process_accessed": False,
        "instruction_boundaries_qualified": False,
        "participant_base_identity_qualified": False,
        "candidates": [
            candidates(
                image, value, limit=args.limit, start_rva=args.start_rva, end_rva=args.end_rva
            )
            for value in args.displacement or (0xEEC, 0xEF0, 0xEF4)
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": report["sha256"]}))


if __name__ == "__main__":
    main()
