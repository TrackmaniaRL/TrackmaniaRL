"""Bounded static PE string/RIP-reference inventory; never opens a live process."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any


class PE:
    def __init__(self, path: Path) -> None:
        self.path, self.raw = path, path.read_bytes()
        raw = self.raw
        pe = struct.unpack_from("<I", raw, 0x3C)[0]
        if raw[pe : pe + 4] != b"PE\0\0":
            raise ValueError("PE header required")
        machine, count, _, _, _, optional_size, _ = struct.unpack_from("<HHIIIHH", raw, pe + 4)
        optional = pe + 24
        if machine != 0x8664 or struct.unpack_from("<H", raw, optional)[0] != 0x20B:
            raise ValueError("x64 PE32+ required")
        self.base = struct.unpack_from("<Q", raw, optional + 24)[0]
        self.entry = struct.unpack_from("<I", raw, optional + 16)[0]
        self.sections: list[dict[str, Any]] = []
        for index in range(count):
            at = optional + optional_size + index * 40
            name, virtual_size, rva, size, offset = struct.unpack_from("<8sIIII", raw, at)
            flags = struct.unpack_from("<I", raw, at + 36)[0]
            self.sections.append(
                {
                    "name": name.rstrip(b"\0").decode("ascii"),
                    "rva": rva,
                    "virtual_size": virtual_size,
                    "file_offset": offset,
                    "file_size": size,
                    "flags": flags,
                }
            )
        exception_rva, exception_size = struct.unpack_from("<II", raw, optional + 112 + 3 * 8)
        exception_offset = self.offset(exception_rva)
        self.functions = sorted(
            struct.unpack_from("<III", raw, exception_offset + i)
            for i in range(0, exception_size, 12)
        )
        self.starts = [item[0] for item in self.functions]

    def offset(self, rva: int) -> int:
        for section in self.sections:
            delta = rva - int(section["rva"])
            if 0 <= delta < section["file_size"]:
                return int(section["file_offset"]) + delta
        raise ValueError(f"RVA not backed by source bytes: {rva:#x}")

    def address(self, offset: int) -> int:
        for section in self.sections:
            delta = offset - int(section["file_offset"])
            if 0 <= delta < section["file_size"]:
                return int(section["rva"]) + delta
        raise ValueError("file offset outside sections")

    def function(self, rva: int) -> dict[str, Any] | None:
        index = bisect.bisect_right(self.starts, rva) - 1
        if index < 0:
            return None
        start, end, unwind = self.functions[index]
        if not start <= rva < end:
            return None
        at = self.offset(start)
        return {
            "rva": hex(start),
            "end_rva": hex(end),
            "unwind_rva": hex(unwind),
            "file_offset": hex(at),
            "entry_hex": self.raw[at : at + min(end - start, 32)].hex(),
        }


def inventory(image: PE, needles: list[str]) -> dict[str, Any]:
    strings = []
    by_rva: dict[int, list[dict[str, Any]]] = {}
    for text in needles:
        for encoding in ("ascii", "utf-16le"):
            literal = text.encode(encoding)
            offset = 0
            for _ in range(100):
                offset = image.raw.find(literal, offset)
                if offset < 0:
                    break
                rva = image.address(offset)
                row: dict[str, Any] = {
                    "text": text,
                    "encoding": encoding,
                    "rva": hex(rva),
                    "file_offset": hex(offset),
                    "rip_references": [],
                    "pointer_references": [],
                }
                strings.append(row)
                by_rva.setdefault(rva, []).append(row)
                offset += len(literal)
    for section in image.sections:
        if not section["flags"] & 0x20000000:
            continue
        raw = image.raw[section["file_offset"] : section["file_offset"] + section["file_size"]]
        pattern = rb"[\x40-\x4f][\x8b\x8d][\x05\x0d\x15\x1d\x25\x2d\x35\x3d]...."
        for match in re.finditer(pattern, raw, re.DOTALL):
            at = section["rva"] + match.start()
            target = at + 7 + struct.unpack_from("<i", match.group(), 3)[0]
            for row in by_rva.get(target, []):
                row["rip_references"].append(
                    {
                        "rva": hex(at),
                        "instruction_hex": match.group().hex(),
                        "pdata_function": image.function(at),
                    }
                )
    for rva, rows in by_rva.items():
        pointer = struct.pack("<Q", image.base + rva)
        offset = 0
        for _ in range(100):
            offset = image.raw.find(pointer, offset)
            if offset < 0:
                break
            for row in rows:
                row["pointer_references"].append(
                    {"rva": hex(image.address(offset)), "file_offset": hex(offset)}
                )
            offset += 8
    return {
        "schema": "tmrl.windows-static-native-inventory/v1",
        "path": str(image.path),
        "sha256": hashlib.sha256(image.raw).hexdigest(),
        "bytes": len(image.raw),
        "image_base": hex(image.base),
        "entry_rva": hex(image.entry),
        "sections": image.sections,
        "pdata_functions": len(image.functions),
        "strings": strings,
        "executed": False,
        "live_process_accessed": False,
        "limits": (
            "At most100 matches per string/encoding and pointer. RIP matches are byte-pattern "
            "candidates pending instruction-boundary verification; not exhaustive xrefs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--needle", action="append")
    args = parser.parse_args()
    report = inventory(
        PE(args.binary),
        args.needle
        or [
            "CInputPort",
            "CInputPortDx8",
            "CInputDevice",
            "ReadHardwareCurState",
            "CInputReplay",
            "SimulationTime",
            "InputSteer",
            "InputGasPedal",
            "InputBrakePedal",
            "CSceneVehicle",
            "CGameCtnPlayground",
            "CInputScriptPad",
        ],
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "image_base": report["image_base"],
                "pdata_functions": report["pdata_functions"],
                "strings": [
                    {
                        "text": row["text"],
                        "rva": row["rva"],
                        "rip": row["rip_references"],
                        "pointers": row["pointer_references"],
                    }
                    for row in report["strings"]
                    if row["rip_references"] or row["pointer_references"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
