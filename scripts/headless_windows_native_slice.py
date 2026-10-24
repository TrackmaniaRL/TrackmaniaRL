"""Read bounded PE byte slices and pointer/xref candidates from an offline image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from typing import Any

from headless_windows_native_inventory import PE


def slice_image(  # noqa: PLR0913 - independent bounded CLI slice controls
    image: PE, rva: int, *, size: int, references: bool
) -> dict[str, Any]:
    if not 0 < size <= 16384:
        raise ValueError("slice size must be in 1..16384")
    offset = image.offset(rva)
    data = image.raw[offset : offset + size]
    words = []
    for delta in range(0, len(data) - 7, 8):
        value = struct.unpack_from("<Q", data, delta)[0]
        row: dict[str, Any] = {"rva": hex(rva + delta), "u64": hex(value)}
        try:
            target = value - image.base
            at = image.offset(target)
            row["target_rva"] = hex(target)
            row["function"] = image.function(target)
            row["ascii_candidate"] = image.raw[at : at + 100].split(b"\0", 1)[0].decode("ascii")
        except (ValueError, UnicodeDecodeError):
            pass
        words.append(row)
    result = {"rva": hex(rva), "file_offset": hex(offset), "hex": data.hex(), "words": words}
    if references:
        refs: list[dict[str, Any]] = []
        pointer = struct.pack("<Q", image.base + rva)
        at = image.raw.find(pointer)
        while at >= 0 and len(refs) < 100:
            refs.append({"kind": "pointer", "rva": hex(image.address(at))})
            at = image.raw.find(pointer, at + 1)
        for section in image.sections:
            if not section["flags"] & 0x20000000:
                continue
            raw = image.raw[section["file_offset"] : section["file_offset"] + section["file_size"]]
            for match in re.finditer(
                rb"[\x40-\x4f][\x8b\x8d][\x05\x0d\x15\x1d\x25\x2d\x35\x3d]....", raw, re.DOTALL
            ):
                address = section["rva"] + match.start()
                if address + 7 + struct.unpack_from("<i", match.group(), 3)[0] == rva:
                    refs.append(
                        {
                            "kind": "rip-candidate",
                            "rva": hex(address),
                            "hex": match.group().hex(),
                            "function": image.function(address),
                        }
                    )
            for match in re.finditer(rb"[\xe8\xe9]....", raw, re.DOTALL):
                address = section["rva"] + match.start()
                if address + 5 + struct.unpack_from("<i", match.group(), 1)[0] == rva:
                    refs.append(
                        {
                            "kind": "direct-branch-candidate",
                            "rva": hex(address),
                            "hex": match.group().hex(),
                            "function": image.function(address),
                        }
                    )
        result["references"] = refs
    return result


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rva", action="append", type=lambda text: int(text, 0), required=True)
    parser.add_argument("--size", type=lambda text: int(text, 0), default=128)
    parser.add_argument("--references", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = PE(args.binary)
    report = {
        "schema": "tmrl.windows-static-native-slices/v1",
        "binary": str(args.binary),
        "sha256": hashlib.sha256(image.raw).hexdigest(),
        "bytes": len(image.raw),
        "executed": False,
        "live_process_accessed": False,
        "limits": "RIP references are byte candidates; inspect instruction boundaries separately.",
        "slices": [
            slice_image(image, rva, size=args.size, references=args.references) for rva in args.rva
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
