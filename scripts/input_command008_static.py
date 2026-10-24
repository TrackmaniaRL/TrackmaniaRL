"""Bounded offline evidence for the proposed 008 PRE-consumer write point.

Reads a pinned private PE and invokes objdump only. Never opens a process,
installs a plugin, writes game memory, or qualifies an applied-input ACK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.headless_windows_native_inventory import PE

EXPECTED_SHA256 = "3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda"
PRE_RVA = 0x841553
PRE_HEX = (
    "48898424c8000000e8d0cef7ff448bbb800300004c8da3f01c0000f30f103552de4d01"
    "33ed450f57c00f57ff8bf54585ff0f84b5000000440f29542440488dbb"
)
EARLIER_RVA = 0x851F96
EARLIER_HEX = (
    "488945504c8965c0e8bdbefefff30f10879c00000033c0f30f103dc7cf4c01498bcc0f"
    "2fc7f30f1087980000000f97c04533f60f2fc7894590f20f104628f20f"
)
# Explicit bounded windows: PE .pdata entries can describe only prologue/chained
# unwind fragments, so their first end address is not a complete function bound.
WINDOWS = {
    "consumer": (0x841500, 0x841787),
    "first_callee_transform": (0x7BE430, 0x7BE500),
    "wheel_first": (0x843190, 0x844130),
    "wheel_second": (0x844130, 0x8447B0),
    "gas_brake_processing": (0x8447B0, 0x845120),
    "mode_zero": (0x83DB50, 0x83DB72),
    "mode_one": (0x83DB80, 0x83DBA2),
    "mode_zero_steer": (0x83DBD0, 0x83DBEB),
    "clamp_bit4": (0x83DBF0, 0x83DC0B),
    "clamp_gas_override_condition": (0x83DC30, 0x83DC60),
    "clamp_gas_override_value": (0x83DD90, 0x83DE10),
    "separate_headtrack_false_positive": (0x7EB2C0, 0x7EB710),
    "headtrack_class_identity": (0x7EACE0, 0x7EAD50),
    "external_input_clamp": (0x83DE60, 0x83DF0C),
    "earlier_primary_caller": (0x851F00, 0x852115),
}


def disassemble(image: PE, start: int, end: int) -> str:
    if not 0 < end - start <= 16384:
        raise ValueError("one bounded window of at most16KiB required")
    result = subprocess.run(
        [
            "objdump",
            "-d",
            "-w",
            "-M",
            "intel",
            f"--start-address={image.base + start:#x}",
            f"--stop-address={image.base + end:#x}",
            str(image.path),
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    if len(result.stdout) > 500_000:
        raise ValueError("unexpected disassembly output growth")
    return result.stdout


def decoded_rows(text: str, image_base: int) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([0-9a-f]+):\s*([0-9a-f ]+)\t(.+)", line)
        if match:
            rows.append(
                {
                    "rva": hex(int(match[1], 16) - image_base),
                    "hex": bytes.fromhex(match[2]).hex(),
                    "instruction": match[3].strip(),
                }
            )
    if not rows:
        raise ValueError("objdump produced no decoded instructions")
    return rows


def pinned_prefixes(image: PE) -> tuple[str, str, str]:
    """Validate source identity before offering any decoded instruction evidence."""
    digest = hashlib.sha256(image.raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("pinned private executable identity mismatch")
    prefix = image.raw[image.offset(PRE_RVA) : image.offset(PRE_RVA) + 64].hex()
    if prefix != PRE_HEX:
        raise ValueError("exact64 PRE prefix mismatch")
    earlier_prefix = image.raw[image.offset(EARLIER_RVA) : image.offset(EARLIER_RVA) + 64].hex()
    if earlier_prefix != EARLIER_HEX:
        raise ValueError("exact64 earlier caller prefix mismatch")
    return digest, prefix, earlier_prefix


def run(binary: Path, output: Path) -> dict[str, Any]:
    if (
        sys.platform != "linux"
        or not output.is_absolute()
        or output.resolve() != output
        or str(output).startswith("/mnt/")
        or any(parent.is_symlink() for parent in output.parents)
    ):
        raise ValueError("canonical fresh Linux/ext4 output required")
    image = PE(binary)
    digest, prefix, earlier_prefix = pinned_prefixes(image)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "tmrl.input-command008-static-evidence/v1",
        "binary": str(binary),
        "binary_sha256": digest,
        "binary_bytes": len(image.raw),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "objdump_version": subprocess.run(
            ["objdump", "--version"], capture_output=True, check=True, text=True, timeout=5
        ).stdout.splitlines()[0],
        "pre_rva": hex(PRE_RVA),
        "pre_file_offset": hex(image.offset(PRE_RVA)),
        "pre_prefix64_hex": prefix,
        "stolen_instruction8_hex": prefix[:16],
        "padding": 3,
        "earlier_caller_candidate": {
            "rva": hex(EARLIER_RVA),
            "file_offset": hex(image.offset(EARLIER_RVA)),
            "prefix64_hex": earlier_prefix,
            "stolen8_hex": earlier_prefix[:16],
            "owner_register": "rdi",
            "step_context_register": "rsi",
            "step_argument_offset": "0x14",
            "whole_step_pre_qualified": False,
        },
        "image_executed": False,
        "live_process_accessed": False,
        "game_code_or_data_written": False,
        "applied_input_ack_qualified": False,
        "interprocedural_no_overwrite_proven": False,
        "thread_safety_qualified": False,
        "functions": {},
    }
    for name, (start, end) in WINDOWS.items():
        text = disassemble(image, start, end)
        filename = name + ".txt"
        (output / filename).write_text(text, encoding="utf-8")
        decoded = decoded_rows(text, image.base)
        if name == "consumer":
            target = next(row for row in decoded if row["rva"] == hex(PRE_RVA))
            if target["hex"] != prefix[:16]:
                raise ValueError("PRE is not the full expected single8-byte instruction")
        report["functions"][name] = {
            "window_start_rva": hex(start),
            "window_end_rva": hex(end),
            "pdata_fragment_at_start": image.function(start),
            "disassembly": filename,
            "disassembly_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "decoded_instruction_count": len(decoded),
            "instructions": decoded,
            "input_displacement_candidates": [
                row for row in decoded if re.search(r"\+0x(?:98|9c|a0)\]", row["instruction"])
            ],
        }
    report["scope"] = (
        "Fifteen explicit bounded disassembly windows only. "
        "PE pdata may describe unwind fragments. "
        "Displacement candidates do not prove base identity "
        "or instruction execution; manually trace aliases/branches. No complete call-graph, "
        "writer-order, global clock, callback quiescence or live write result is asserted."
    )
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.binary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pre_rva": report["pre_rva"],
                "prefix64": report["pre_prefix64_hex"],
                "functions": len(report["functions"]),
                "executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
