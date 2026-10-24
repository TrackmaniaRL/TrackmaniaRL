"""Build a source-bound nominal native simulation clock proof without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import artifacts

SCHEMA = "tmrl.native-simulation-clock/v1"
UNIT_SCHEMA = "tmrl.native-simulation-unit-source/v1"
SCALE_VA = 0x399E90
SCALE_HEX = "6f12833a"
WINDOWS = {
    "counter_store_and_post": (
        0x1219750,
        "8b5dd041895f48488d3d8aadc200be01000000e808a97800488dbdf0feffff"
        "e8dcad900089d84401f03b45c0488b5d980f87da0900008945d0",
    ),
    "dt_caller": (
        0x121A050,
        "4c8b7db8498b7f188b75d0448b75c44489f231c94c8b8550ffffff41b9fe000000e83a51f8ff",
    ),
    "dt_convert": (
        0x119F1DB,
        "89559c895594f30f2ac2f30f5905a3ac1ffff30f118500fffffff30f1145a0",
    ),
    "dt_component_call": (0x11A150B, "4c89e78b75c4f30f108500ffffffe8425b79ff"),
}


class ClockProofError(ValueError):
    """The exact source or observed native ledger differs from this clock profile."""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _u32(raw: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", raw, offset)[0])


def elf_offset(image: bytes, address: int, length: int) -> int:
    """Map a complete file-backed ELF64 little-endian PT_LOAD range exactly once."""
    if image[:7] != b"\x7fELF\x02\x01\x01" or len(image) < 64 or length <= 0:
        raise ClockProofError("bounded ELF64 little-endian source required")
    phoff = struct.unpack_from("<Q", image, 32)[0]
    width, count = struct.unpack_from("<HH", image, 54)
    if width != 56 or phoff + width * count > len(image):
        raise ClockProofError("complete ELF64 program headers required")
    matches = []
    for index in range(count):
        kind, _, offset, virtual, _, size, _, _ = struct.unpack_from(
            "<IIQQQQQQ", image, phoff + index * width
        )
        if kind == 1 and virtual <= address and address + length <= virtual + size:
            selected = offset + address - virtual
            if selected + length <= len(image):
                matches.append(selected)
    if len(matches) != 1:
        raise ClockProofError("exactly one complete file-backed native address mapping required")
    return int(matches[0])


def check_unit_windows(windows: dict[str, bytes], scale: bytes) -> dict[str, Any]:
    if set(windows) != set(WINDOWS):
        raise ClockProofError("all native counter/dt call-chain guards required")
    for name, (_, expected) in WINDOWS.items():
        if windows[name] != bytes.fromhex(expected):
            raise ClockProofError("native clock instruction guard differs: " + name)
    if scale.hex() != SCALE_HEX:
        raise ClockProofError("pinned binary32 nominal1/1000 scale differs")
    # Decode the actual RIP-relative operand and call, rather than assuming that
    # a coincidentally nearby 0.001 literal belongs to the validated dt path.
    conversion = windows["dt_convert"]
    operand = 0x119F1ED + struct.unpack_from("<i", conversion, 14)[0]
    caller = windows["dt_caller"]
    callee = 0x121A076 + struct.unpack_from("<i", caller, 34)[0]
    if operand != SCALE_VA or callee != 0x119F1B0:
        raise ClockProofError("native dt constant/callee linkage differs")
    actual_scale = struct.unpack("<f", scale)[0]
    rounded_dt10 = struct.pack("<f", 10 * actual_scale)
    return {
        "nominal_scale": {"numerator": 1, "denominator": 1000},
        "binary32_scale_hex": scale.hex(),
        "binary32_scale_value": actual_scale,
        "dt10_binary32_product_nearest_even_hex": rounded_dt10.hex(),
        "dt10_binary32_product_nearest_even": struct.unpack("<f", rounded_dt10)[0],
        "nominal_seconds_per_tick": 0.001,
        "nominal_seconds_per_interval": 0.01,
        "rounding_note": "product illustrates round-to-nearest; does not override engine FP state",
    }


def controlled_limit(reset: int, first_p: int, count: int) -> int:
    if (reset, first_p, count) != (2200, 152, 4152):
        raise ClockProofError("exact V7 neutral reset/current-reader/tape count required")
    return reset + (count - first_p - 1) * 10


def audit_reference(report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial = report["reset"]["state"]
    if initial["nativeT"] != 2200 or initial["currentRecordIndex"] != 152:
        raise ClockProofError("actual V7 reset source must be native2200/currentp152")
    reader = bytes.fromhex(initial["readerHex"])
    count = struct.unpack_from("<I", reader, 116)[0]
    maximum = controlled_limit(initial["nativeT"], initial["currentRecordIndex"], count)
    previous, seen = initial["nativeT"], 0
    owner = None
    for index, row in enumerate(rows):
        result = row["result"]
        events = result["authorization"]["nativeEventHex"]
        if row["decision"] != index or len(events) != result["actualIterations"]:
            raise ClockProofError("complete consecutive actual reference intervals required")
        for event in events:
            raw = bytes.fromhex(event)
            if len(raw) != 256 or raw[:8] != b"TMPOST03":
                raise ClockProofError("actual native POST256 required")
            if (
                (_u32(raw, 8), _u32(raw, 12), _u32(raw, 148), _u32(raw, 212)) != (256, 3, 75, 10)
                or _u32(raw, 16) != _u32(raw, 20)
                or (owner is not None and _u32(raw, 16) != owner)
                or not _u32(raw, 112) == _u32(raw, 116) == _u32(raw, 144) == previous + 10
                or _u32(raw, 136) != 10
                or _u32(raw, 140) != 10
                or not all(math.isfinite(value) for value in struct.unpack_from("<10f", raw, 152))
            ):
                raise ClockProofError("actual own POST counter/frame_dt/grid inconsistency")
            owner, previous, seen = _u32(raw, 16), _u32(raw, 112), seen + 1
        if result["observation"]["nativeT"] != previous:
            raise ClockProofError("reference row endpoint differs from actual final POST")
    last = report["last_result"]
    if (
        not rows
        or seen != report["actual_intervals"]
        or len(rows) != report["rows"]
        or previous != last["observation"]["nativeT"]
        or last["terminated"] is not True
        or last["truncated"] is not False
        or last["outcome"]["qualifiedBy"] != "env002.full-original-validator+TMRESULT"
    ):
        raise ClockProofError("complete retained native finish reference required")
    return {
        "actual_post_intervals": seen,
        "observed_minimum_native": initial["nativeT"],
        "observed_maximum_native": previous,
        "controlled_maximum_from_reader_guard": maximum,
        "actual_computed_result": last["outcome"]["computedResult"],
        "validation_acceptance": last["outcome"]["validationAcceptance"],
        "unit_proof_is_binary_source_not_extrapolated_observation": True,
    }


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path.read_bytes())}


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output
    if output.exists():
        raise ClockProofError("fresh proof output required")
    config = json.loads(args.config.read_bytes())
    assets_path = Path(config["asset_manifest"])
    assets = json.loads(assets_path.read_bytes())
    if (
        config["protocol_profile"] != "immediate-current/v7"
        or config["reset_profile"] != "neutral-race-start/v7"
        or assets["profile"] != "native-immediate-kinematics6/v7"
    ):
        raise ClockProofError("explicit pinned V7 config/assets required")
    runtime = Path(config["runtime_root"])
    paths = {role: runtime / relative for role, relative in assets["roles"].items()}
    paths["reference"] = args.reference
    for role in ("server", "tape", "map", "reference"):
        if digest(paths[role].read_bytes()) != artifacts.PINS[artifacts.ROLE_PATHS[role]]:
            raise ClockProofError("pinned original native asset differs: " + role)
    manifest_path = paths["shim_source_manifest"]
    if (
        digest(manifest_path.read_bytes())
        != "92cc6ebdfb81a9c32ca345dbe884065ecf523ec3cf70fb37a8b6e12eff20b325"
    ):
        raise ClockProofError("original shim source manifest differs")
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(
        row for row in manifest["files"] if row["path"] == "search/forkshim/src/post_observe.rs"
    )
    shim_source = args.shim_post_source.read_bytes()
    if (
        digest(shim_source) != entry["sha256"]
        or b"dt!=10||frame_dt!=dt||current!=completed||simclock!=completed" not in shim_source
    ):
        raise ClockProofError("pinned source POST dt/counter invariants differ")
    image = paths["server"].read_bytes()
    spans, observed = {}, {}
    for name, (address, expected) in WINDOWS.items():
        size = len(bytes.fromhex(expected))
        offset = elf_offset(image, address, size)
        observed[name] = image[offset : offset + size]
        spans[name] = {
            "virtual_address": hex(address),
            "file_offset": hex(offset),
            "hex": observed[name].hex(),
        }
    scale_offset = elf_offset(image, SCALE_VA, 4)
    scale = check_unit_windows(observed, image[scale_offset : scale_offset + 4])
    report_path, rows_path = args.finish_run / "report.json", args.finish_run / "steps.jsonl"
    report = json.loads(report_path.read_bytes())
    rows = [json.loads(line) for line in rows_path.read_bytes().splitlines()]
    audit = audit_reference(report, rows)
    sources = {role: _source(path) for role, path in paths.items()}
    sources.update(
        shim_post_source=_source(args.shim_post_source),
        finish_reference=_source(report_path),
        finish_intervals=_source(rows_path),
        runtime_config=_source(args.config),
        asset_manifest=_source(assets_path),
    )
    scope = {
        "native_profile": "native-immediate-kinematics6/v7",
        "reset_profile": "neutral-race-start/v7",
        "input_schedule": "current-record-before-load/v1",
        **{
            role + "_sha256": sources[role]["sha256"]
            for role in ("map", "reference", "server", "tape")
        },
    }
    output.mkdir(parents=True)
    generator_snapshot = output / "generator-source.py"
    generator_snapshot.write_bytes(Path(__file__).read_bytes())
    sources["generator"] = _source(generator_snapshot)
    unit = {
        "schema": UNIT_SCHEMA,
        "semantics": "nominal-engine-simulation-seconds; no wall/race origin mapping",
        "sources": sources,
        "scope": scope,
        "instruction_spans": spans,
        "scale_constant": {
            "virtual_address": hex(SCALE_VA),
            "file_offset": hex(scale_offset),
            **scale,
        },
        "counter_source": "actual POST.nativeT = owned sim+0x48 = completed counter",
        "post_source_invariant": "dt==frame_dt==10 and simclock==completed==current",
        "observed_native_reference": audit,
        "windows_evidence_used": False,
        "race_origin_mapping": None,
        "engine_executed": False,
    }
    unit_path = output / "unit-proof.json"
    unit_path.write_text(json.dumps(unit, indent=2) + "\n")
    artifact = {
        "schema": SCHEMA,
        "native_clock_qualified": True,
        "counter_source": "actual POST.nativeT",
        "raw_interval": 10,
        "seconds_per_tick": 0.001,
        "minimum_native": 2200,
        "maximum_native": audit["controlled_maximum_from_reader_guard"],
        "scope": scope,
        "sources": {**sources, "unit_proof": _source(unit_path)},
        "unit_semantics": unit["semantics"],
        "windows_clock_mapping": None,
        "paired_training_qualified": False,
    }
    (output / "clock.json").write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "shim-post-source", "finish-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = build(parser.parse_args())
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "scope": result["scope"],
                "maximum_native": result["maximum_native"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
