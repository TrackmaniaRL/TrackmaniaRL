"""Bounded V7 replay of literal 007 consumer controls under three explicit phases.

This experiment does not qualify Windows issued-command ACK, race-clock mapping
or paired training. All candidate phases are retained without selecting a winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

from scripts.headless_provision_immediate import provision
from scripts.headless_replay_windows_tapes import ObservedSession
from scripts.input_consumer_probe import verify_capture, verify_restoration


def word_bytes(words: list[int]) -> bytes:
    if any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in words):
        raise ValueError("literal uint32 words required")
    return struct.pack("<" + "I" * len(words), *words)


def load_trace(source: dict[str, Any]) -> list[dict[str, Any]]:
    verify_restoration(source, source["nonce"])
    verify_capture(source)
    rows = source["records"]
    if len(rows) < 3:
        raise ValueError("at least three actual consumer observations required")
    for index, row in enumerate(rows):
        if (
            index
            and row["R13D_step_argument_raw"] - rows[index - 1]["R13D_step_argument_raw"] != 10
        ):
            raise ValueError("007 raw argument sequence must have exactly ten-unit increments")
        gas, brake, steer, filtered = struct.unpack(
            "<4f", word_bytes(row["actual_control_bits_u32"])
        )
        physical = struct.unpack("<10f", word_bytes(row["raw_body_bits_u32"]))
        if (
            not all(math.isfinite(value) for value in (*physical, gas, brake, steer, filtered))
            or gas not in (0, 1)
            or brake not in (0, 1)
            or not -1 <= steer <= 1
            or not -1.001 <= filtered <= 1.001
        ):
            raise ValueError("actual 007 fields outside the finite native input domain")
    return list(rows)


def neutral_posts(source: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        source.get("schema") != "tmrl.neutral-native-clock-posts/v1"
        or source.get("completed") is not True
    ):
        raise ValueError("completed actual neutral POST capture required")
    rows = source["rows"]
    if not 2 <= len(rows) <= 1000:
        raise ValueError("bounded retained neutral sequence required")
    for index, row in enumerate(rows):
        raw = bytes.fromhex(row["postHex"])
        if (
            len(raw) != 256
            or raw[:8] != b"TMPOST03"
            or struct.unpack_from("<I", raw, 112)[0] != row["native_t"]
            or struct.unpack_from("<I", raw, 148)[0] != 75
            or raw[88:100] != bytes(12)
            or (index and row["native_t"] != rows[index - 1]["native_t"] + 10)
        ):
            raise ValueError("consecutive literal neutral POSTs required")
    return list(rows)


def anchor_index(rows: list[dict[str, Any]], neutral: list[dict[str, Any]]) -> int:
    wanted = word_bytes(rows[0]["raw_body_bits_u32"][4:10])
    matches = [
        index
        for index, row in enumerate(neutral)
        if bytes.fromhex(row["postHex"])[168:192] == wanted
    ]
    if len(matches) != 1:
        raise ValueError("first actual physics6 must have exactly one neutral content match")
    return matches[0]


def phase_plan(name: str, count: int, anchor: int) -> dict[str, Any]:
    """Predeclared mappings; no outcome-dependent selection or synthetic prefix."""
    if name == "A-current-to-next":
        return {
            "start_native_t": anchor,
            "start_pose_index": 0,
            "pairs": [(i, i + 1) for i in range(count - 1)],
        }
    if name == "B-previous-to-next":
        return {
            "start_native_t": anchor + 10,
            "start_pose_index": 1,
            "pairs": [(i - 1, i + 1) for i in range(1, count - 1)],
        }
    if name == "C-current-to-same":
        return {
            "start_native_t": anchor - 10,
            "start_pose_index": None,
            "pairs": [(i, i) for i in range(count)],
        }
    raise ValueError("explicit A/B/C phase required")


def compare(post: bytes, target: dict[str, Any]) -> dict[str, Any]:
    wanted = word_bytes(target["raw_body_bits_u32"])
    actual = post[152:192]
    delta = [
        a - b
        for a, b in zip(struct.unpack("<10f", actual), struct.unpack("<10f", wanted), strict=True)
    ]
    return {
        "physics6_exact": actual[16:] == wanted[16:],
        "physical10_exact": actual == wanted,
        "native_physics6_hex": actual[16:].hex(),
        "windows_physics6_hex": wanted[16:].hex(),
        "position_difference": delta[4:7],
        "velocity_difference": delta[7:10],
        "Y_difference": delta[5],
        "VY_difference": delta[8],
    }


def close_case(session: ObservedSession, case: dict[str, Any]) -> None:
    """Retain cleanup diagnostics even when the close operation itself fails."""
    try:
        session.close(error=not case["completed"])
    finally:
        case["cleanup"] = session.cleanup
    if not isinstance(session.cleanup, dict) or session.cleanup.get("remainingOwned") != []:
        raise RuntimeError("007 native phase experiment lacks zero-owned cleanup proof")


def run(args: argparse.Namespace) -> None:
    output = args.output
    if (
        sys.platform != "linux"
        or not output.is_absolute()
        or str(output).startswith("/mnt/")
        or output.resolve() != output
    ):
        raise ValueError("canonical fresh Linux/ext4 output required")
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("output parents must not be symlinks")
    output.mkdir(parents=True, exist_ok=False)
    paths = (args.trace, args.neutral_report, args.v6_config, args.receipt)
    report: dict[str, Any] = {
        "schema": "tmrl.input-consumer-native-phase-replay/v1",
        "completed": False,
        "issued_command_ack_claimed": False,
        "windows_race_mapping_claimed": False,
        "paired_training_qualified": False,
        "winner_selected": False,
        "production_pins_modified": False,
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cases": {},
    }
    begin = time.monotonic()
    try:
        rows = load_trace(json.loads(args.trace.read_bytes()))
        neutral = neutral_posts(json.loads(args.neutral_report.read_bytes()))
        index = anchor_index(rows, neutral)
        anchor = neutral[index]["native_t"]
        report["anchor"] = {
            "method": "unique exact initial physical position+velocity float32 bits",
            "native_t": anchor,
            "neutral_index": index,
            "windows_R13_raw": rows[0]["R13D_step_argument_raw"],
            "window_race_used": False,
        }
        report["source_rows"] = len(rows)
        receipt = json.loads(args.receipt.read_bytes())
        deployment_dir = output / "deployment"
        report["provision"] = provision(
            argparse.Namespace(
                destination=deployment_dir,
                v6_config=args.v6_config,
                source_evidence_dir=Path(__file__).parents[1]
                / "trackmaniarl/trackmania/native_headless/_runtime/native_input_v7",
                shim=Path(receipt["artifact"]),
                duration_limit=39950,
                max_trace_bytes=64 * 1024 * 1024,
            )
        )
        config = json.loads((deployment_dir / "config.json").read_bytes())
        runtime = Path(config["runtime_root"])
        assets = json.loads(Path(config["asset_manifest"]).read_bytes())
        relocated_receipt = {**receipt, "artifact": str(runtime / assets["roles"]["shim"])}
        driver = json.loads(Path(config["deployment_manifest"]).read_bytes())[
            "driverBuildApproval"
        ]["driverSha256"]
        for name in ("A-current-to-next", "B-previous-to-next", "C-current-to-same"):
            plan = phase_plan(name, len(rows), anchor)
            case: dict[str, Any] = {
                "mapping": plan,
                "rows": [],
                "completed": False,
                "administrative_neutral_seek": [],
            }
            report["cases"][name] = case
            start = plan["start_native_t"]
            retained = next((row for row in neutral if row["native_t"] == start), None)
            if retained is None or start < 2190:
                case["unsupported"] = "required start neutral POST was not actually retained"
                continue
            start_index = plan["start_pose_index"]
            if start_index is not None and bytes.fromhex(retained["postHex"])[
                168:192
            ] != word_bytes(rows[start_index]["raw_body_bits_u32"][4:10]):
                case["unsupported"] = (
                    "required starting pose does not match the retained neutral POST"
                )
                continue
            session = None
            try:
                session = ObservedSession(
                    runtime, relocated_receipt, output / (output.name + "-" + name), 2190, driver
                )
                session.reset()
                remaining = (start - 2190) // 10
                while remaining:
                    if (output / "STOP").exists() or time.monotonic() - begin > 170:
                        raise InterruptedError("007 native phase experiment STOP/deadline")
                    n = min(remaining, 5)
                    case["administrative_neutral_seek"].append(session.step(36, n))
                    remaining -= n
                if session.boundary[152:192] != bytes.fromhex(retained["postHex"])[152:192]:
                    raise ValueError("fresh native seek differs from actual retained neutral state")
                case["initial_post_hex"] = session.boundary.hex()
                for source_index, target_index in plan["pairs"]:
                    if (output / "STOP").exists() or time.monotonic() - begin > 170:
                        raise InterruptedError("007 native phase experiment STOP/deadline")
                    source = rows[source_index]
                    bits = source["actual_control_bits_u32"]
                    waveform = {
                        "sgbHex": word_bytes([bits[2], bits[0], bits[1]]).hex(),
                        "source_row": source_index,
                        "source_R13_raw": source["R13D_step_argument_raw"],
                        "source_kind": "007 actual physical consumer state",
                        "phase_hypothesis": name,
                    }
                    evidence = session.step_observed(waveform)
                    post = bytes.fromhex(evidence["postHex"])
                    case["rows"].append(
                        {
                            "source_row": source_index,
                            "target_row": target_index,
                            "native_t": evidence["afterNativeT"],
                            "target_R13_raw": rows[target_index]["R13D_step_argument_raw"],
                            **compare(post, rows[target_index]),
                            "evidence": evidence,
                        }
                    )
                case["completed"] = True
            finally:
                if session is not None:
                    close_case(session, case)
            case["compared_targets"] = len(case["rows"])
            case["physics6_exact_targets"] = sum(row["physics6_exact"] for row in case["rows"])
            case["first_mismatch"] = next(
                (row for row in case["rows"] if not row["physics6_exact"]), None
            )
            case["endpoint"] = case["rows"][-1]
            print(
                json.dumps(
                    {
                        "case": name,
                        "targets": case["compared_targets"],
                        "exact": case["physics6_exact_targets"],
                        "endpoint_Y_difference": case["endpoint"]["Y_difference"],
                        "endpoint_VY_difference": case["endpoint"]["VY_difference"],
                        "remainingOwned": case["cleanup"]["remainingOwned"],
                    }
                ),
                flush=True,
            )
        report["completed"] = all(
            case.get("completed") is True for case in report["cases"].values()
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["whole_main_seconds"] = time.monotonic() - begin
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--neutral-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--v6-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
