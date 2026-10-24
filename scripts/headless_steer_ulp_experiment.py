"""Bounded counterfactual using two actually observed adjacent steering values.

These variants are hypotheses, never a reconstruction of an ambiguous tick or
qualification of Windows applied inputs. Production sources and pins stay fixed.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

from scripts.diagnostic_inputs import configured_path
from scripts.headless_replay_windows_tapes import ObservedSession


def measured_candidates(trace: dict[str, Any], race: int) -> tuple[float, float]:
    samples = []
    for row in trace["cases"]["mild_left"]["rows"]:
        samples.extend([row["before_raw33"], *[p["raw33"] for p in row["packets"]]])
        samples.append(row["after_raw33"])
    values = sorted({struct.pack("<f", r[30]) for r in samples if r[3] == race})
    if len(values) != 2:
        raise ValueError("exactly two observed steering bit patterns required")
    words = [struct.unpack("<I", value)[0] for value in values]
    if abs(words[0] - words[1]) != 1:
        raise ValueError("observed steering candidates must be adjacent float32 values")
    return struct.unpack("<f", values[0])[0], struct.unpack("<f", values[1])[0]


def main() -> None:
    output = Path(sys.argv[1])
    if sys.platform != "linux" or not output.is_absolute() or str(output).startswith("/mnt/"):
        raise ValueError("fresh Linux/ext4 experiment required")
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path(str(configured_path("TRACKMANIARL_NATIVE_CONFIG")))
    receipt_path = Path(str(configured_path("TRACKMANIARL_NATIVE_RECEIPT")))
    baseline_path = Path(str(configured_path("TRACKMANIARL_NATIVE_REFERENCE")))
    trace_path = Path(
        "/mnt/f/AITrackmania/reports/headless-training-parity-20261021-r4/"
        "fresh006-timing-fixed002/windows-timing.json"
    )
    config = json.loads(config_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    baseline = json.loads(baseline_path.read_bytes())["cases"]["mild_left"]["rows"]
    candidates = measured_candidates(json.loads(trace_path.read_bytes()), 1530)
    if len(baseline) != 200:
        raise ValueError("finite original 200-interval baseline required")
    deployment = Path(config["deployment_manifest"]).read_bytes()
    if hashlib.sha256(deployment).hexdigest() != config["deployment_manifest_sha256"]:
        raise ValueError("original deployment manifest differs")
    driver = json.loads(deployment)["driverBuildApproval"]["driverSha256"]
    report: dict[str, Any] = {
        "schema": "tmrl.steer-ulp-counterfactual/v1",
        "qualification_claimed": False,
        "windows_input_reconstruction_claimed": False,
        "learning_executed": False,
        "production_modified": False,
        "candidates": list(candidates),
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (config_path, receipt_path, baseline_path, trace_path)
        },
        "cases": {},
    }
    begin = time.monotonic()
    try:
        for variant in ("all-steering", "only-1530", "from-1530"):
            case: dict[str, Any] = {"rows": [], "completed": False}
            report["cases"][variant] = case
            session = None
            try:
                session = ObservedSession(
                    Path(config["runtime_root"]),
                    receipt,
                    output / (output.name + "-" + variant),
                    2190,
                    driver,
                )
                session.reset()
                case["administrative_neutral"] = session.step(36, 1)
                for old in baseline:
                    if (output / "STOP").exists() or time.monotonic() - begin > 170:
                        raise InterruptedError("counterfactual STOP/deadline")
                    measured_source = dict(old["evidence"]["waveformSource"])
                    source = {
                        "raceMs": measured_source["raceMs"],
                        "sgbHex": measured_source["sgbHex"],
                        "original_sgb_hex": measured_source["sgbHex"],
                        "counterfactual": True,
                        "measured_reference": measured_source,
                    }
                    steer, gas, brake = struct.unpack("<3f", bytes.fromhex(source["sgbHex"]))
                    change = steer != 0 and (
                        variant == "all-steering"
                        or (variant == "only-1530" and source["raceMs"] == 1530)
                        or (variant == "from-1530" and source["raceMs"] >= 1530)
                    )
                    if change:
                        if steer not in candidates:
                            raise ValueError("baseline steer differs from measured pair")
                        steer = next(value for value in candidates if value != steer)
                        source["sgbHex"] = struct.pack("<3f", steer, gas, brake).hex()
                    evidence = session.step_observed(source)
                    raw = bytes.fromhex(evidence["postHex"])
                    case["rows"].append(
                        {
                            "race_ms": old["afterRaceMs"],
                            "hypothesis_input_changed": change,
                            "matches_baseline_native": raw[168:192].hex()
                            == old["native_physics6_hex"],
                            "matches_old_windows": raw[168:192].hex()
                            == old["windows_physics6_hex"],
                            "physics6_hex": raw[168:192].hex(),
                            "evidence": evidence,
                        }
                    )
                case["completed"] = True
            except BaseException as error:
                case["error"] = repr(error)
                raise
            finally:
                try:
                    if session is not None:
                        session.close(error=not case["completed"])
                except BaseException as error:
                    case["cleanup_error"] = repr(error)
                    raise
                finally:
                    if session is not None:
                        case["cleanup"] = getattr(session, "cleanup", None)
            case["matches_baseline_native"] = sum(
                row["matches_baseline_native"] for row in case["rows"]
            )
            case["matches_old_windows"] = sum(row["matches_old_windows"] for row in case["rows"])
            print(
                json.dumps(
                    {
                        variant: {
                            k: case[k]
                            for k in ("completed", "matches_baseline_native", "matches_old_windows")
                        }
                    }
                ),
                flush=True,
            )
    finally:
        report["whole_main_seconds"] = time.monotonic() - begin
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
