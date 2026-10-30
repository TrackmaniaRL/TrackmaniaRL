"""Bounded source-reference demonstration through the unchanged V7 public facade.

No model, optimization or learner replay. Canonical action IDs are selected only
when their N1 SGB bytes exactly equal the original source packet at current p.
The production neutral reset, EOF qualification and input bounds stay intact.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import features
from trackmaniarl.trackmania.native_headless._runtime.outcome import MAP_UID
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.immediate_environment import NativeImmediateEnvironment

EXPORT_SHA256 = "214b54f8b79c40af5db23375bd88ab9ff817d2de98315245de2a9c6792887777"


def source_actions(raw: bytes) -> tuple[dict[int, int], dict[str, Any]]:
    if hashlib.sha256(raw).hexdigest() != EXPORT_SHA256:
        raise ValueError("exact supplied Jimmy source export required")
    document = json.loads(raw)
    if document["metadata"]["map_uid_literals"] != [MAP_UID]:
        raise ValueError("Jimmy source map UID mismatch")
    packets = document["input_archives"][0]["packets"]
    if len(packets) != 1923:
        raise ValueError("exact Jimmy source packet count required")
    mapping: dict[bytes, int] = {}
    for action in range(78):
        key = struct.pack("<3f", *features.action_sgb(action)[0])
        mapping.setdefault(key, action)
    plan = {}
    for index, packet in enumerate(packets):
        if packet["tick_index"] != index or packet["race_time_ms"] != -1510 + index * 10:
            raise ValueError("source packet index and literal input clock must be consecutive")
        if packet["tick_index"] < 151:
            continue
        key = struct.pack("<3f", packet["steer_normalized"], packet["accel"], packet["brake"])
        if key not in mapping:
            raise ValueError("source input is not exactly available in action78/N1")
        plan[packet["tick_index"]] = mapping[key]
    return plan, {
        "source_packets": len(packets),
        "finite_driving_source_actions": len(plan),
        "source_export_sha256": EXPORT_SHA256,
        "action_quantization_error": 0,
        "action_ids": sorted(set(plan.values())),
        "source_finish_ms": 17715,
    }


def main() -> None:
    config_path, source_path, output = map(Path, sys.argv[1:4])
    if output.exists():
        raise FileExistsError("fresh public reference-demonstration directory required")
    original = json.loads(config_path.read_bytes())
    original.update(
        trace_dir=str(output / "trace"),
        work_dir=str(Path(original["runtime_root"]) / "build/server-probe" / output.name),
        timeout_s=120,
        max_trace_bytes=64 * 1024 * 1024,
    )
    config = NativeRuntimeConfig.from_mapping(original)
    plan, source = source_actions(source_path.read_bytes())
    output.mkdir(parents=True)
    (output / "source-headless_finish_reference.py").write_bytes(Path(__file__).read_bytes())
    (output / "run-config.json").write_text(json.dumps(original, indent=2) + "\n")
    report: dict[str, Any] = {
        "schema": "tmrl.public-v7-source-finish-reference/v1",
        "finished": False,
        "learning_executed": False,
        "learner_credit_intervals": 0,
        "public_facade": "NativeImmediateEnvironment.step_native(action78,N1)",
        "source": source,
        "rows": 0,
        "actual_intervals": 0,
    }
    facade = NativeImmediateEnvironment(config)
    started, completed = time.monotonic(), False
    try:
        state, info = facade.reset()
        report["reset"] = {"state": state, "info": info}
        session = facade._session
        immutable = (session.out / "parent.inputs.bin").read_bytes()
        with (output / "steps.jsonl").open("x", encoding="utf-8") as stream:
            for decision in range(2000):
                if (output / "STOP").exists():
                    raise InterruptedError("public reference demonstration cooperative STOP")
                p = state["currentRecordIndex"]
                if p in plan:
                    action, provenance = plan[p], "literal normalized source Jimmy packet"
                elif 1923 <= p < 4151:
                    original_sgb = immutable[p * 32 + 4 : p * 32 + 16]
                    if original_sgb != bytes(12):
                        raise ValueError("post-source original tail must be explicitly neutral")
                    action, provenance = 36, "literal neutral tail of approved long tape"
                else:
                    raise ValueError(
                        "current input index outside exact source/declared neutral tail"
                    )
                result = facade.step_native(action, native_iterations=1)
                stream.write(
                    json.dumps(
                        {
                            "decision": decision,
                            "p": p,
                            "action": action,
                            "source": provenance,
                            "result": result,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
                report["rows"] += 1
                report["actual_intervals"] += result["actualIterations"]
                state = result["observation"]
                if result["terminated"] or result["truncated"]:
                    report["last_result"] = result
                    report["finished"] = result["terminated"]
                    completed = True
                    break
            if not completed:
                report["stop_reason"] = "finite2000-decision demonstration cap; no finish invented"
                completed = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            facade.close(error=not completed)
        finally:
            report["wall_seconds_not_a_benchmark"] = time.monotonic() - started
            with (output / "report.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
