"""Summarize fresh probes without promoting unmatched trajectories to physics proof."""

import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path

from trackmaniarl.trackmania.native_headless._runtime.features import native16
from trackmaniarl.trackmania.parity import require_learning_comparison


def main() -> None:
    native_path, windows_path, replay_path, output = map(Path, sys.argv[1:5])
    native, windows, replay = [
        json.loads(p.read_bytes()) for p in (native_path, windows_path, replay_path)
    ]
    cases = {c["name"]: c for c in native["cases"]}
    report = {
        "schema": "tmrl.headless-parity-qualification/v1",
        "status": "nadal niezakwalifikowany",
        "native_infrastructure_passed": native["passed"],
        "windows_probe_passed": windows["passed"],
        "replay_integration_passed": replay["passed"],
        "evidence_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (native_path, windows_path, replay_path)
        },
        "learning_gate": {
            "reset_qualified": False,
            "applied_inputs_qualified": False,
            "clock_mapping_qualified": False,
            "observation_semantics_qualified": False,
            "same_model_reward_budget": False,
        },
        "comparisons": [],
    }
    for name, win in windows["cases"].items():
        case = cases[name]
        raw = native16(bytes.fromhex(case["initial"]["native76Hex"]))
        win_initial = win["initial_raw33"]
        own_end = native16(bytes.fromhex(case["rows"][-1]["result"]["observation"]["native76Hex"]))
        win_end = win["rows"][-1]["after_raw33"]
        report["comparisons"].append(
            {
                "name": name,
                "issued_actions_equal": [r["action"] for r in case["rows"]]
                == [r["action"] for r in win["rows"]],
                "native_initial": raw,
                "windows_initial_common8": win["common8"],
                "windows_initial_race_ms": win_initial[3],
                "native_initial_t": case["initial"]["nativeT"],
                "native_initial_pending_sgb": case["initial"]["pendingSGB"],
                "initial_position_distance_coordinate_units": math.dist(
                    raw["physicalPosition"], win_initial[4:7]
                ),
                "initial_velocity_distance_readout_units": math.dist(
                    raw["physicalVelocity"], win_initial[7:10]
                ),
                "end_position_distance_coordinate_units": math.dist(
                    own_end["physicalPosition"], win_end[4:7]
                ),
                "physics_regression_claim": False,
            }
        )
        # Hex snapshots stay in private ext4 evidence, never in the summary.
        report["comparisons"][-1]["native_initial"].pop("raw76Hex")
    report["reset_states"] = {}
    for name in ("legacy-reset", "neutral-reset"):
        state = cases[name]["initial"]
        raw = native16(bytes.fromhex(state["native76Hex"]))
        raw.pop("raw76Hex")
        report["reset_states"][name] = {
            "state": raw,
            "native_t": state["nativeT"],
            "pending_sgb": state["pendingSGB"],
        }
    steps = [r["host_seconds"] for c in cases.values() for r in c["rows"]]
    resets = [c["reset_host_seconds"] for c in cases.values()]
    report["timing"] = {
        "native_main_seconds": native["full_host_seconds"],
        "windows_main_seconds": windows["host_seconds"],
        "step_p50_seconds": statistics.median(steps),
        "step_p95_seconds": sorted(steps)[math.ceil(0.95 * len(steps)) - 1],
        "reset_p50_seconds": statistics.median(resets),
        "reset_p95_seconds": sorted(resets)[math.ceil(0.95 * len(resets)) - 1],
    }
    report["resources"] = {
        "python": platform.python_version(),
        "system": platform.platform(),
        "native_main_max_rss_kib": native["max_rss_kib"],
        "native_probe_bytes": native["disk_bytes"],
    }
    report["cleanup"] = {
        "cases": len(cases),
        "captured_identities": sum(len(c.get("owned_identities", [])) for c in cases.values()),
        "remaining_owned": [p for c in cases.values() for p in c.get("remaining_owned", [])],
        "cleanup_errors": [
            c["close"]["cleanupError"]
            for c in cases.values()
            if c.get("close", {}).get("cleanupError")
        ],
    }
    report["replay"] = replay
    try:
        require_learning_comparison(report["learning_gate"])
    except ValueError as error:
        report["comparison_training_not_run_reason"] = str(error)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "timing", "resources", "cleanup", "reset_states")}
        )
    )


if __name__ == "__main__":
    main()
