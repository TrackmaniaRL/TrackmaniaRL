"""Fresh, isolated PPO repair trials; no resume and no automatic full campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from experiments.tmrl_test_comparison.launch_checks import LaunchRun, final_checkpoint
from pilot_tools import ppo_pilot as pilot
from pilot_tools.ppo_regression_gate import _suite

WINDOW_END = datetime(2026, 10, 14, 9, 4, 23, tzinfo=UTC)
PREDECESSOR = pilot.BASE / "tmrl-ppo-trend-s17/checkpoints/update-00000100.pt"
PREDECESSOR_SHA = "a5927a3187bc2533b99dfdca7a336181dd0bd1f0416aab0ced3aa491c5aa5e60"
ORIGINAL_CONFIGURATION = pilot.configuration


def measurement(checkpoint: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Keep timing failure visible without confusing it with corrupt measurements."""
    result = _suite(evaluation)
    if (not result["measurement_valid"]
            or Path(evaluation.get("checkpoint") or "").resolve() != checkpoint.resolve()
            or evaluation.get("checkpoint_sha256") != pilot.sha(checkpoint)):
        raise RuntimeError("Incomplete, erroneous or misbound evaluation; preserve and close")
    result["strict_gate_passed"] = result["quality_gate_passed"] and result["timing_gate_passed"]
    return result


def configure(variant: str, seed: int, revision: int = 1) -> None:
    suffix = "" if revision == 1 else "-r2"
    pilot.DIRECTORY = pilot.BASE / f"ppo-overnight-{variant}-s{seed}-20261013{suffix}"
    pilot.RUN_ID = f"tmrl-ppo-overnight-{variant}-s{seed}{suffix}"

    def configuration() -> dict[str, Any]:
        data = ORIGINAL_CONFIGURATION()
        data["seed"] = seed
        if variant == "kl":
            data["components"]["learner"]["class_path"] = "pilot_tools.ppo_guarded:GuardedPrefixPPO"
            data["components"]["additional_loggers"][0]["class_path"] = (
                "pilot_tools.ppo_guarded:GuardedPpoWandbTracker")
        elif variant == "recovery":
            data["components"]["feature_pipeline"]["class_path"] = (
                "pilot_tools.ppo_recovery:PersistentPpoRecoveryPipeline")
        data["metadata"].update(stage="overnight-repair-20261013", repair_variant=variant,
                                purpose="Isolated repair; final CP100 gate; no CP selection")
        return data

    pilot.configuration = configuration
    pilot._launch_owned = launch_owned


def verify() -> dict[str, Any]:
    plan = pilot.verify()
    if (plan.get("predecessor_checkpoint") != str(PREDECESSOR)
            or plan.get("predecessor_checkpoint_sha256") != PREDECESSOR_SHA
            or pilot.sha(PREDECESSOR) != PREDECESSOR_SHA
            or plan.get("authorization_window_end") != WINDOW_END.isoformat()
            or plan.get("timing_policy") != "report_separately_no_strict_gate_waiver"):
        raise RuntimeError("Predecessor or overnight decision contract changed")
    return plan


def launch_owned(plan: dict[str, Any]) -> None:
    started = time.monotonic()
    state: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(), "status": "RUNNING",
        "run_id": pilot.RUN_ID,
        "launcher": {"pid": psutil.Process().pid, "create_time": psutil.Process().create_time()},
        "owned_identities": [], "stop_reason": None,
    }
    with (pilot.DIRECTORY / "launch.json").open("x", encoding="utf-8") as target:
        json.dump(state, target, indent=2)
    guard = subprocess.Popen(
        [sys.executable, "-m", "pilot_tools.ppo_deadline_guard", str(pilot.DIRECTORY)],
        cwd=pilot.ROOT, env=dict(os.environ, PYTHONPATH=str(pilot.ROOT), PYTHONNOUSERSITE="1"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    state["deadline_guard"] = {"pid": guard.pid,
                               "create_time": psutil.Process(guard.pid).create_time()}
    pilot._save_status(state)
    ready_deadline = time.monotonic() + 20
    while not (pilot.DIRECTORY / "deadline-guard-ready.json").exists():
        if guard.poll() is not None or time.monotonic() >= ready_deadline:
            raise RuntimeError("Independent deadline guard did not become ready")
        time.sleep(.1)
    config = str(pilot.DIRECTORY / "ppo.yaml")
    commands = [
        ("preflight", ["-m", "experiments.tmrl_test_comparison.preflight", config,
                       "--require-new-run", "--receipt", str(pilot.DIRECTORY / "preflight.json"),
                       "--stop-file", str(pilot.DIRECTORY / "STOP")]),
        ("train-and-final-evaluation", ["-m", "pilot_tools.train_overnight", "train", config,
                                       "--stop-file", str(pilot.DIRECTORY / "STOP")]),
    ]
    observed: dict[tuple[int, float], dict[str, Any]] = {}
    for name, arguments in commands:
        state["stage"] = name
        try:
            pilot._run_stage(plan, state, (name, arguments, started))
        finally:
            for identity in state["owned_identities"]:
                observed[(identity["pid"], identity["create_time"])] = identity
            state["all_observed_identities"] = list(observed.values())
            pilot._save_status(state)
    run = LaunchRun.load(pilot.DIRECTORY / "ppo.yaml")
    checkpoint = final_checkpoint(run)
    final = measurement(checkpoint, json.loads((run.directory / "evaluation.json").read_text()))
    midpoint = run.directory / "evaluation-checkpoints/update-00000050.pt"
    middle = measurement(midpoint, json.loads(
        (run.directory / "evaluations/update-00000050/evaluation.json").read_text()))
    pilot.validate_traces(run.directory, (midpoint, checkpoint))
    state.update(status="COMPLETE", ended_at=datetime.now(UTC).isoformat(),
                 checkpoint=str(checkpoint), checkpoint_sha256=pilot.sha(checkpoint),
                 final_measurement=final, midpoint_measurement=middle,
                 gate_passed=final["strict_gate_passed"], automatic_full_training=False)
    pilot._save_status(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "launch"))
    parser.add_argument("--variant", choices=("kl", "recovery", "control"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29), default=17)
    parser.add_argument("--revision", type=int, choices=(1, 2), default=1)
    parser.add_argument("--powershell", type=Path)
    args = parser.parse_args()
    configure(args.variant, args.seed, args.revision)
    if args.command == "prepare":
        if args.powershell is None:
            parser.error("--powershell is required")
        plan = pilot.prepare(args.powershell)
        plan.update(predecessor_queue=str(pilot.BASE / "ppo-trend-continuation-20261013-r2"),
                    predecessor_checkpoint=str(PREDECESSOR),
                    predecessor_checkpoint_sha256=PREDECESSOR_SHA, seed=args.seed,
                    authorization_window_end=WINDOW_END.isoformat(),
                    timing_policy="report_separately_no_strict_gate_waiver")
        path = pilot.DIRECTORY / "plan.json"
        path.write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    plan = verify()
    if args.command == "launch":
        if datetime.now(UTC) + timedelta(seconds=plan["maximum_seconds"]+120) > WINDOW_END:
            raise RuntimeError("Insufficient overnight authorization window for this bounded trial")
        pilot.launch()
    print(json.dumps({"status": pilot.observed_status(plan), "queue": str(pilot.DIRECTORY),
                      "run_id": pilot.RUN_ID, "launch_requested": args.command == "launch"}))


if __name__ == "__main__":
    main()
