"""Explicitly authorized continuation on the original frozen runtime and deadline."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

BASE = Path("H:/Studia/inzynierskie/inzynierkav2/AITrackmania/artifacts/tmrl-test-comparison")
FROZEN = BASE / "ppo-trend-runtime-20261013"
ORIGINAL = BASE / "ppo-trend-20261013"
QUEUE = BASE / "ppo-trend-continuation-20261013"
RUN = BASE / "tmrl-ppo-trend-s17"
CHECKPOINT = RUN / "checkpoints/update-00000050.pt"
EXPECTED_SHA = "7fba0d9bf019aecc65ae6a7924a1614caa0841c9da0d30295e38522934a6dcb6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--authorized-continuation-seconds", type=int, choices=[7200])
    args = parser.parse_args()
    # Import training and supervision exclusively from the unchanged frozen checkout.
    sys.path.insert(0, str(FROZEN))
    import psutil
    import torch

    from pilot_tools import ppo_pilot as pilot
    from trackmaniarl.observability.wandb_metrics import _load_wandb_key_from_dotenv

    plan = pilot.verify()
    proof = json.loads((ORIGINAL / "post-closure-verification.json").read_text())
    launch = json.loads((ORIGINAL / "launch.json").read_text())
    assert proof["all_owned_closed"]
    assert proof["mutex_free"]
    assert pilot.sha(CHECKPOINT) == EXPECTED_SHA
    assert pilot.sha(ORIGINAL / "STOP") == proof["stop_sha256"]
    for item in proof["owned_identities"]:
        try:
            process = psutil.Process(item["pid"])
            assert abs(process.create_time() - item["create_time"]) > .001
        except psutil.NoSuchProcess:
            pass
    original_stops = pilot.stops()
    helper_sha = pilot.sha(Path(__file__))
    assert not any(p.exists() for p in original_stops if p != ORIGINAL / "STOP")
    actual_start = datetime.fromisoformat(launch["started_at"])
    elapsed = (datetime.now(UTC) - actual_start).total_seconds()
    budget = args.authorized_continuation_seconds
    remaining = (budget - 600) if budget else (13800 - elapsed)
    if remaining < 6000:
        raise RuntimeError("Insufficient time for 50 remaining rollouts plus final evaluation")
    assert not (QUEUE / "launch.json").exists(), "Continuation already used"
    assert not (QUEUE / "STOP").exists(), "Continuation STOP active"
    if not args.launch:
        print(json.dumps({"validated": True, "checkpoint_sha256": EXPECTED_SHA,
                          "original_elapsed_seconds": elapsed, "remaining_updates": 50}))
        return
    assert torch.cuda.is_available()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
    _load_wandb_key_from_dotenv(str(pilot.KEY_SOURCE))
    assert os.environ.get("WANDB_API_KEY"), "Private W&B key unavailable"
    pilot.powershell_probe(Path(plan["powershell"]["executable"]))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, False, "Global\\TrackmaniaRL.ComparisonController")
    assert handle
    acquired = kernel.WaitForSingleObject(handle, 0) in {0, 128}
    if not acquired:
        kernel.CloseHandle(handle)
        raise RuntimeError("Comparison controller already owned")
    QUEUE.mkdir(exist_ok=True)
    state = {"started_at": datetime.now(UTC).isoformat(),
             "original_started_at": launch["started_at"], "status": "RUNNING",
             "run_id": pilot.RUN_ID, "launcher": {"pid": os.getpid(),
             "create_time": psutil.Process().create_time()}, "stop_reason": None,
             "owned_identities": [], "automatic_full_training": False}
    # Only the already preserved supervisor STOP is superseded by the human's
    # explicit continuation request. Its exact hash is checked every supervisor tick.
    def continuation_stops() -> list[Path]:
        paths = [p for p in original_stops if p != ORIGINAL / "STOP"] + [QUEUE / "STOP"]
        if not (ORIGINAL / "STOP").exists() or pilot.sha(ORIGINAL / "STOP") != proof["stop_sha256"]:
            paths.append(ORIGINAL / "STOP")
            (QUEUE / "STOP").touch(exist_ok=True)
        if pilot.sha(Path(__file__)) != helper_sha:
            (QUEUE / "STOP").touch(exist_ok=True)
        return paths

    try:
        with (QUEUE / "launch.json").open("x") as target:
            json.dump(state, target, indent=2)
        with (QUEUE / "authorization.json").open("x") as target:
            json.dump({"human_request": "Znaczy chyba dokoncz obecny pilot by dzis to skonczyc",
                       "deadline_authorization": "Tak, kontynuuj z limitem 2 h" if budget else None,
                       "checkpoint_sha256": EXPECTED_SHA, "frozen_commit": plan["source_commit"],
                       "preserved_supervisor_stop_sha256": proof["stop_sha256"],
                       "supervisor_helper_sha256": helper_sha,
                       "original_deadline_retained": budget is None,
                       "authorized_continuation_seconds": budget,
                       "configuration_changed": False,
                       "midpoint_repeated": False, "full_training": False}, target, indent=2)
        pilot.DIRECTORY = QUEUE
        pilot.stops = continuation_stops
        if budget:
            plan = dict(plan, maximum_seconds=budget)
        started = time.monotonic() - (0 if budget else elapsed)
        config = str(ORIGINAL / "ppo.yaml")
        stages = [
            ("preflight", ["-m", "experiments.tmrl_test_comparison.preflight", config,
                           "--receipt", str(QUEUE / "preflight.json"),
                           "--stop-file", str(QUEUE / "STOP")]),
            ("train-and-final-evaluation", ["-m", "pilot_tools.train_trend", "resume", config,
                                           str(CHECKPOINT), "--stop-file", str(QUEUE / "STOP")]),
        ]
        for name, command in stages:
            state["stage"] = name
            pilot._run_stage(plan, state, (name, command, started))
        run = pilot.LaunchRun.load(ORIGINAL / "ppo.yaml")
        checkpoint = pilot.final_checkpoint(run)
        evaluation = json.loads((RUN / "evaluation.json").read_text())
        pilot.validate_traces(RUN, (checkpoint,))
        timing_error = None
        try:
            pilot.pilot_measurement(checkpoint, evaluation)
        except RuntimeError as error:
            timing_error = str(error)
        state.update(status="COMPLETE", ended_at=datetime.now(UTC).isoformat(),
                     checkpoint=str(checkpoint), checkpoint_sha256=pilot.sha(checkpoint),
                     finishes=sum(t.get("finished") is True for t in evaluation["trials"]),
                     measurement_error=timing_error, gate_passed=timing_error is None
                     and sum(t.get("finished") is True for t in evaluation["trials"]) >= 8)
        pilot._save_status(state)
    except BaseException as error:
        state.update(status="STOPPED_OR_FAILED", error_type=type(error).__name__,
                     ended_at=datetime.now(UTC).isoformat())
        pilot._save_status(state)
        raise
    finally:
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    main()
