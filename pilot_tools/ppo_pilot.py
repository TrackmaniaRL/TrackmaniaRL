"""Prepare/verify a fresh PPO pilot; only explicit launch may touch game controls."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

from experiments.tmrl_test_comparison import campaign
from experiments.tmrl_test_comparison.launch_checks import LaunchRun, final_checkpoint
from experiments.tmrl_test_comparison.owned_processes import wait_for_owned_processes
from trackmaniarl.core.fingerprint import run_fingerprint
from trackmaniarl.core.spec import RunSpec

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("H:/Studia/inzynierskie/inzynierkav2/AITrackmania/artifacts/tmrl-test-comparison")
DIRECTORY = BASE / "ppo-pilot-20261013"
RUN_ID = "tmrl-ppo-p13-prefix-s17"
BUDGET = 145_408
HORIZON = 8_192_000
MAP_UID = "oqIJ5rQDRrNwLPTh9H2p_W4tLof"
KEY_SOURCE = Path("C:/Users/szulc/.codex/worktrees/tmrl-training-fixes/AITrackmania")
CALIBRATION = KEY_SOURCE / "artifacts/tmrl-test-comparison/steering-local.json"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def powershell_probe(engine: Path) -> dict[str, Any]:
    """Use the selected engine's own module path, preserving all security policies."""
    environment = {
        key: value for key, value in os.environ.items() if key.casefold() != "psmodulepath"
    }
    script = (
        "$ErrorActionPreference='Stop'; "
        "@{version=$PSVersionTable.PSVersion.ToString(); "
        "policy=(Get-ExecutionPolicy).ToString(); "
        "edition=$PSVersionTable.PSEdition} | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [str(engine), "-NoProfile", "-NonInteractive", "-Command", script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=True,
    )
    data: dict[str, Any] = json.loads(result.stdout)
    if int(data["version"].split(".")[0]) < 7 or data["policy"] == "Restricted":
        raise RuntimeError("A supported PowerShell 7 with an approved script policy is required")
    return {"executable": str(engine.resolve()), **data, "security_policy_changed": False}


def configuration() -> dict[str, Any]:
    plan = campaign.verify()
    job = next(j for j in plan["jobs"] if j["algorithm"] == "ppo" and j["seed"] == 17)
    path = ROOT / job["config"]
    data: dict[str, Any] = copy.deepcopy(RunSpec.from_yaml(path).model_dump(mode="json"))
    data.update(run_id=RUN_ID, artifacts_dir=str(BASE))
    data["components"]["environment"]["kwargs"]["config"]["steering_curve_path"] = str(CALIBRATION)
    data["training"]["total_transitions"] = BUDGET
    learner = data["components"]["learner"]
    learner["class_path"] = "pilot_tools.ppo_prefix:CampaignPrefixPPO"
    learner["kwargs"]["schedule_transitions"] = HORIZON
    data["evaluation"].update(
        trials_per_map=10, min_finish_rate=0.8, target_median_s=None, target_mean_s=None
    )
    data["metadata"].update(
        stage="fresh-pilot-20261013",
        purpose="Complete 71-rollout diagnostic prefix; LR horizon matches weeks campaign",
        automatic_full_training=False,
        resume=False,
    )

    def absolute_assets(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"geometry_path", "map_path", "steering_curve_path"} and item:
                    value[key] = str((path.parent / item).resolve())
                else:
                    absolute_assets(item)
        elif isinstance(value, list):
            for item in value:
                absolute_assets(item)

    absolute_assets(data)
    RunSpec.model_validate(data)
    return data


def pins() -> dict[str, str]:
    result: dict[str, str] = campaign.source_pins()
    for path in sorted((ROOT / "pilot_tools").glob("*")):
        if path.suffix in {".py", ".ps1"}:
            result[path.relative_to(ROOT).as_posix()] = campaign.digest(path)
    return result


def prepare(engine: Path) -> dict[str, Any]:
    if DIRECTORY.exists() or (BASE / RUN_ID).exists():
        raise RuntimeError("Fresh pilot identity already exists; never overwrite/restart")
    shell = powershell_probe(engine)
    data = configuration()
    spec = RunSpec.model_validate(data)
    curve = Path(data["components"]["environment"]["kwargs"]["config"]["steering_curve_path"])
    if not curve.is_file():
        raise RuntimeError("This host's steering calibration is missing")
    fingerprint = run_fingerprint(spec, DIRECTORY)
    plan = {
        "schema_version": 1,
        "status": "PREPARED_NOT_LAUNCHED",
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_root": str(ROOT),
        "python_executable": sys.executable,
        "private_key_source_root": str(KEY_SOURCE),
        "source_pins": pins(),
        "run_id": RUN_ID,
        "seed": 17,
        "total_transitions": BUDGET,
        "rollout_length": 2048,
        "expected_updates": 71,
        "learning_rate_schedule_transitions": HORIZON,
        "evaluation_trials": 10,
        "wandb_entity": "dsc-pjatk-warsaw",
        "minimum_finishes": 8,
        "maximum_step_race_time_ms": 100,
        "maximum_seconds": 14400,
        "save_margin_seconds": 600,
        "minimum_free_disk_gib": 20,
        "powershell": shell,
        "steering_path": str(curve),
        "steering_sha256": sha(curve),
        "run_fingerprint": fingerprint,
        "automatic_restart": False,
        "automatic_full_training": False,
        "preflight_performed": False,
        "controller_created": False,
    }
    DIRECTORY.mkdir(parents=True)
    config = DIRECTORY / "ppo.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    plan["config_sha256"] = sha(config)
    (DIRECTORY / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def verify() -> dict[str, Any]:
    plan: dict[str, Any] = json.loads((DIRECTORY / "plan.json").read_text(encoding="utf-8"))
    if plan["source_root"] != str(ROOT) or plan["source_pins"] != pins():
        raise RuntimeError("Frozen pilot sources differ; do not launch or resume")
    if sha(DIRECTORY / "ppo.yaml") != plan["config_sha256"]:
        raise RuntimeError("Frozen pilot configuration changed")
    actual = RunSpec.from_yaml(DIRECTORY / "ppo.yaml")
    if actual.model_dump(mode="json") != RunSpec.model_validate(configuration()).model_dump(
        mode="json"
    ):
        raise RuntimeError("Only the declared PPO diagnostic prefix is allowed")
    if sha(Path(plan["steering_path"])) != plan["steering_sha256"]:
        raise RuntimeError("Pinned local steering calibration changed")
    if run_fingerprint(actual, DIRECTORY) != plan["run_fingerprint"]:
        raise RuntimeError("Pilot fingerprint differs")
    expected = {
        "status": "PREPARED_NOT_LAUNCHED",
        "maximum_seconds": 14400,
        "save_margin_seconds": 600,
        "minimum_free_disk_gib": 20,
        "automatic_restart": False,
        "automatic_full_training": False,
        "total_transitions": BUDGET,
        "learning_rate_schedule_transitions": HORIZON,
        "evaluation_trials": 10,
        "wandb_entity": "dsc-pjatk-warsaw",
        "expected_updates": 71,
        "minimum_finishes": 8,
        "maximum_step_race_time_ms": 100,
    }
    if any(plan[key] != value for key, value in expected.items()):
        raise RuntimeError("Pilot limits/decision contract changed")
    return plan


def stops() -> list[Path]:
    return [
        DIRECTORY / "STOP",
        BASE / "STOP",
        ROOT / "artifacts/STOP",
        ROOT / "artifacts/STOP-ppo",
        BASE.parent / "STOP",
        BASE.parent / "STOP-ppo",
    ]


def launch() -> None:
    plan = verify()
    active = [str(path) for path in stops() if path.exists()]
    if active:
        raise RuntimeError(f"STOP active; launch blocked: {active}")
    if (DIRECTORY / "launch.json").exists() or (BASE / RUN_ID).exists():
        raise RuntimeError("Pilot already used; no duplicate, restart or resume")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Pilot source checkout must be clean")
    if shutil.disk_usage(BASE).free < plan["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("Pilot requires 20 GiB free on its artifact drive")
    powershell_probe(Path(plan["powershell"]["executable"]))
    import torch

    from trackmaniarl.observability.wandb_metrics import _load_wandb_key_from_dotenv

    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("CUDA unavailable")
    _load_wandb_key_from_dotenv(str(KEY_SOURCE))
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("This host's private W&B key is required")
    if os.environ.get("WANDB_ENTITY") not in {None, plan["wandb_entity"]}:
        raise RuntimeError("Configured W&B entity differs from the prepared shared project")
    busy = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        command = " ".join(process.info["cmdline"] or [])
        if (
            process.pid != os.getpid()
            and process.info["name"] == "python.exe"
            and any(
                token in command
                for token in (
                    "trackmaniarl train",
                    "trackmaniarl resume",
                    "trackmaniarl benchmark",
                    "tmrl_test_comparison.preflight",
                )
            )
        ):
            busy.append(process.pid)
    if busy:
        raise RuntimeError(f"Another controller may be active: {busy}")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mutex = kernel.CreateMutexW(None, False, "Global\\TrackmaniaRL.ComparisonController")
    if not mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = kernel.WaitForSingleObject(mutex, 0) in {0, 0x80}
    if not acquired:
        kernel.CloseHandle(mutex)
        raise RuntimeError("Comparison controller mutex is owned")
    try:
        _launch_owned(plan)
    except BaseException as error:
        status = DIRECTORY / "status.json"
        state = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        state.update(
            status="STOPPED_OR_FAILED",
            error_type=type(error).__name__,
            ended_at=datetime.now(UTC).isoformat(),
            automatic_full_training=False,
        )
        _save_status(state)
        raise
    finally:
        kernel.ReleaseMutex(mutex)
        kernel.CloseHandle(mutex)


def _launch_owned(plan: dict[str, Any]) -> None:
    started = time.monotonic()
    state: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "run_id": RUN_ID,
        "launcher": {"pid": os.getpid(), "create_time": psutil.Process().create_time()},
        "owned_identities": [],
        "stop_reason": None,
    }
    with (DIRECTORY / "launch.json").open("x", encoding="utf-8") as target:
        target.write(json.dumps(state, indent=2) + "\n")
    config = str(DIRECTORY / "ppo.yaml")
    commands = [
        (
            "preflight",
            [
                "-m",
                "experiments.tmrl_test_comparison.preflight",
                config,
                "--require-new-run",
                "--receipt",
                str(DIRECTORY / "preflight.json"),
                "--stop-file",
                str(DIRECTORY / "STOP"),
            ],
        ),
        (
            "train-and-final-evaluation",
            ["-m", "trackmaniarl", "train", config, "--stop-file", str(DIRECTORY / "STOP")],
        ),
    ]
    for name, arguments in commands:
        state["stage"] = name
        _run_stage(plan, state, (name, arguments, started))
    run = LaunchRun.load(DIRECTORY / "ppo.yaml")
    checkpoint = final_checkpoint(run)
    evaluation = json.loads((run.directory / "evaluation.json").read_text(encoding="utf-8"))
    finishes = pilot_measurement(checkpoint, evaluation)
    state.update(
        status="COMPLETE",
        ended_at=datetime.now(UTC).isoformat(),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        finishes=finishes,
        gate_passed=finishes >= 8,
        automatic_full_training=False,
    )
    _save_status(state)


def pilot_measurement(checkpoint: Path, evaluation: dict[str, Any]) -> int:
    from trackmaniarl.commands.evaluation import _has_runtime_errors, _runtime_trial_statistics

    trials = evaluation.get("trials", [])
    maximum_step, timing_valid = _runtime_trial_statistics(trials)
    if (
        evaluation.get("status") != "complete"
        or evaluation.get("expected_trials") != 10
        or len(trials) != 10
        or Path(evaluation.get("checkpoint") or "").resolve() != checkpoint.resolve()
        or evaluation.get("checkpoint_sha256") != sha(checkpoint)
        or any(t.get("map_uid") != MAP_UID for t in trials)
        or _has_runtime_errors(trials)
        or not timing_valid
        or maximum_step is None
        or maximum_step > 100
        or {t.get("trial_index") for t in trials} != set(range(10))
        or any(
            t.get("termination_reason")
            in {"operator_interruption", "capture_interruption", "telemetry_error"}
            for t in trials
        )
    ):
        raise RuntimeError("Final evaluation is incomplete or invalid; retain evidence")
    return sum(t.get("finished") is True for t in trials)


def _save_status(state: dict[str, Any]) -> None:
    temporary = DIRECTORY / "status.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(DIRECTORY / "status.json")


def _run_stage(
    plan: dict[str, Any], state: dict[str, Any], command: tuple[str, list[str], float]
) -> None:
    name, arguments, started = command
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONNOUSERSITE="1",
        WANDB_ENTITY=plan["wandb_entity"],
        WANDB_MODE="online",
    )
    owned: dict[tuple[int, float], psutil.Process] = {}
    stop_file = DIRECTORY / "STOP"
    stage_started = time.monotonic()
    with (DIRECTORY / f"{name}.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            [sys.executable, "-u", *arguments],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        parent = psutil.Process(child.pid)
        owned[(parent.pid, parent.create_time())] = parent
        state["owned_identities"] = [{"pid": p, "create_time": t} for p, t in sorted(owned)]
        _save_status(state)
        while child.poll() is None:
            for known in list(owned.values()):
                try:
                    if known.is_running():
                        for descendant in known.children(recursive=True):
                            owned[(descendant.pid, descendant.create_time())] = descendant
                except psutil.NoSuchProcess:
                    pass
            elapsed = time.monotonic() - started
            reasons = ["user_stop"] if any(p.exists() for p in stops()) else []
            if elapsed >= plan["maximum_seconds"] - plan["save_margin_seconds"]:
                reasons.append("save_deadline")
            if name == "preflight" and time.monotonic() - stage_started >= 60:
                reasons.append("preflight_deadline")
            if shutil.disk_usage(BASE).free < 5 * 1024**3:
                reasons.append("low_disk")
            if reasons and state["stop_reason"] is None:
                state.update(status="SAVING", stop_reason=reasons[0])
                try:
                    with stop_file.open("x", encoding="utf-8") as target:
                        target.write(reasons[0] + "\n")
                except FileExistsError:
                    pass
            state["owned_identities"] = [{"pid": p, "create_time": t} for p, t in sorted(owned)]
            state["elapsed_seconds"] = round(elapsed, 2)
            _save_status(state)
            if elapsed >= plan["maximum_seconds"]:
                for process in reversed(list(owned.values())):
                    if process.is_running():
                        process.terminate()
                break
            time.sleep(0.5)
        alive = wait_for_owned_processes(owned.values(), timeout=30)
        if alive:
            state["stop_reason"] = "owned_descendants_survived_launcher"
            for process in reversed(alive):
                if process.is_running():
                    process.terminate()
            alive = wait_for_owned_processes(owned.values(), timeout=30)
        state["surviving_identities"] = [
            {"pid": p.pid, "create_time": p.create_time()} for p in alive
        ]
        state["returncode"] = child.poll()
        if (
            alive
            or child.returncode != 0
            or state["stop_reason"] is not None
            or any(p.exists() for p in stops())
        ):
            state["status"] = "STOPPED_OR_FAILED"
            _save_status(state)
            raise RuntimeError(
                "Pilot stage incomplete; preserve evidence; no retry or full training"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "launch"))
    parser.add_argument("--powershell", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.powershell is None:
            parser.error("--powershell must name the verified installed PowerShell 7")
        plan = prepare(args.powershell)
    else:
        plan = verify()
        if args.command == "launch":
            launch()
            plan = json.loads((DIRECTORY / "status.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": plan["status"],
                "run_id": RUN_ID,
                "transitions": BUDGET,
                "schedule_horizon": HORIZON,
                "evaluation_trials": 10,
                "launch_requested": args.command == "launch",
            }
        )
    )


if __name__ == "__main__":
    main()
