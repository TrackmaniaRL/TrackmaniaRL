"""Prepare and verify a frozen multi-computer campaign without controlling the game."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil
import yaml

from experiments.tmrl_test_comparison.generate import HERE, ROOT, SEEDS, configuration
from experiments.tmrl_test_comparison.owned_processes import wait_for_owned_processes
from experiments.tmrl_test_comparison.policy import (
    CAMPAIGN_ASSIGNMENTS,
    require_standard_config,
)
from trackmaniarl.core.spec import RunSpec

PLAN = HERE / "campaign.json"
LONG_TRANSITIONS = 8_192_000
CAMPAIGN_ID = "weeks-20261008-v1"
MAP_UID = "oqIJ5rQDRrNwLPTh9H2p_W4tLof"


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".py", ".ps1", ".toml", ".lock", ".yaml", ".json"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def source_pins() -> dict[str, str]:
    paths = list((ROOT / "trackmaniarl").rglob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths += list(HERE.glob("*.py")) + list(HERE.glob("*.ps1"))
    paths += [
        ROOT / "my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx",
        ROOT / "my-trackmania-agent/assets/trackmaniarl-test.geometry.npz",
    ]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}


def prepare(transitions: int) -> dict[str, Any]:
    """Generate a new profile, refusing to overwrite a published or used plan."""
    if transitions < 2_048_000 or transitions % 2048:
        raise ValueError("Long budget must be >= 2048000 and a multiple of PPO rollout 2048")
    directory = HERE / "configs/weeks"
    if PLAN.exists() or directory.exists():
        raise RuntimeError("Campaign already prepared; create a new version instead of overwriting")
    jobs = []
    configs = []
    for worker, algorithms in CAMPAIGN_ASSIGNMENTS.items():
        for algorithm in algorithms:
            for seed in SEEDS:
                data = configuration(algorithm, seed, "full")
                data["run_id"] = f"tmrl-test-{CAMPAIGN_ID}-{algorithm}-s{seed}"
                data["training"]["total_transitions"] = transitions
                data["metadata"].update(stage="weeks", campaign=CAMPAIGN_ID)
                RunSpec.model_validate(data)
                path = directory / f"{algorithm}-s{seed}.yaml"
                configs.append((path, yaml.safe_dump(data, sort_keys=False)))
                jobs.append(
                    {
                        "worker": worker,
                        "algorithm": algorithm,
                        "seed": seed,
                        "run_id": data["run_id"],
                        "config": path.relative_to(ROOT).as_posix(),
                    }
                )
    if any((ROOT / "artifacts/tmrl-test-comparison" / job["run_id"]).exists() for job in jobs):
        raise RuntimeError("A campaign run already exists; never silently allocate replacement IDs")
    directory.mkdir(parents=True)
    for path, content in configs:
        path.write_text(content, encoding="utf-8")
    for job in jobs:
        job["config_sha256"] = digest(ROOT / job["config"])
    plan = {
        "schema_version": 1,
        "hash_mode": "sha256_lf_normalized_text_raw_binary",
        "campaign_id": CAMPAIGN_ID,
        "status": "PREPARED_NOT_LAUNCHED",
        "total_transitions_per_run": transitions,
        "seeds": list(SEEDS),
        "evaluation_trials": 30,
        "excluded_algorithms": ["sd-sac", "dsac", "discrete-sac"],
        "assignments": {key: list(value) for key, value in CAMPAIGN_ASSIGNMENTS.items()},
        "minimum_free_disk_gib": 100,
        "automatic_restart": False,
        "automatic_resume": False,
        "maximum_run_hours": 336,
        "shutdown_grace_seconds": 600,
        "jobs": jobs,
        "source_pins": source_pins(),
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def verify() -> dict[str, Any]:
    plan: dict[str, Any] = json.loads(PLAN.read_text(encoding="utf-8"))
    if (
        plan["campaign_id"] != CAMPAIGN_ID
        or plan["hash_mode"] != "sha256_lf_normalized_text_raw_binary"
        or plan["status"] != "PREPARED_NOT_LAUNCHED"
        or plan["seeds"] != list(SEEDS)
        or plan["evaluation_trials"] != 30
        or plan["assignments"] != {key: list(value) for key, value in CAMPAIGN_ASSIGNMENTS.items()}
        or plan["excluded_algorithms"] != ["sd-sac", "dsac", "discrete-sac"]
        or plan["automatic_restart"] is not False
        or plan["automatic_resume"] is not False
        or plan["maximum_run_hours"] != 336
        or plan["shutdown_grace_seconds"] != 600
        or plan["minimum_free_disk_gib"] != 100
        or plan["total_transitions_per_run"] < 2_048_000
        or plan["total_transitions_per_run"] % 2048
    ):
        raise ValueError("Campaign protocol/limits changed; prepare a new version instead")
    if plan["source_pins"] != source_pins():
        raise RuntimeError(
            "Frozen campaign source/assets/dependencies differ; do not launch or resume"
        )
    expected = {
        (worker, algorithm, seed)
        for worker, algorithms in CAMPAIGN_ASSIGNMENTS.items()
        for algorithm in algorithms
        for seed in SEEDS
    }
    found = set()
    ids = set()
    for job in plan["jobs"]:
        key = (job["worker"], job["algorithm"], job["seed"])
        if key in found or job["run_id"] in ids:
            raise ValueError("Duplicate campaign job or run ID")
        found.add(key)
        ids.add(job["run_id"])
        path = ROOT / job["config"]
        if path.parent.resolve() != (HERE / "configs/weeks").resolve():
            raise ValueError("Campaign config lies outside configs/weeks")
        if digest(path) != job["config_sha256"]:
            raise ValueError(f"Campaign configuration changed: {path.name}")
        spec = require_standard_config(path)
        if (
            spec.run_id != job["run_id"]
            or spec.seed != job["seed"]
            or spec.metadata.get("algorithm") != job["algorithm"]
            or spec.training.total_transitions != plan["total_transitions_per_run"]
            or spec.evaluation is None
            or spec.evaluation.trials_per_map != 30
            or spec.evaluation.maps[0].expected_map_uid != MAP_UID
        ):
            raise ValueError(f"Campaign job contract differs: {path.name}")
    if found != expected or len(ids) != 15:
        raise ValueError("Campaign must contain exactly five algorithms x three seeds")
    return plan


def remaining_jobs(plan: dict[str, Any], worker: str, start_at: str | None) -> list[dict[str, Any]]:
    jobs = [job for job in plan["jobs"] if job["worker"] == worker]
    if start_at is None:
        return jobs
    positions = [i for i, job in enumerate(jobs) if job["run_id"] == start_at]
    if len(positions) != 1:
        raise ValueError("--start-at must name a run assigned to this worker")
    # Manual continuation may skip only complete, measured runs, never failed prefixes.
    from experiments.tmrl_test_comparison.launch_checks import (
        LaunchRun,
        final_checkpoint,
        measurement_gate,
    )

    for job in jobs[: positions[0]]:
        run = LaunchRun.load(ROOT / job["config"])
        checkpoint = final_checkpoint(run)
        artifacts = (
            [run.directory / "evaluation.json"]
            if job["algorithm"] == "ppo"
            else list(run.directory.parent.glob(f"{run.spec.run_id}-benchmark-*/evaluation.json"))
        )
        if len(artifacts) != 1:
            raise RuntimeError(
                "Skipped run must have exactly one final evaluation; inspect manually"
            )
        measurement_gate(run, checkpoint, artifacts[0])
    return jobs[positions[0] :]


def stop_paths(worker: str) -> list[Path]:
    base = ROOT / "artifacts/tmrl-test-comparison"
    return [
        ROOT / "artifacts/STOP",
        base / "STOP",
        base / CAMPAIGN_ID / "STOP",
        base / CAMPAIGN_ID / f"STOP-{worker}",
    ] + [ROOT / "artifacts" / f"STOP-{algorithm}" for algorithm in CAMPAIGN_ASSIGNMENTS[worker]]


def check_script_policy() -> str:
    """Inspect Windows policy without changing it or executing a launcher script."""
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    try:
        policy = subprocess.check_output(
            [*command, "Get-ExecutionPolicy"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Cannot inspect PowerShell policy/modules on this host; launch blocked. "
            "Have the operator resolve the Windows setup; no policy was changed."
        ) from error
    if policy == "Restricted":
        raise RuntimeError(
            "Windows blocks PowerShell scripts on this host; use an operator-approved "
            "script policy/signing setup before launch. No policy was changed."
        )
    for name in ("run_campaign.ps1", "run_assigned.ps1"):
        path = HERE / name
        zone = Path(str(path) + ":Zone.Identifier")
        remote = zone.exists() and any(
            f"ZoneId={value}" in zone.read_text(encoding="utf-8") for value in (3, 4)
        )
        if policy == "AllSigned" or (policy == "RemoteSigned" and remote):
            quoted = str(path).replace("'", "''")
            signature = subprocess.check_output(
                [*command, f"(Get-AuthenticodeSignature -LiteralPath '{quoted}').Status"], text=True
            ).strip()
            if signature != "Valid":
                raise RuntimeError("Windows requires a valid script signature; launch blocked")
    if policy not in {"AllSigned", "RemoteSigned", "Unrestricted", "Bypass"}:
        raise RuntimeError(f"Unrecognized effective PowerShell policy: {policy}")
    return policy


def host_check(plan: dict[str, Any], worker: str) -> Path:
    """Read-only host checks plus a new receipt; no game/telemetry/controller calls."""
    if platform.system() != "Windows" or platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("Campaign requires the pinned Python 3.12 environment on Windows")
    active_stops = [str(path) for path in stop_paths(worker) if path.exists()]
    if active_stops:
        raise RuntimeError(f"STOP active; preserve it and resolve manually: {active_stops}")
    script_policy = check_script_policy()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError(
            "Use a clean frozen campaign checkout; do not train in a dirty workspace"
        )
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < plan["minimum_free_disk_gib"]:
        raise RuntimeError("Insufficient free disk; archive unrelated data before launching")
    import torch

    from trackmaniarl.observability.wandb_metrics import _load_wandb_key_from_dotenv
    from trackmaniarl.trackmania.steering_curve import SteeringCurve

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" or not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not inherit CPU-only audit environment variables")
    _load_wandb_key_from_dotenv(str(ROOT))
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Configure this host's private W&B key before training")
    if os.environ.get("WANDB_ENTITY") != "dsc-pjatk-warsaw":
        raise RuntimeError("Set WANDB_ENTITY=dsc-pjatk-warsaw for the agreed shared project")
    curve = ROOT / "artifacts/tmrl-test-comparison/steering-local.json"
    SteeringCurve.load(curve)
    receipt = {
        "status": "OFFLINE_HOST_CHECK_PASSED_GAME_PREFLIGHT_STILL_REQUIRED",
        "checked_at": datetime.now(UTC).isoformat(),
        "worker": worker,
        "hostname": platform.node(),
        "campaign_sha256": digest(PLAN),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "powershell_policy": script_policy,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "free_disk_gib": round(free, 2),
        "steering_sha256": digest(curve),
        "wandb_key_available": True,
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
        "controller_created": False,
    }
    path: Path = (
        ROOT
        / "artifacts/tmrl-test-comparison"
        / CAMPAIGN_ID
        / "hosts"
        / f"{worker}-{uuid4().hex}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(receipt, indent=2) + "\n")
    return path


def run_job(plan: dict[str, Any], job: dict[str, Any]) -> None:
    """Supervise one assigned launcher, bridging every STOP to its training stop file."""
    host_check(plan, job["worker"])
    directory = ROOT / "artifacts/tmrl-test-comparison" / CAMPAIGN_ID / job["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "run_id": job["run_id"],
        "status": "STARTING",
        "owned_identities": [],
        "stop_reason": None,
    }
    status = directory / "status.json"

    def save() -> None:
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(status)

    save()
    owned: dict[tuple[int, float], psutil.Process] = {}
    started = time.monotonic()
    stop_at = None
    primary_stop = ROOT / "artifacts" / f"STOP-{job['algorithm']}"
    with (directory / "launcher.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-File",
                str(HERE / "run_assigned.ps1"),
                job["algorithm"],
                "-Seeds",
                str(job["seed"]),
                "-ConfigDirectory",
                str(HERE / "configs/weeks"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        parent = psutil.Process(process.pid)
        owned[(parent.pid, parent.create_time())] = parent
        state["status"] = "RUNNING"
        state["owned_identities"] = [
            {"pid": pid, "create_time": created} for pid, created in sorted(owned)
        ]
        save()
        while process.poll() is None or any(child.is_running() for child in owned.values()):
            # Cached Process objects check creation time before signalling reused PIDs.
            for known in list(owned.values()):
                try:
                    if known.is_running():
                        for child in known.children(recursive=True):
                            owned[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
            elapsed = time.monotonic() - started
            stops = [str(path) for path in stop_paths(job["worker"]) if path.exists()]
            low_disk = shutil.disk_usage(ROOT).free < 10 * 1024**3
            orphaned = process.poll() is not None
            if stop_at is None and (
                stops or low_disk or orphaned or elapsed >= plan["maximum_run_hours"] * 3600
            ):
                stop_at = time.monotonic()
                state["status"] = "SAVING"
                state["stop_reason"] = (
                    "user_stop"
                    if stops
                    else "low_disk"
                    if low_disk
                    else "launcher_exited_with_descendants"
                    if orphaned
                    else "runtime_cap"
                )
                state["observed_stop_files"] = stops
                if not primary_stop.exists():
                    primary_stop.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with primary_stop.open("x", encoding="utf-8") as target:
                            target.write(f"{state['stop_reason']}: {job['run_id']}\n")
                    except FileExistsError:
                        pass
            state["owned_identities"] = [
                {"pid": pid, "create_time": created} for pid, created in sorted(owned)
            ]
            state["elapsed_seconds"] = round(elapsed, 2)
            save()
            if stop_at is not None and time.monotonic() - stop_at >= plan["shutdown_grace_seconds"]:
                state["status"] = "SHUTDOWN_TIMEOUT"
                for child in reversed(list(owned.values())):
                    try:
                        if child.is_running():
                            child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(2)
        alive = wait_for_owned_processes(owned.values(), timeout=30)
        state["surviving_identities"] = [
            {"pid": child.pid, "create_time": child.create_time()} for child in alive
        ]
        state["returncode"] = process.poll()
    state["ended_at"] = datetime.now(UTC).isoformat()
    if state["status"] != "SHUTDOWN_TIMEOUT":
        state["status"] = (
            "COMPLETE"
            if process.returncode == 0
            and not alive
            and stop_at is None
            and not any(path.exists() for path in stop_paths(job["worker"]))
            else "STOPPED_OR_FAILED"
        )
    save()
    if state["status"] != "COMPLETE":
        raise RuntimeError(f"{state['status']}; inspect {status}; no retry or next seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "host-check", "jobs", "config-check", "run")
    )
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--transitions", type=int, default=LONG_TRANSITIONS)
    parser.add_argument("--worker", choices=tuple(CAMPAIGN_ASSIGNMENTS))
    parser.add_argument("--start-at")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    plan = prepare(args.transitions) if args.command == "prepare" else verify()
    if args.command in {"host-check", "jobs", "run"} and args.worker is None:
        parser.error("--worker is required")
    if args.command == "host-check":
        print(host_check(plan, args.worker))
    elif args.command == "jobs":
        print(json.dumps(remaining_jobs(plan, args.worker, args.start_at)))
    elif args.command == "config-check":
        if args.config is None:
            parser.error("config is required")
        if not any(
            (ROOT / job["config"]).resolve() == args.config.resolve() for job in plan["jobs"]
        ):
            raise ValueError("Config is not a pinned campaign job")
    elif args.command == "run":
        matches = [
            job
            for job in plan["jobs"]
            if job["worker"] == args.worker and job["run_id"] == args.run_id
        ]
        if len(matches) != 1:
            parser.error("--run-id must name one assigned job")
        run_job(plan, matches[0])
    else:
        print(f"{plan['status']}: {len(plan['jobs'])} jobs, SD-SAC excluded; no game input")


if __name__ == "__main__":
    main()
