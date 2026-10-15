"""Native Python host checks and bounded, identity-aware campaign supervision."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

from experiments.jakub_1212 import plan as protocol


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1)


def host_check(plan: dict[str, Any], worker: str) -> Path:
    if sys.platform not in {"win32", "linux"} or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use native Python3.12 on Windows or Linux X11")
    if sys.platform == "linux":
        from trackmaniarl.trackmania.linux_reset import check_session

        if "microsoft" in platform.release().lower():
            raise RuntimeError("WSL is not a supported game host; use native Linux desktop")
        check_session()
        if not any(os.access(p, os.R_OK | os.W_OK) for p in ("/dev/uinput", "/dev/input/uinput")):
            raise RuntimeError(
                "No user access to /dev/uinput; provision uinput normally, not sudo training"
            )
        if os.geteuid() == 0:
            raise RuntimeError("Do not run the trainer as root")
    stops = [str(p) for p in protocol.stop_paths(worker) if p.exists()]
    if stops:
        raise RuntimeError(f"STOP active; preserve and diagnose: {stops}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=protocol.ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("Use the same clean frozen Git checkout on every host")
    free = (
        shutil.disk_usage(
            protocol.BASE.parent if protocol.BASE.parent.exists() else protocol.ROOT
        ).free
        / 1024**3
    )
    if free < plan["minimum_free_disk_gib"]:
        raise RuntimeError("At least100GiB free is required before launch")
    import torch

    from trackmaniarl.observability.wandb_metrics import _load_wandb_key_from_dotenv
    from trackmaniarl.trackmania.steering_curve import SteeringCurve

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" or not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not use the CPU-only audit environment")
    _load_wandb_key_from_dotenv(os.environ.get("TRACKMANIA_CREDENTIALS_DIR", str(protocol.ROOT)))
    if not os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_ENTITY") != "dsc-pjatk-warsaw":
        raise RuntimeError("Configure your private W&B key and WANDB_ENTITY=dsc-pjatk-warsaw")
    curve = protocol.BASE / "steering-local.json"
    SteeringCurve.load(curve)
    plugin = Path.home() / "OpenplanetNext/Plugins/SAC_GetData.op"
    receipt = {
        "status": "HOST_CHECK_PASSED_GAME_PREFLIGHT_STILL_REQUIRED",
        "worker": worker,
        "checked_at": datetime.now(UTC).isoformat(),
        "system": platform.system(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=protocol.ROOT, text=True
        ).strip(),
        "campaign_sha256": hashlib.sha256(protocol.PLAN.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "library_version": __import__("trackmaniarl").__version__,
        "library_file": __import__("trackmaniarl").__file__,
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "plugin_sha256": hashlib.sha256(plugin.read_bytes()).hexdigest()
        if plugin.is_file()
        else None,
        "gpu": torch.cuda.get_device_name(0),
        "free_disk_gib": round(free, 2),
        "steering_sha256": hashlib.sha256(curve.read_bytes()).hexdigest(),
        "wandb_key_available": True,
        "controller_created": False,
        "powershell_required": False,
    }
    path = protocol.BASE / protocol.CAMPAIGN_ID / "hosts" / f"{worker}-{uuid4().hex}.json"
    save_json(path, receipt)
    return path


def assert_idle() -> None:
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if process.pid != os.getpid() and re.search(
                r"(?:trackmaniarl|tmrl_test_comparison[.]launch_checks)\s+"
                r"(?:train|resume|benchmark|actor|learner)\b",
                command,
            ):
                raise RuntimeError(
                    f"Another training/evaluation process is active: PID{process.pid}"
                )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue


def alive(process: psutil.Process) -> bool:
    try:
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def capture(owned: dict[tuple[int, float], psutil.Process]) -> None:
    for process in list(owned.values()):
        if not alive(process):
            continue
        try:
            for child in process.children(recursive=True):
                owned[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            continue


def close_owned(owned: dict[tuple[int, float], psutil.Process]) -> list[dict[str, Any]]:
    capture(owned)
    for method in ("terminate", "kill"):
        for process in reversed(list(owned.values())):
            try:
                if alive(process):
                    getattr(process, method)()
            except psutil.NoSuchProcess:
                continue
        psutil.wait_procs([p for p in owned.values() if alive(p)], timeout=5)
    return [{"pid": p.pid, "create_time": created} for (_, created), p in owned.items() if alive(p)]


def request_stop(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(reason + "\n")
    except FileExistsError:
        pass


def watch_stage(command: list[str], context: dict[str, Any]) -> None:
    state, owned = context["state"], context["owned"]
    assert_idle()
    if time.monotonic() >= context["save_at"]:
        raise RuntimeError("Save deadline reached before next stage")
    if any(p.exists() for p in protocol.stop_paths(context["worker"])):
        raise RuntimeError("STOP before next stage")
    logpath = context["directory"] / f"{state['stage']}.log"
    options: dict[str, Any] = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if sys.platform == "linux":
        options["start_new_session"] = True
    with logpath.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            command, cwd=protocol.ROOT, stdout=log, stderr=subprocess.STDOUT, **options
        )
        process = psutil.Process(child.pid)
        owned[(process.pid, process.create_time())] = process
        exited_at = None
        while True:
            try:
                capture(owned)
                if time.monotonic() - context.get("last_pin_check", 0.0) >= 60:
                    protocol.verify()
                    plugin = Path.home() / "OpenplanetNext/Plugins/SAC_GetData.op"
                    if context.get("plugin_sha256") and (
                        not plugin.is_file()
                        or hashlib.sha256(plugin.read_bytes()).hexdigest()
                        != context["plugin_sha256"]
                    ):
                        raise RuntimeError("Plugin changed during the frozen run")
                    context["last_pin_check"] = time.monotonic()
                guard = context.get("guard")
                if guard is not None and guard.poll() is not None:
                    raise RuntimeError("Independent deadline guard exited during an active stage")
                code = child.poll()
                live = [p for p in owned.values() if alive(p)]
                now = time.monotonic()
                if code is not None and not live:
                    if code or context["stop_at"] is not None:
                        raise RuntimeError(f"Stage{state['stage']} failed/stopped: exit{code}")
                    return
                if code is not None and exited_at is None:
                    exited_at = now
                stops = [str(p) for p in protocol.stop_paths(context["worker"]) if p.exists()]
                reason = (
                    "user_stop"
                    if stops
                    else "low_disk"
                    if shutil.disk_usage(protocol.ROOT).free < 10 * 1024**3
                    else "runtime_cap"
                    if now >= context["save_at"]
                    else "launcher_exited_with_descendants"
                    if exited_at is not None and now - exited_at >= 5
                    else None
                )
                if reason and context["stop_at"] is None:
                    context["stop_at"] = now
                    state.update(status="SAVING", stop_reason=reason, observed_stop_files=stops)
                    request_stop(context["stop"], reason)
                state["owned_identities"] = [
                    {"pid": pid, "create_time": created} for pid, created in owned
                ]
                state["elapsed_seconds"] = round(now - context["started"], 2)
                save_json(context["status"], state)
                if now >= context["hard_at"] or (
                    context["stop_at"] is not None and now - context["stop_at"] >= context["grace"]
                ):
                    survivors = close_owned(owned)
                    child.wait(timeout=5)
                    state["surviving_identities"] = survivors
                    raise RuntimeError("Bounded shutdown; no retry or next seed")
                time.sleep(1)
            except KeyboardInterrupt:
                if context["stop_at"] is None:
                    context["stop_at"] = time.monotonic()
                    state.update(status="SAVING", stop_reason="operator_interrupt")
                    request_stop(context["stop"], "operator_interrupt")


def shutdown(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Allow a bounded STOP save even when the supervisor encounters an exception."""
    deadline = min(context["hard_at"], time.monotonic() + context["grace"])
    while time.monotonic() < deadline:
        capture(context["owned"])
        if not any(alive(p) for p in context["owned"].values()):
            return []
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    return close_owned(context["owned"])


def run_job(plan: dict[str, Any], job: dict[str, Any]) -> None:
    host_receipt = host_check(plan, job["worker"])
    host = json.loads(host_receipt.read_text(encoding="utf-8"))
    assert_idle()
    config = protocol.ROOT / job["config"]
    from experiments.jakub_1212.launch_checks import LaunchRun, final_checkpoint

    run = LaunchRun.load(config)
    if run.directory.exists():
        raise RuntimeError("Run already exists; never restart or resume this queue")
    directory = protocol.BASE / protocol.CAMPAIGN_ID / job["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    grace = plan["shutdown_grace_seconds"]
    state: dict[str, Any] = {
        "status": "RUNNING",
        "run_id": job["run_id"],
        "started_at": datetime.now(UTC).isoformat(),
        "stage": "preflight",
        "stop_reason": None,
        "supervisor": {"pid": os.getpid(), "create_time": psutil.Process().create_time()},
    }
    context: dict[str, Any] = {
        "state": state,
        "plugin_sha256": host["plugin_sha256"],
        "owned": {},
        "directory": directory,
        "status": directory / "status.json",
        "worker": job["worker"],
        "started": started,
        "grace": grace,
        "stop_at": None,
        "save_at": started + plan["maximum_run_hours"] * 3600 - grace,
        "hard_at": started + plan["maximum_run_hours"] * 3600,
        "stop": protocol.ROOT / "artifacts" / f"STOP-{job['algorithm']}",
    }
    python = [sys.executable, "-m"]
    stops = [
        argument
        for path in protocol.stop_paths(job["worker"])
        for argument in ("--stop-file", str(path))
    ]
    try:
        save_json(
            directory / "launch.json",
            {"started_at": state["started_at"], "launcher": state["supervisor"]},
        )
        save_json(
            directory / "plan.json",
            {
                "source_root": str(protocol.ROOT),
                "maximum_seconds": plan["maximum_run_hours"] * 3600,
                "save_margin_seconds": grace,
                "stop_paths": [str(p) for p in protocol.stop_paths(job["worker"])],
            },
        )
        save_json(context["status"], state)
        with (directory / "guard.log").open("x", encoding="utf-8") as guard_log:
            guard = subprocess.Popen(
                [*python, "experiments.jakub_1212.deadline_guard", str(directory)],
                cwd=protocol.ROOT,
                stdout=guard_log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        context["guard"] = guard
        ready = directory / "deadline-guard-ready.json"
        for _ in range(100):
            if ready.exists():
                break
            if guard.poll() is not None:
                raise RuntimeError("Deadline guard failed before readiness")
            time.sleep(0.1)
        if not ready.exists():
            request_stop(context["stop"], "deadline_guard_not_ready")
            raise RuntimeError("Deadline guard did not confirm readiness")
        state["independent_guard"] = json.loads(ready.read_text(encoding="utf-8"))
        state["independent_guard_forwarding"] = {
            "pid": guard.pid,
            "create_time": psutil.Process(guard.pid).create_time(),
        }
        watch_stage(
            [
                *python,
                "experiments.jakub_1212.preflight",
                str(config),
                "--require-new-run",
                "--receipt",
                str(directory / "preflight.json"),
                *stops,
            ],
            context,
        )
        state["stage"] = "train"
        watch_stage(
            [*python, "trackmaniarl", "train", str(config), "--stop-file", str(context["stop"])],
            context,
        )
        import torch

        torch.set_num_threads(2)
        checkpoint = final_checkpoint(run)
        if any(p.exists() for p in protocol.stop_paths(job["worker"])):
            raise RuntimeError("STOP before final evaluation")
        state["stage"] = "evaluation"
        kind = "evaluation" if job["algorithm"] == "ppo" else "benchmark"
        watch_stage(
            [
                *python,
                "experiments.jakub_1212.launch_checks",
                kind,
                str(config),
                str(checkpoint),
                "--stop-file",
                str(context["stop"]),
            ],
            context,
        )
        with checkpoint.open("rb") as stream:
            checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        state.update(
            status="COMPLETE", checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha
        )
    except BaseException as error:
        request_stop(context["stop"], f"campaign_failure:{type(error).__name__}")
        state.update(
            status="STOPPED_OR_FAILED",
            error_type=type(error).__name__,
            error=str(error),
            surviving_identities=shutdown(context),
        )
        raise
    finally:
        state.update(
            ended_at=datetime.now(UTC).isoformat(),
            owned_identities=[{"pid": p, "create_time": c} for p, c in context["owned"]],
        )
        save_json(context["status"], state)
