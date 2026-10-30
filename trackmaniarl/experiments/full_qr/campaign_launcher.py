"""Parent ownership for the fresh full QR learner and actor processes.

The learner arms the existing 14400-second budget after model/server setup.
This parent never supplies an environment or substitutes missing reward events.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from trackmaniarl.experiments.full_qr.campaign_budget import PILOT_SECONDS, SUPPORTED_PILOT_SECONDS
from trackmaniarl.experiments.full_qr.campaign_policy import (
    campaign_storage_reserves,
    clock_seconds,
    requested_pilot_clock,
    requested_pilot_seconds,
    validate_clock_source,
)
from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
from trackmaniarl.experiments.full_qr.campaign_signal import MappedStopSignal
from trackmaniarl.experiments.full_qr.campaign_worker import WorkerSettings, publish_once
from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os


@dataclass(frozen=True, slots=True)
class LaunchSettings:
    worker: WorkerSettings
    learner_python: Path
    actor_python: Path
    actors: int
    startup_seconds: float = 120.0

    def preflight(self) -> None:
        """Fail before spawning workers or resolving a model."""
        if sys.platform != "linux":
            raise RuntimeError("full headless parent requires Linux")
        spec = self.worker.validate()
        if self.worker.worker_name != "learner" or self.worker.endpoint != "127.0.0.1:0":
            raise ValueError("fresh learner with an OS-assigned local port required")
        if type(self.actors) is not int or not 1 <= self.actors <= 64:
            raise ValueError("measured actor count in [1,64] required")
        if not math.isfinite(self.startup_seconds) or not 0 < self.startup_seconds <= 300:
            raise ValueError("bounded worker startup timeout required")
        for executable in (self.learner_python, self.actor_python):
            if (
                not executable.is_absolute()
                or not executable.is_file()
                or not os.access(executable, os.X_OK)
            ):
                raise ValueError("explicit executable learner/actor interpreters required")
        if self.worker.token_path.stat().st_mode & 0o077:
            raise ValueError("private campaign token required")
        self.worker.token()
        if (self.worker.artifact_root / spec.run_id).exists():
            raise ValueError("never reuse an existing experiment")
        _storage(self.worker.directory, self.worker.artifact_root)


def _storage(directory: Path, artifact_root: Path) -> None:
    for reserve in campaign_storage_reserves(directory, artifact_root):
        if shutil.disk_usage(reserve.path).free < reserve.minimum_free_bytes:
            raise RuntimeError(f"campaign storage reserve exhausted: {reserve.path}")


def _request(settings: WorkerSettings) -> tuple[Path, str]:
    path = settings.directory / f"{settings.worker_name}.json"
    raw = json.dumps({key: str(value) for key, value in asdict(settings).items()}).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | linux_os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return path, hashlib.sha256(raw).hexdigest()


def _command(executable: Path, settings: WorkerSettings, role: str) -> list[str]:
    request, pin = _request(settings)
    return [
        str(executable),
        "-m",
        "trackmaniarl.experiments.full_qr.campaign_entry",
        "--role",
        role,
        "--settings",
        str(request),
        "--sha256",
        pin,
    ]


def _read_ready(
    path: Path, identity: dict[str, Any], duration_seconds: int = PILOT_SECONDS
) -> dict[str, Any]:
    ready: dict[str, Any] = json.loads(path.read_text())
    if (
        ready.get("pid") != identity["worker"]["pid"]
        or type(ready.get("port")) is not int
        or not 1 <= ready["port"] <= 65535
        or duration_seconds not in SUPPORTED_PILOT_SECONDS
        or ready.get("duration_seconds") != duration_seconds
    ):
        raise ValueError("learner ready identity, bound port or duration differs")
    return ready


def _budget_deadline(  # noqa: PLR0913 - independent persisted state, readiness and pinned timing contract.
    budget: dict[str, Any],
    ready: dict[str, Any],
    *,
    duration_seconds: int,
    clock_source: str,
    startup_seconds: float,
) -> float:
    started = budget.get("started_monotonic")
    if type(started) not in (int, float):
        raise ValueError("persisted actual learner budget differs from readiness")
    assert isinstance(started, (int, float))
    if (
        not math.isfinite(started)
        or not 0 <= clock_seconds(clock_source) - started < startup_seconds
        or budget.get("clock_source", "monotonic") != clock_source
        or ready.get("clock_source", "monotonic") != clock_source
        or budget.get("duration_seconds") != duration_seconds
        or budget.get("started_utc") != ready.get("started_utc")
        or budget.get("deadline_utc") != ready.get("deadline_utc")
    ):
        raise ValueError("persisted actual learner budget differs from readiness")
    return float(started + duration_seconds)


def verify_campaign_result(  # noqa: PLR0913 - independent result, closure, process exits and actor count.
    result: dict[str, Any],
    closure: dict[str, Any],
    exits: dict[str, int],
    actors: int,
    *,
    duration_seconds: int = PILOT_SECONDS,
    clock_source: str = "monotonic",
) -> None:
    """No success from a timer alone, a failed actor, or a missing checkpoint."""
    elapsed = result.get("budget", {}).get("elapsed_seconds")
    validate_clock_source(clock_source)
    if (
        result.get("schema") != "tmrl.full-qr-learner-result/v1"
        or result.get("budget", {}).get("stop_reason") != "wall_clock_limit"
        or type(duration_seconds) is not int
        or duration_seconds not in SUPPORTED_PILOT_SECONDS
        or result.get("campaign_mode", "qualification")
        != ("diagnostic" if duration_seconds == 1800 else "qualification")
        or result.get("budget", {}).get("duration_seconds") != duration_seconds
        or result.get("budget", {}).get("clock_source", "monotonic") != clock_source
        or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed < duration_seconds
        or closure.get("remainingOwned") != []
        or closure.get("cleanupErrors") != []
        or closure.get("forced") != []
        or set(exits) != {"learner", *(f"actor-{i:03d}" for i in range(actors))}
        or any(code != 0 for code in exits.values())
    ):
        raise ValueError(
            "diagnostic or four-hour/seven-hour worker completion and clean owned closure required"
        )
    checkpoint = result.get("checkpoint", {})
    for name in ("transitions", "updates", "episodes"):
        value = result.get(name)
        if type(value) is not int or value <= 0 or checkpoint.get(name) != value:
            raise ValueError("positive final checkpoint counters must equal learner counters")
    path = Path(checkpoint["path"])
    if not path.is_file() or path.is_symlink() or path.stat().st_size != checkpoint.get("bytes"):
        raise ValueError("final checkpoint file is missing or changed")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    if digest.hexdigest() != checkpoint.get("sha256"):
        raise ValueError("final checkpoint changed after learner verification")


def launch(settings: LaunchSettings, *, source_root: Path) -> dict[str, Any]:
    """Run serial setup, independent worker supervision and bounded own cleanup."""
    settings.preflight()
    if (
        not source_root.is_absolute()
        or source_root.resolve(strict=True) != source_root
        or str(source_root).startswith("/mnt/")
        or not (source_root / "trackmaniarl/experiments/full_qr/campaign_entry.py").is_file()
    ):
        raise ValueError("canonical separately deployed ext4 worker source required")
    worker = settings.worker
    signal = MappedStopSignal(worker.stop_path)
    cohort = None
    report: dict[str, Any] = {"schema": "tmrl.full-qr-parent/v1", "phase": "preparation"}
    failure = None
    try:
        if signal.is_set():
            raise ValueError("never reuse an already stopped pilot")
        cohort = CampaignProcessCohort(worker.directory / "processes", signal.set)
        identity = cohort.start(
            "learner", _command(settings.learner_python, worker, "learner"), cwd=source_root
        )
        report["processes"] = [identity]
        ready_path = worker.directory / "learner/ready.json"
        startup = time.monotonic() + settings.startup_seconds
        while not ready_path.exists():
            if cohort.exited() or signal.is_set():
                raise RuntimeError("learner exited before readiness")
            if time.monotonic() >= startup:
                raise TimeoutError("learner startup deadline")
            _storage(worker.directory, worker.artifact_root)
            time.sleep(0.1)
        duration_seconds = requested_pilot_seconds(worker.validate())
        report["campaign_mode"] = "diagnostic" if duration_seconds == 1800 else "qualification"
        ready = _read_ready(ready_path, identity, duration_seconds)
        report.update(phase="training", ready=ready)
        budget = json.loads((worker.directory / "budget/budget-state.json").read_text())
        clock_source = requested_pilot_clock(worker.validate())
        hard_deadline = _budget_deadline(
            budget,
            ready,
            duration_seconds=duration_seconds,
            clock_source=clock_source,
            startup_seconds=settings.startup_seconds,
        )
        endpoint = f"127.0.0.1:{ready['port']}"
        for index in range(settings.actors):
            actor = replace(worker, worker_name=f"actor-{index:03d}", endpoint=endpoint)
            report["processes"].append(
                cohort.start(
                    actor.worker_name,
                    _command(settings.actor_python, actor, "actor"),
                    cwd=source_root,
                )
            )
        publish_once(worker.directory / "parent-ready.json", report)
        next_storage = 0.0
        while not signal.is_set():
            if cohort.exited():
                raise RuntimeError("a pilot worker exited before the shared stop")
            if time.monotonic() >= next_storage:
                _storage(worker.directory, worker.artifact_root)
                next_storage = time.monotonic() + 1
            if clock_seconds(clock_source) >= hard_deadline:
                # The learner independently marks wall_clock_limit and saves.
                signal.set()
                break
            time.sleep(0.1)
    except BaseException as error:
        failure = error
        report.update(phase="failed", error=repr(error))
    finally:
        try:
            if cohort is not None:
                report["closure"] = cohort.close(grace_seconds=120)
                # close() has closed every control socket. Read the verified
                # supervisor receipts, rather than polling closed descriptors.
                report["exits"] = {
                    name: child["returncode"]
                    for name, child in report["closure"]["children"].items()
                }
        finally:
            signal.close()
    if failure is None:
        try:
            result = json.loads((worker.directory / "learner/result.json").read_text())
            spec = worker.validate()
            checkpoint_root = worker.artifact_root / spec.run_id / "checkpoints"
            if (
                result.get("pid") != identity["worker"]["pid"]
                or Path(result.get("checkpoint", {}).get("path", "")).parent != checkpoint_root
            ):
                raise ValueError("final learner identity or campaign checkpoint directory differs")
            verify_campaign_result(
                result,
                report["closure"],
                report["exits"],
                settings.actors,
                duration_seconds=requested_pilot_seconds(spec),
                clock_source=requested_pilot_clock(spec),
            )
            report.update(phase="completed", result=result)
        except BaseException as error:
            failure = error
            report.update(phase="failed", error=repr(error))
    publish_once(worker.directory / "parent-result.json", report)
    if failure is not None:
        raise RuntimeError(
            "full QR pilot failed; owned closure recorded in parent-result.json"
        ) from failure
    return report
