"""SDK worker lifecycle and progress reporting for full-QR campaigns.

Nothing starts on import. The parent owns qualification, the immutable launch
specification, worker leases and final cleanup. These entries never resume a
checkpoint or supply missing environment events.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import run_fingerprint
from trackmaniarl.experiments.full_qr.campaign_budget import BudgetSettings
from trackmaniarl.experiments.full_qr.campaign_coordinator import (
    BudgetedCoordinator,
    PilotCoordinatorSettings,
)
from trackmaniarl.experiments.full_qr.campaign_policy import (
    campaign_storage_reserves,
    for_wall_clock_budget,
    requested_pilot_clock,
    requested_pilot_seconds,
)
from trackmaniarl.experiments.full_qr.campaign_signal import MappedStopSignal


class WorkerProgress:
    def __init__(  # noqa: PLR0913 - independent coordinator, destination, stop and diagnostic cadence.
        self, coordinator: Any, path: Path, stop: Any, interval: float = 1.0
    ) -> None:
        self.coordinator, self.path, self.stop = coordinator, path, stop
        self.interval = interval
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, name="qr-worker-progress", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.done.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("worker counter reporter did not close")

    def _watch(self) -> None:
        previous, changed, dumped = -1, monotonic(), False
        try:
            while not self.done.wait(self.interval):
                counters = self.coordinator.counters
                now = monotonic()
                if counters.updates != previous:
                    previous, changed, dumped = counters.updates, now, False
                elif counters.updates > 0 and now - changed > 10 and not dumped:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    dumped = True
                row = {
                    "pid": os.getpid(),
                    "monotonic": now,
                    "transitions": counters.transitions,
                    "updates": counters.updates,
                    "episodes": counters.episodes,
                    "update_credit": counters.update_credit,
                }
                temporary = self.path.with_suffix(".stage")
                temporary.write_text(json.dumps(row, allow_nan=False))
                for attempt in range(10):
                    try:
                        os.replace(temporary, self.path)
                        break
                    except PermissionError:
                        # Windows readers may briefly deny replacement; ext4
                        # does not. Never turn an unbounded I/O failure into a
                        # silently stale heartbeat.
                        if attempt == 9:
                            raise
                        if self.done.wait(0.01):
                            temporary.unlink(missing_ok=True)
                            return
        except BaseException:
            self.stop.set()
            raise


def publish_once(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish one fresh ready/result; never replace a prior worker."""
    stage = path.with_name(path.name + ".stage")
    with stage.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(stage, path)
    stage.unlink()


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    directory: Path
    config_path: Path
    config_sha256: str
    token_path: Path
    stop_path: Path
    endpoint: str
    worker_name: str
    artifact_root: Path

    def validate(self) -> RunSpec:
        root = self.directory
        if root.resolve(strict=True) != root or str(root).startswith("/mnt/"):
            raise ValueError("canonical private ext4 campaign directory required")
        for path in (self.config_path, self.token_path, self.stop_path):
            if path.resolve(strict=True) != path or path.parent != root:
                raise ValueError("only canonical files from this campaign may be used")
        if hashlib.sha256(self.config_path.read_bytes()).hexdigest() != self.config_sha256:
            raise ValueError("immutable campaign specification changed")
        if (
            not self.endpoint.startswith("127.0.0.1:")
            or not self.endpoint[10:].isdigit()
            or not 0 <= int(self.endpoint[10:]) <= 65535
        ):
            raise ValueError("explicit local learner endpoint required")
        if (
            not self.worker_name
            or self.worker_name in (".", "..")
            or any(x not in "abcdefghijklmnopqrstuvwxyz0123456789-" for x in self.worker_name)
        ):
            raise ValueError("simple owned worker name required")
        spec = RunSpec.from_yaml(self.config_path)
        if for_wall_clock_budget(spec) != spec:
            raise ValueError("the worker requires the reviewed wall-clock transition cap")
        if (
            not self.artifact_root.is_absolute()
            or self.artifact_root.name != "artifacts"
            or self.artifact_root.resolve() != self.artifact_root
            or spec.artifacts_dir != self.artifact_root
        ):
            raise ValueError("only the new campaign's explicitly bound artifact root is allowed")
        if (
            spec.components.environment is None
            or spec.components.environment.class_path
            != "trackmaniarl.trackmania.native_headless.full_qr.factory:FullQrEnvironmentFactory"
            or spec.components.learner.kwargs.get("model_initialization_checkpoint") is not None
            or spec.metadata.get("resume_checkpoint") is not None
            or spec.training.offline_pretrain_updates
            or not spec.training.save_final_checkpoint
        ):
            raise ValueError("fresh full native QR configuration with supported budget required")
        return spec

    def token(self) -> str:
        value = self.token_path.read_text(encoding="utf-8").strip()
        if len(value) < 32:
            raise ValueError("private campaign token is too short")
        return value


def run_learner_worker(settings: WorkerSettings) -> None:
    spec = settings.validate()
    if (settings.artifact_root / spec.run_id).exists():
        raise ValueError("never start a fresh learner in an existing run directory")
    output = settings.directory / settings.worker_name
    output.mkdir(exist_ok=False)
    signal = MappedStopSignal(settings.stop_path)
    run = None
    progress = None
    try:
        run = resolve_run(spec, base_dir=settings.config_path.parent)
        coordinator = BudgetedCoordinator(
            run,
            PilotCoordinatorSettings(
                CoordinatorConfig(
                    bind=settings.endpoint,
                    token=settings.token(),
                    fingerprint=run_fingerprint(spec, settings.config_path.parent),
                    external_stop=signal,
                ),
                BudgetSettings(
                    settings.directory / "budget",
                    campaign_storage_reserves(settings.directory, settings.artifact_root),
                    duration_seconds=requested_pilot_seconds(spec),
                    clock_source=requested_pilot_clock(spec),
                ),
                lambda ready: publish_once(output / "ready.json", dict(ready)),
            ),
        )
        progress = WorkerProgress(coordinator, output / "progress.json", signal)
        progress.start()
        result = coordinator.run_forever()
        publish_once(
            output / "result.json",
            {
                "schema": "tmrl.full-qr-learner-result/v1",
                "pid": os.getpid(),
                "budget": coordinator.budget.state,
                "checkpoint": coordinator.final_checkpoint_proof,
                "transitions": result.transitions,
                "updates": result.updates,
                "episodes": result.episodes,
                "process_cleanup_verified": False,
            },
        )
    finally:
        if progress is not None:
            progress.close()
        signal.set()
        signal.close()
        if run is not None:
            run.logger.close()


def run_actor_worker(settings: WorkerSettings) -> None:
    settings.validate()
    output = settings.directory / settings.worker_name
    output.mkdir(exist_ok=False)
    signal = MappedStopSignal(settings.stop_path)
    try:
        actor = ActorRuntime(
            ActorProcessRequest(
                str(settings.config_path),
                settings.endpoint,
                settings.worker_name,
                settings.token(),
                signal,
            )
        )
        actor.run_forever()
        publish_once(
            output / "result.json",
            {
                "schema": "tmrl.full-qr-actor-result/v1",
                "pid": os.getpid(),
                "stop_reason": actor.stop_reason,
                "shared_stop": signal.is_set(),
                "process_cleanup_verified": False,
            },
        )
    finally:
        # An actor exiting early invalidates the whole pilot; notify all peers.
        signal.set()
        signal.close()
