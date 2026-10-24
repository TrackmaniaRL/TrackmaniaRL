"""Wire the four-hour budget into the existing distributed learner lifecycle.

This does not launch actors, qualify an event source or declare campaign
cleanup/completion. The parent launcher must own those processes and check the
returned checkpoint proof. No watchdog is armed during construction/model setup.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from trackmaniarl.core.checkpoints import validate_checkpoint_v2
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.training_support import TrainingResult
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.experiments.full_qr.campaign_budget import (
    SUPPORTED_PILOT_SECONDS,
    BudgetSettings,
    CampaignBudget,
)
from trackmaniarl.experiments.full_qr.campaign_spec import (
    requested_pilot_clock,
    requested_pilot_seconds,
)


@dataclass(frozen=True, slots=True)
class PilotCoordinatorSettings:
    coordinator: CoordinatorConfig
    budget: BudgetSettings
    on_ready: Callable[[Mapping[str, Any]], None]

    def __post_init__(self) -> None:
        if self.coordinator.resume_checkpoint is not None or self.coordinator.demo_paths:
            raise ValueError("the full QR pilot must be fresh, without resume or demonstrations")
        signal = self.coordinator.external_stop
        if signal is None or not all(
            callable(getattr(signal, x, None)) for x in ("set", "is_set", "wait")
        ):
            raise ValueError("one owned shared actor/learner stop signal is required")
        if signal.is_set():
            raise ValueError("never reuse an already stopped campaign signal")
        if self.budget.duration_seconds not in SUPPORTED_PILOT_SECONDS:
            raise ValueError(
                "the production pilot requires exactly14400 or25200 wall-clock seconds"
            )


def verify_final_checkpoint(
    run: ResolvedRun,
    result: TrainingResult,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Read the actual atomic SDK checkpoint, checking stored final counters."""
    if (
        result.transitions <= 0
        or result.updates <= 0
        or result.episodes <= 0
        or not result.checkpoints
    ):
        raise ValueError("a final checkpoint with real transitions and updates is required")
    path = result.checkpoints[-1]
    directory = (run.run_dir / "checkpoints").resolve(strict=True)
    if path.is_symlink() or path.resolve(strict=True).parent != directory:
        raise ValueError("final checkpoint must belong to this campaign")
    state = run.checkpoint_codec.load(path)
    if (
        state.get("schema_version") != "2.0"
        or state.get("run_fingerprint") != expected_fingerprint
        or state.get("distributed", {}).get("transitions") != result.transitions
        or state.get("distributed", {}).get("updates") != result.updates
        or state.get("distributed", {}).get("episodes") != result.episodes
    ):
        raise ValueError("final checkpoint identity or stored counters differ")
    learner = state.get("learner")
    if not isinstance(learner, Mapping):
        raise ValueError("final checkpoint must contain the complete learner")
    validate_checkpoint_v2(learner)
    model = getattr(run.learner, "model", None)
    model_fingerprint = getattr(model, "architecture_fingerprint", None)
    if not callable(model_fingerprint):
        raise ValueError("final checkpoint requires the learner model architecture contract")
    if (
        learner["training"]["update_count"] != result.updates
        or learner["architecture_fingerprint"] != model_fingerprint()
    ):
        raise ValueError("final checkpoint learner architecture or updates differ")
    stack = [learner["online"], learner["target"], learner["optimizers"]]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (tuple, list)):
            stack.extend(value)
        elif isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite model or optimizer tensor in final checkpoint")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "transitions": result.transitions,
        "updates": result.updates,
        "episodes": result.episodes,
        "architecture_fingerprint": learner["architecture_fingerprint"],
    }


class BudgetedCoordinator(Coordinator):
    """Normal SDK learning/checkpoints, stopped by the same flag as the actors."""

    def __init__(self, run: ResolvedRun, pilot: PilotCoordinatorSettings) -> None:
        if run.environment_factory is None or not run.spec.training.save_final_checkpoint:
            raise ValueError("full environment and final checkpoint saving are required")
        if pilot.budget.duration_seconds != requested_pilot_seconds(run.spec):
            raise ValueError("the timer must match the pinned requested wall-clock duration")
        if pilot.budget.clock_source != requested_pilot_clock(run.spec):
            raise ValueError("the timer must match the pinned requested clock source")
        self.pilot = pilot
        self.budget = CampaignBudget(pilot.budget, stop_signal=pilot.coordinator.external_stop)
        self.final_checkpoint_proof: dict[str, Any] | None = None
        super().__init__(run, pilot.coordinator)

    def _start_server(self) -> None:
        # SDK calls this only after fresh model setup and initial publication.
        super()._start_server()
        state = self.budget.start()
        self.pilot.on_ready(
            {
                "pid": os.getpid(),
                "port": self._bound_port,
                "started_utc": state["started_utc"],
                "deadline_utc": state["deadline_utc"],
                "duration_seconds": state["duration_seconds"],
                "stop_file": str(self.budget.stop_file),
                "clock_source": state["clock_source"],
            }
        )

    def _external_stop_requested(self) -> bool:
        # Work-boundary check remains effective if the independent timer thread
        # or disk persistence is delayed. The watchdog also wakes actor waiters.
        stop = self.budget.stop_requested()
        if stop and self.budget.limit_reached():
            self.budget.request_stop("wall_clock_limit")
        return stop or super()._external_stop_requested()

    def run_forever(self) -> TrainingResult:
        try:
            result = super().run_forever()
            if not self.budget.stop_requested():
                raise RuntimeError("learner ended before the requested budget; pilot is incomplete")
            self.final_checkpoint_proof = verify_final_checkpoint(
                self.run, result, self.fingerprint
            )
            if self.budget.state.get("stop_reason") != "wall_clock_limit":
                raise RuntimeError(
                    "pilot stopped before the requested duration; checkpoint retained"
                )
            return result
        finally:
            if self.budget.state:
                self.budget.close()
