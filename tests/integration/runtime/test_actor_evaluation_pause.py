"""Losing a vision frame during in-training evaluation must not kill the actor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tests.integration.runtime.distributed_runtime_support import _Pipeline
from tests.integration.runtime.test_distributed_actor_rollouts import (
    _evaluation_actor,
    _EvaluationProbe,
)
from trackmaniarl.core.environment_errors import EnvironmentPausedError


class _FocusLossEnvironment:
    def __init__(self) -> None:
        self.steps = 0
        self.reset_seeds: list[int] = []

    def reset(self, *, seed: int) -> tuple[int, dict[str, Any]]:
        self.reset_seeds.append(seed)
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        if self.steps == 2:
            raise EnvironmentPausedError("game moved to background")
        finished = self.steps > 2
        return (
            self.steps,
            2.0,
            finished,
            False,
            {
                "progress_pct": 100.0 if finished else 23.0,
                "termination_reason": "finished" if finished else "",
                "race_time_ms": 12500.0 if finished else 50.0,
            },
        )


def test_in_training_evaluation_keeps_focus_loss_dnf_and_resumes_next_trial() -> None:
    probe = _EvaluationProbe()
    actor = _evaluation_actor(probe)
    actor.actor_id = "test-actor"
    actor.spec.evaluation = SimpleNamespace(trials_per_map=2)
    environment = _FocusLossEnvironment()

    actor._evaluate(environment, _Pipeline())

    assert not actor.stop.is_set()
    assert len(probe.requests) == 1
    assert probe.requests[0].transitions == []
    summaries = probe.requests[0].evaluations
    assert summaries is not None
    assert len(summaries) == 2
    assert summaries[0]["termination"] == "capture_interruption"
    assert summaries[0]["steps"] == 1
    assert summaries[0]["finished"] == 0.0
    assert summaries[0]["telemetry/error"] == 0.0
    assert summaries[0]["return"] == 2.0
    assert summaries[1]["termination"] == "finished"
    assert summaries[1]["finished"] == 1.0
    assert environment.reset_seeds == [1_000_007, 1_000_007, 1_000_008]
