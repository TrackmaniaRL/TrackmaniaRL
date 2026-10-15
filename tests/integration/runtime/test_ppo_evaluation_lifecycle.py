from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests.integration.runtime.core_runtime_support import PpoFakeEnvironment
from tests.integration.runtime.test_trainer_runtime import _ppo_spec
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.training import Trainer


class _TrackedEnvironment(PpoFakeEnvironment):
    def __init__(self, phase: str, events: list[str]) -> None:
        super().__init__()
        self.phase = phase
        self.events = events
        self.closed = False
        self.steps = 0
        self.seeds: list[int | None] = []

    def reset(self, *, seed: int | None = None) -> tuple[torch.Tensor, dict[str, object]]:
        assert not self.closed
        self.seeds.append(seed)
        return super().reset(seed=seed)

    def step(self, action: object) -> tuple[torch.Tensor, float, bool, bool, dict[str, object]]:
        assert not self.closed, "A rollout reused its closed pre-evaluation environment"
        self.steps += 1
        return super().step(action)

    def close(self) -> None:
        assert not self.closed, "Environment closed twice"
        self.closed = True
        self.events.append(f"close:{self.phase}")


class _ExclusiveFactory:
    def __init__(self) -> None:
        self.phase = "training"
        self.events: list[str] = []
        self.environments: list[_TrackedEnvironment] = []

    def create(self, *, seed: int) -> _TrackedEnvironment:
        del seed
        assert all(environment.closed for environment in self.environments)
        self.events.append(f"create:{self.phase}")
        environment = _TrackedEnvironment(self.phase, self.events)
        self.environments.append(environment)
        return environment


class _ExclusiveEvaluator:
    def __init__(self, factory: _ExclusiveFactory) -> None:
        self.factory = factory

    def evaluate(self, policy: object) -> dict[str, float]:
        del policy
        self.factory.phase = "evaluation"
        environment = self.factory.create(seed=0)
        environment.close()
        self.factory.phase = "training"
        return {"eval/finish_rate": 1.0}


@pytest.mark.parametrize("interval", [None, 1], ids=["final", "periodic"])
def test_ppo_hands_environment_to_evaluation_and_restarts_fresh_rollouts(
    tmp_path: Path, interval: int | None
) -> None:
    spec = _ppo_spec(tmp_path)
    training = spec.training.model_copy(
        update={
            "total_transitions": 9,
            "sequence_length": 3,
            "batch_size": 1,
            "max_episode_steps": 5,
            "evaluate_every_episodes": interval,
        }
    )
    factory = _ExclusiveFactory()
    run = replace(
        resolve_run(spec.model_copy(update={"training": training})),
        environment_factory=factory,
        evaluator=_ExclusiveEvaluator(factory),
    )
    try:
        result = Trainer(run).train()
        state = run.checkpoint_codec.load(result.checkpoints[-1])
    finally:
        run.logger.close()

    repetitions = 1 if interval is None else 3
    assert (
        factory.events
        == ["create:training", "close:training", "create:evaluation", "close:evaluation"]
        * repetitions
    )
    training_environments = [item for item in factory.environments if item.phase == "training"]
    assert sum(item.steps for item in training_environments) == 9
    assert (result.transitions, result.updates) == (9, 3)
    reset_seeds = [seed for item in training_environments for seed in item.seeds]
    assert len(reset_seeds) == len(set(reset_seeds))
    assert state["counters"]["next_episode_index"] == (5 if interval is None else 6)
