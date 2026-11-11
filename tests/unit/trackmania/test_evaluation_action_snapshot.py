"""Evaluation diagnostics retain the policy action despite buffer reuse."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.distributed.actor_evaluation import (
    EvaluatedAction,
    _evaluation_action,
    _evaluation_step,
)
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator
from trackmaniarl.trackmania.human_recovery_recording import _policy_step


def test_local_and_distributed_evaluation_honor_mode_without_mutating_exploration() -> None:
    class Policy:
        def __init__(self) -> None:
            self.epsilon = 0.8
            self.modes: list[PolicyMode] = []

        def set_exploration_epsilon(self, value: float) -> None:
            self.epsilon = value

        def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
            self.modes.append(mode)
            return 1 if mode is PolicyMode.EVALUATION else 0

    class Environment:
        def step(self, action: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 0, 0.0, False, False, {}

    policy = Policy()
    context = SimpleNamespace(policy=policy, prepared=0, environment=Environment())
    for _ in range(2):
        local = TrackmaniaEvaluator._take_step(None, context).action  # type: ignore[arg-type]
        remote = _evaluation_action(context).value
        assert local == remote == 1
        assert policy.epsilon == 0.8
    assert policy.modes == [PolicyMode.EVALUATION] * 4


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_evaluation_snapshots_action_before_environment_mutation(backend: str) -> None:
    action = np.array([0.25], dtype=np.float32) if backend == "numpy" else torch.tensor([0.25])

    class Policy:
        def act(self, observation: Any, mode: Any) -> Any:
            return action

    class Environment:
        def step(self, issued: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
            issued[...] = 0.75
            return 0, 0.0, False, False, {}

    loop = SimpleNamespace(policy=Policy(), prepared=0, environment=Environment())
    result = TrackmaniaEvaluator._take_step(None, loop)  # type: ignore[arg-type]
    assert float(result.action[0]) == 0.25
    assert float(action[0]) == 0.75
    action[...] = 1.0
    assert float(result.action[0]) == 0.25


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_distributed_evaluation_snapshots_action_before_environment_mutation(backend: str) -> None:
    action = np.array([0.25], dtype=np.float32) if backend == "numpy" else torch.tensor([0.25])

    class Environment:
        def step(self, issued: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
            issued[...] = 0.75
            return 0, 0.0, False, False, {}

    context = SimpleNamespace(environment=Environment(), runtime=SimpleNamespace(stop=Event()))
    result = _evaluation_step(context, 0, EvaluatedAction(action, 0.0))
    assert float(result.action[0]) == 0.25
    assert float(action[0]) == 0.75
    action[...] = 1.0
    assert float(result.action[0]) == 0.25


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("collector_kind", ["local", "distributed"])
def test_evaluation_policy_cannot_mutate_prepared_observation(
    backend: str, collector_kind: str
) -> None:
    observation = np.array([1.0], dtype=np.float32) if backend == "numpy" else torch.tensor([1.0])

    class Policy:
        def act(self, value: Any, mode: Any) -> Any:
            value[...] += 10
            return value

    class Environment:
        def step(self, action: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 0, 0.0, False, False, {}

    context = SimpleNamespace(policy=Policy(), prepared=observation, environment=Environment())
    if collector_kind == "distributed":
        action = _evaluation_action(context).value
    else:
        action = TrackmaniaEvaluator._take_step(None, context).action  # type: ignore[arg-type]
    assert float(action[0]) == 11.0
    assert float(observation[0]) == 1.0


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_recovery_policy_does_not_mutate_recorded_previous_frame(backend: str) -> None:
    observation = np.zeros(33, dtype=np.float32) if backend == "numpy" else torch.zeros(33)

    class Policy:
        def act(self, value: Any, mode: Any) -> int:
            value[...] += 10
            return 0

    class Environment:
        def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            return np.zeros(33, dtype=np.float32), 0.0, False, False, {}

    request = SimpleNamespace(policy=Policy(), environment=Environment())
    _policy_step(request, observation)
    assert float(observation[0]) == 0.0
