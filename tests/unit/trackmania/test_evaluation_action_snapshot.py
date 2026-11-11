"""Evaluation diagnostics retain the policy action despite buffer reuse."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.distributed.actor_evaluation import EvaluatedAction, _evaluation_step
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator


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
