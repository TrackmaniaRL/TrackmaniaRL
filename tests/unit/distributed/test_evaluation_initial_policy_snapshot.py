from __future__ import annotations

from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from trackmaniarl.distributed import actor_evaluation
from trackmaniarl.distributed.codec import WireCodec


@pytest.mark.parametrize("assignment_id", ["assignment", ""])
def test_evaluation_snapshot_is_captured_before_warmup_and_trials(
    monkeypatch: Any, assignment_id: str
) -> None:
    class StatefulPolicy:
        def __init__(self) -> None:
            self.state = torch.tensor([3.0])
            self.epsilon = 0.4

        def export_state(self) -> dict[str, Any]:
            return {"state": self.state, "epsilon": self.epsilon}

        def load_state(self, state: dict[str, Any]) -> None:
            self.state = state["state"]
            self.epsilon = state["epsilon"]

        def set_exploration_epsilon(self, epsilon: float) -> None:
            self.epsilon = epsilon

        def act(self) -> None:
            self.state.add_(1)

    policy = StatefulPolicy()
    requests: list[Any] = []
    codec = WireCodec(1_000_000)
    runtime = SimpleNamespace(
        spec=SimpleNamespace(evaluation=None),
        codec=codec,
        _evaluation_request_lock=Lock(),
        _evaluation_request=(codec.encode(policy.export_state()), 7, assignment_id),
        _new_policy=lambda: policy,
        _active_evaluation_assignment_id="",
        _active_evaluation_initial_snapshot=b"",
        _spool=requests.append,
    )
    runtime._evaluation_policy = lambda: actor_evaluation.evaluation_policy(runtime)

    def trials(plan: Any, count: int) -> list[dict[str, Any]]:
        assert count == 1
        plan.policy.act()  # warmup
        plan.policy.act()  # evaluated step
        return [{"policy_version": 7}]

    monkeypatch.setattr(actor_evaluation, "_evaluation_trials", trials)
    actor_evaluation.evaluate(runtime, object(), object())

    assert policy.state.item() == 5.0
    assert policy.epsilon == 0.4
    assert len(requests) == 1
    snapshot = runtime.codec.decode(requests[0].evaluation_snapshot)
    if assignment_id:
        assert snapshot["state"].item() == 3.0
        assert snapshot["epsilon"] == 0.4
        assert requests[0].evaluations[0]["evaluation_assignment_id"] == assignment_id
    else:
        assert snapshot["state"].item() == 5.0
        assert snapshot["epsilon"] == 0.4
        assert "evaluation_assignment_id" not in requests[0].evaluations[0]
