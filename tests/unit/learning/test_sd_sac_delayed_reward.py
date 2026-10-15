"""Learn a delayed goal from transitions, with an independent soft-Bellman oracle."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.models.actors import CategoricalActor


def _soft_values() -> torch.Tensor:
    # Two states: wait costs 0.1; advance reaches the next state then the goal.
    # Solve the optimal entropy-regularized Bellman equations independently.
    values = torch.zeros(2, dtype=torch.float64)
    for _ in range(300):
        q = torch.stack(
            [
                torch.tensor([-0.1, -0.1]) + 0.9 * values,
                torch.stack((-0.01 + 0.9 * values[1], values.new_tensor(1.0))),
            ],
            dim=1,
        )
        values = 0.05 * torch.logsumexp(q / 0.05, dim=1)
    return q.float()


@pytest.mark.parametrize("objective", ["sac", "soft_q_forward_kl"])
def test_sd_sac_learns_to_advance_to_a_delayed_goal(objective: str) -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 2, 2)
    model.q1, model.q2 = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    learner = StableDiscreteSoftActorCritic(
        model,
        execution=TorchExecutionConfig(device="cpu", torch_threads=1),
        learning_rate=0.03,
        target_tau=0.1,
        entropy_coefficient=0.05,
        learn_entropy_coefficient=False,
        entropy_penalty_coefficient=0,
        actor_objective=objective,
    )
    learner.setup({"seed": 17})
    observations = torch.eye(2).repeat_interleave(2, dim=0)
    batch = replace(
        _batch(BatchKind.DISCRETE),
        observations=observations,
        actions=torch.tensor([0, 1, 0, 1]),
        rewards=torch.tensor([-0.1, -0.01, -0.1, 1.0]),
        next_observations=torch.eye(2)[torch.tensor([0, 1, 1, 0])],
        terminated=torch.tensor([False, False, False, True]),
        # A collection time limit must still bootstrap from the final observation.
        truncated=torch.tensor([True, False, False, False]),
        bootstrap_discounts=torch.tensor([0.9, 0.9, 0.9, 0.0]),
        transition_ids=list(range(4)),
    )
    for _ in range(500):
        learner.update(batch)

    oracle = _soft_values()
    for critic in (model.q1, model.q2):
        assert torch.allclose(critic(torch.eye(2)), oracle, atol=0.015)
    policy = learner.policy()
    assert [policy.act(state, PolicyMode.EVALUATION) for state in torch.eye(2)] == [1, 1]
    # Wait has a bootstrapped value; the goal action has only its known reward.
    assert float(model.q1(torch.eye(2))[0, 0].detach()) > 0.6
    assert float(model.q1(torch.eye(2))[1, 1].detach()) == pytest.approx(1.0, abs=0.01)
