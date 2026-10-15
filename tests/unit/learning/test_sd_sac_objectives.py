from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch
from trackmaniarl.algorithms.sd_sac_objectives import (
    actor_diagnostics,
    soft_q_log_probabilities,
    terminal_value_loss,
)
from trackmaniarl.models.actors import CategoricalActor


class FixedValues(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values))
        self.saw_grad_enabled = False

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.saw_grad_enabled |= torch.is_grad_enabled()
        return self.values.expand(len(observations), -1)


def test_actor_gradient_matches_scalar_objective_without_a_critic_graph() -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 3)
    model.q1 = FixedValues([0.0, 0.1, 0.04])
    model.q2 = FixedValues([0.02, 0.08, 0.06])
    learner = StableDiscreteSoftActorCritic(
        model, entropy_penalty_reference="behavior", entropy_penalty_coefficient=0.005
    )
    learner.setup({"seed": 17})
    original = _batch(BatchKind.DISCRETE)
    batch = replace(
        original, metadata={"behavior_entropies": torch.ones(len(original.transition_ids))}
    )
    prepared = discrete_batch(learner._batch(batch))
    step = learner._actor_step(prepared, torch.tensor(0.01))
    actual = torch.autograd.grad(step.loss, model.actor.logits.bias)[0]
    p = model.actor.probabilities(prepared.observations)
    entropy = -(p * p.log()).sum(1)
    expected_loss = (p * (0.01 * p.log() - torch.tensor([0.01, 0.09, 0.05]))).sum(1).mean()
    expected_loss += 0.005 * (entropy - 1).square().mean()
    expected = torch.autograd.grad(expected_loss, model.actor.logits.bias)[0]

    assert torch.allclose(actual, expected)
    assert not model.q1.saw_grad_enabled
    assert not model.q2.saw_grad_enabled
    assert model.q1.values.grad is None
    assert model.q2.values.grad is None
    assert float(step.diagnostics["loss/entropy_penalty"]) == pytest.approx(
        float((0.005 * (entropy - 1).square().mean()).detach())
    )
    assert float(step.diagnostics["critic/action_value_spread"]) == pytest.approx(0.08)


def test_terminal_loss_is_not_diluted_by_nonterminal_rows() -> None:
    q = torch.full((256,), 18.0, requires_grad=True)
    terminal = torch.zeros(256, dtype=torch.bool)
    terminal[0] = True
    loss = terminal_value_loss((q, q), torch.full((256,), -2.0), terminal)
    loss.backward()
    assert float(loss.detach()) == 800
    assert float(q.grad[0]) == 80
    assert not bool(q.grad[1:].any())


def test_terminal_loss_respects_relative_importance_weights() -> None:
    q = torch.tensor([1.0, 3.0], requires_grad=True)
    loss = terminal_value_loss(
        (q, q), torch.zeros(2), torch.ones(2, dtype=torch.bool), torch.tensor([0.25, 0.75])
    )
    assert float(loss.detach()) == 14


def test_no_true_terminal_has_zero_auxiliary_gradient() -> None:
    q = torch.tensor([1.0, 3.0], requires_grad=True)
    loss = terminal_value_loss((q, q), torch.zeros(2), torch.zeros(2, dtype=torch.bool))
    loss.backward()
    assert float(loss.detach()) == 0
    assert torch.equal(q.grad, torch.zeros(2))


def test_auxiliary_loss_restores_terminal_gradient_when_clipping_blocks_td() -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 3)
    model.q1, model.q2 = FixedValues([18.0, 0.0, 0.0]), FixedValues([18.0, 0.0, 0.0])
    learner = StableDiscreteSoftActorCritic(model, terminal_value_loss_coefficient=1.0)
    learner.setup({"seed": 17})
    learner.target_model.q1.values.data[0] = 19
    learner.target_model.q2.values.data[0] = 19
    batch = replace(
        _batch(BatchKind.DISCRETE),
        observations=torch.zeros(2, 4),
        next_observations=torch.zeros(2, 4),
        actions=torch.zeros(2, dtype=torch.long),
        rewards=torch.full((2,), -2.0),
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, True]),
        bootstrap_discounts=torch.zeros(2),
    )
    prepared = discrete_batch(learner._batch(batch))
    step = learner._critic_step(prepared, torch.tensor(0.01))
    step.loss.backward()
    assert float(step.diagnostics["critic/clip_blocked_gradient_fraction"]) == 1
    assert float(step.diagnostics["loss/terminal_value"]) == 800
    assert float(model.q1.values.grad[0]) == 40
    assert float(model.q2.values.grad[0]) == 40


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_invalid_terminal_coefficient_is_rejected(coefficient: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        StableDiscreteSoftActorCritic(
            DiscreteSacModel(), terminal_value_loss_coefficient=coefficient
        )


def test_default_keeps_previous_checkpoint_options_and_loss() -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel())
    learner.setup({"seed": 17})
    assert "terminal_value_loss_coefficient" not in learner.state_dict()["sd_sac_options"]
    batch = discrete_batch(learner._batch(_batch(BatchKind.DISCRETE)))
    step = learner._critic_step(batch, torch.tensor(0.2))
    losses, _ = learner._critic_losses(batch, (step.q1, step.q2, step.targets))
    assert torch.equal(step.loss, losses.mean())


def test_checkpoint_pins_terminal_objective_before_loading_weights() -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel(), terminal_value_loss_coefficient=1.0)
    learner.setup({"seed": 17})
    state = deepcopy(learner.state_dict())
    restored = StableDiscreteSoftActorCritic(
        DiscreteSacModel(), terminal_value_loss_coefficient=1.0
    )
    restored.setup({"seed": 17})
    restored.load_state_dict(state)
    assert restored.state_dict()["sd_sac_options"]["terminal_value_loss_coefficient"] == 1
    wrong = StableDiscreteSoftActorCritic(DiscreteSacModel())
    wrong.setup({"seed": 17})
    before = deepcopy(wrong.model.state_dict())
    with pytest.raises(ValueError, match="options do not match"):
        wrong.load_state_dict(state)
    assert all(torch.equal(v, before[k]) for k, v in wrong.model.state_dict().items())


def test_broad_policy_reports_wrong_greedy_maximum_and_small_margin() -> None:
    probabilities = torch.full((1, 78), 1 / 78)
    probabilities[0, 0] += 0.001
    probabilities[0, 57] -= 0.001
    values = torch.zeros_like(probabilities)
    values[0, 57] = 0.1
    target = soft_q_log_probabilities(values, torch.tensor(0.01))
    metrics = actor_diagnostics((probabilities, probabilities.log()), (values, values), target)

    assert float(metrics["policy/entropy_min"]) > math.log(78) - 0.01
    assert float(metrics["policy/q_greedy_agreement"]) == 0
    assert float(metrics["policy/greedy_probability_margin"]) == pytest.approx(0.001)
    assert float(metrics["policy/critic_greedy_probability"]) == pytest.approx(1 / 78 - 0.001)
    assert float(metrics["critic/greedy_value_margin"]) == pytest.approx(0.1)
    assert float(metrics["policy/soft_q_cross_entropy"]) > math.log(78)
    assert all(torch.isfinite(value) and not value.requires_grad for value in metrics.values())
