"""Check the actor objective against fixed action values and stale replay entropy."""

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch
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


def test_small_scale_78_action_objective_improves_despite_stale_uniform_entropy() -> None:
    """Exercise the same categorical width and anchor scale without game timing."""
    torch.manual_seed(17)
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 78)
    values = [0.0] * 78
    values[39] = 0.05
    model.q1, model.q2 = FixedValues(values), FixedValues(values)
    learner = StableDiscreteSoftActorCritic(
        model,
        entropy_penalty_reference="behavior",
        entropy_penalty_coefficient=0.005,
        learn_entropy_coefficient=False,
        entropy_coefficient=2e-5,
    )
    learner.setup({"seed": 17})
    batch = replace(
        _batch(BatchKind.DISCRETE),
        observations=torch.zeros(2, 4),
        metadata={"behavior_entropies": torch.full((2,), math.log(78))},
    )
    prepared = discrete_batch(learner._batch(batch))
    initial = float(model.actor.probabilities(prepared.observations)[0, 39].detach())
    for _ in range(3000):
        step = learner._actor_step(prepared, torch.tensor(2e-5))
        learner._optimize(step.loss, learner.actor_optimizer)
    final = model.actor.probabilities(prepared.observations).detach()[0]

    assert int(final.argmax()) == 39
    assert float(final[39]) > 3 * initial
    assert torch.isfinite(final).all()


def test_forward_kl_corrects_a_broad_wrong_greedy_maximum_at_candidate_rate() -> None:
    """A high-entropy actor must learn the missed action, not merely stay diverse."""
    torch.manual_seed(17)
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 78)
    values = [0.0] * 78
    values[57] = 0.12
    model.q1, model.q2 = FixedValues(values), FixedValues(values)
    learner = StableDiscreteSoftActorCritic(
        model,
        actor_objective="soft_q_forward_kl",
        actor_learning_rate=0.0001,
        entropy_penalty_reference="behavior",
        entropy_penalty_coefficient=0.0005,
        entropy_coefficient_min=0.01,
    )
    learner.setup({"seed": 17})
    with torch.no_grad():
        model.actor.logits.weight.zero_()
        model.actor.logits.bias.zero_()
        model.actor.logits.bias[0] = 0.02
    batch = replace(
        _batch(BatchKind.DISCRETE),
        observations=torch.zeros(2, 4),
        metadata={"behavior_entropies": torch.full((2,), math.log(78))},
    )
    prepared = discrete_batch(learner._batch(batch))
    before = model.actor.probabilities(prepared.observations).detach()
    assert int(before[0].argmax()) == 0
    initial_kl = float(
        learner._actor_step(prepared, torch.tensor(0.01)).diagnostics["policy/soft_q_forward_kl"]
    )
    for _ in range(200):
        step = learner._actor_step(prepared, torch.tensor(0.01))
        learner._optimize(step.loss, learner.actor_optimizer)
    final = model.actor.probabilities(prepared.observations).detach()
    after = learner._actor_step(prepared, torch.tensor(0.01))
    assert int(final[0].argmax()) == 57
    assert float(after.diagnostics["policy/soft_q_forward_kl"]) < initial_kl
    assert torch.equal(model.q1.values.detach(), torch.tensor(values))
    assert torch.equal(model.q2.values.detach(), torch.tensor(values))
    assert model.q1.values.grad is None
    assert model.q2.values.grad is None
