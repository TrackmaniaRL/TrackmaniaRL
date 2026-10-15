"""Known terminal rewards must train even when the clipped TD loss has no gradient."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from tests.unit.learning.test_sd_sac_actor_objective import FixedValues
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch
from trackmaniarl.algorithms.sd_sac_objectives import terminal_value_loss
from trackmaniarl.models.actors import CategoricalActor


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
