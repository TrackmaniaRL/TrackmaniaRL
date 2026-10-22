"""Twin critic errors must not cancel in prioritized replay feedback."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from tests.unit.learning.test_sd_sac_objectives import FixedValues
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sd_sac_objectives import actor_diagnostics
from trackmaniarl.models.actors import CategoricalActor


@pytest.mark.parametrize(("rule", "expected"), [("mean_absolute", 2.0), ("mean_value", 0.0)])
def test_opposing_errors_keep_priority_unless_legacy_rule_selected(
    rule: str, expected: float
) -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 3)
    model.q1, model.q2 = FixedValues([2.0] * 3), FixedValues([-2.0] * 3)
    learner = StableDiscreteSoftActorCritic(
        model,
        priority_td_error=rule,
        entropy_penalty_coefficient=0,
        execution={"device": "cpu", "torch_threads": 1},
    )
    learner.setup({"seed": 17})
    batch = _batch(BatchKind.DISCRETE)
    batch = replace(
        batch,
        rewards=torch.zeros(8),
        terminated=torch.ones(8, dtype=torch.bool),
        bootstrap_discounts=torch.zeros(8),
    )
    metrics, priorities = learner.update(batch)
    assert priorities.transition_ids == batch.transition_ids
    assert priorities.priorities == pytest.approx([expected] * 8)
    assert metrics["critic/twin_td_mae"] == 2
    assert metrics["critic/twin_td_rmse"] == 2
    assert metrics["critic/td_rmse"] == 0  # Mean-value diagnostic retained.
    assert metrics["critic/priority_cancellation_fraction"] == 1


def _learner(**options: object) -> StableDiscreteSoftActorCritic:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(), execution={"device": "cpu", "torch_threads": 1}, **options
    )
    learner.setup({"seed": 17})
    return learner


def test_default_priority_rule_and_resume_are_explicit() -> None:
    current = _learner()
    state = deepcopy(current.state_dict())
    assert state["sd_sac_options"]["priority_td_error"] == "mean_absolute"
    current.load_state_dict(state)
    legacy = _learner(priority_td_error="mean_value")
    old_state = deepcopy(legacy.state_dict())
    assert "priority_td_error" not in old_state["sd_sac_options"]
    legacy.load_state_dict(old_state)
    for source, target in ((old_state, current), (state, legacy)):
        before = deepcopy(target.model.state_dict())
        with pytest.raises(ValueError, match="options do not match"):
            target.load_state_dict(source)
        assert all(torch.equal(before[k], v) for k, v in target.model.state_dict().items())
    del old_state["sd_sac_options"]
    legacy.load_state_dict(old_state)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        current.load_state_dict(old_state)


def test_unknown_priority_rule_rejected() -> None:
    with pytest.raises(ValueError, match="priority_td_error"):
        _learner(priority_td_error="unknown")


def test_action_disagreement_distinguishes_value_offset_from_rank_conflict() -> None:
    logs = torch.tensor([[0.0, 1.0, 2.0]]).log_softmax(1)
    q = torch.tensor([[0.0, 1.0, 2.0]])
    offset = actor_diagnostics((logs.exp(), logs), (q, q + 10), logs)
    assert float(offset["critic/all_action_disagreement"]) == 10
    assert float(offset["critic/centered_action_disagreement"]) == 0
    assert float(offset["critic/mean_greedy_twin_agreement"]) == 1
    conflict = actor_diagnostics((logs.exp(), logs), (q, q.flip(1)), logs)
    assert float(conflict["critic/centered_action_disagreement"]) > 1
    assert float(conflict["critic/mean_greedy_twin_agreement"]) == 0
