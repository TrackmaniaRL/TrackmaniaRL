"""Demonstration retention must stay active without controlling actor actions."""

import pytest
import torch

from trackmaniarl.algorithms.value_based.objectives import ValueObjectiveContext
from trackmaniarl.experiments.shortcut_demonstrations import RetainedExpertObjective


def test_balanced_retention_only_updates_demonstrated_valid_positions() -> None:
    values = torch.tensor([[[0.0, 2.0]], [[2.0, 0.0]], [[4.0, 0.0]]], requires_grad=True)
    actions = torch.tensor([[0], [1], [1]])
    objective = RetainedExpertObjective([1.0, 3.0], weight=64)
    context = ValueObjectiveContext(
        values,
        actions,
        torch.ones_like(actions, dtype=torch.bool),
        {"demo_flags": [True, True, False], "collected_transition_count": 10**9},
    )
    loss = objective.loss(context)
    loss.backward()
    assert values.grad is not None
    assert torch.all(values.grad[2] == 0)
    assert values.grad[0, 0, 0] < 0
    assert values.grad[1, 0, 1] < 0
    assert values.grad[1].abs().sum() == pytest.approx(3 * values.grad[0].abs().sum())
    assert loss.item() > 0


def test_no_demo_batch_has_zero_loss_and_gradient() -> None:
    values = torch.randn(2, 1, 3, requires_grad=True)
    context = ValueObjectiveContext(
        values,
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1, dtype=torch.bool),
        {"demo_flags": [False, False]},
    )
    loss = RetainedExpertObjective([1, 1, 1]).loss(context)
    loss.backward()
    assert loss == 0
    assert torch.all(values.grad == 0)


@pytest.mark.parametrize(
    ("weights", "weight"),
    [([1, 1], 0), ([1, 1], float("nan")), ([0, 0], 1), ([1, -1], 1), ([1, float("inf")], 1)],
)
def test_invalid_retention_recipe_is_rejected(weights: list[float], weight: float) -> None:
    with pytest.raises(ValueError, match="required"):
        RetainedExpertObjective(weights, weight)
