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


def test_masked_valid_demonstration_action_is_rejected_before_cross_entropy() -> None:
    values = torch.zeros(1, 1, 3, requires_grad=True)
    context = ValueObjectiveContext(
        values,
        torch.tensor([[1]]),
        torch.tensor([[True]]),
        {"demo_flags": [True]},
        torch.tensor([True, False, True]),
    )
    with pytest.raises(ValueError, match="demonstration action is excluded"):
        RetainedExpertObjective([1, 1, 1]).loss(context)
    assert values.grad is None


def test_padding_and_non_demo_rows_do_not_poison_loss_or_index_mask() -> None:
    values = torch.tensor([[[1.0, 0.0]], [[float("nan"), 1.0]], [[2.0, 1.0]]], requires_grad=True)
    context = ValueObjectiveContext(
        values,
        torch.tensor([[0], [-99], [1]]),
        torch.tensor([[True], [False], [True]]),
        {"demo_flags": [True, True, False]},
        torch.tensor([True, False]),
    )
    loss = RetainedExpertObjective([1, 1]).loss(context)
    assert torch.isfinite(loss)
    loss.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.all(values.grad[1:] == 0)


def test_all_masked_no_valid_demo_has_zero_loss_and_gradient() -> None:
    values = torch.randn(2, 1, 3, requires_grad=True)
    context = ValueObjectiveContext(
        values,
        torch.tensor([[999], [-1]]),
        torch.tensor([[False], [True]]),
        {"demo_flags": [True, False]},
        torch.tensor([False, False, False]),
    )
    loss = RetainedExpertObjective([1, 1, 1]).loss(context)
    loss.backward()
    assert loss == 0
    assert values.grad is not None
    assert torch.all(values.grad == 0)


def test_zero_weight_demonstrated_class_has_zero_loss_and_gradient() -> None:
    values = torch.randn(1, 1, 3, requires_grad=True)
    context = ValueObjectiveContext(
        values,
        torch.tensor([[1]]),
        torch.tensor([[True]]),
        {"demo_flags": [True]},
        torch.tensor([True, True, False]),
    )
    loss = RetainedExpertObjective([1, 0, 1]).loss(context)
    loss.backward()
    assert torch.isfinite(loss)
    assert loss == 0
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.all(values.grad == 0)


def test_batch_masks_and_demo_weights_select_only_valid_demonstrations() -> None:
    values = torch.tensor(
        [[[0.0, 3.0, 1.0], [5.0, 2.0, 3.0]], [[2.0, 3.0, 1.0], [1.0, 4.0, 2.0]]],
        requires_grad=True,
    )
    actions = torch.tensor([[2, -1], [1, 0]])
    context = ValueObjectiveContext(
        values,
        actions,
        torch.tensor([[True, False], [True, True]]),
        {
            "demo_flags": [[True, True], [False, True]],
            "demonstration_weights": [[0.5, float("nan")], [float("inf"), 1.0]],
        },
        torch.tensor([True, False, True]),
    )
    loss = RetainedExpertObjective([1, 1, 1], weight=1).loss(context)
    expected = (
        0.5 * torch.nn.functional.cross_entropy(values[0, 0, [0, 2]][None], torch.tensor([1]))
        + torch.nn.functional.cross_entropy(values[1, 1, [0, 2]][None], torch.tensor([0]))
    ) / 2
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert values.grad[0, 0, 2] < 0
    assert values.grad[1, 1, 0] < 0
    assert torch.all(values.grad[..., 1] == 0)
    assert torch.all(values.grad[0, 1] == 0)
    assert torch.all(values.grad[1, 0] == 0)


@pytest.mark.parametrize("quality", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_selected_demo_weight_is_rejected(quality: float) -> None:
    context = ValueObjectiveContext(
        torch.zeros(1, 1, 2, requires_grad=True),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        {"demo_flags": [True], "demonstration_weights": [quality]},
    )
    with pytest.raises(ValueError, match="finite demonstration weight"):
        RetainedExpertObjective([1, 1]).loss(context)


@pytest.mark.parametrize(
    ("weights", "weight"),
    [([1, 1], 0), ([1, 1], float("nan")), ([0, 0], 1), ([1, -1], 1), ([1, float("inf")], 1)],
)
def test_invalid_retention_recipe_is_rejected(weights: list[float], weight: float) -> None:
    with pytest.raises(ValueError, match="required"):
        RetainedExpertObjective(weights, weight)
