"""Fail closed on invalid SD-SAC models/batches before any optimizer mutation."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch


def _learner(model: nn.Module | None = None) -> StableDiscreteSoftActorCritic:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel() if model is None else model,
        entropy_penalty_coefficient=0,
        execution={"device": "cpu", "torch_threads": 1},
    )
    learner.setup({"seed": 17})
    return learner


@pytest.mark.parametrize("shared", ["actor_q1", "actor_q2", "q1_q2"])
def test_overlapping_parameter_owners_rejected_before_optimizer_creation(shared: str) -> None:
    model = DiscreteSacModel()
    if shared == "actor_q1":
        model.q1.encoder = model.actor.encoder
    elif shared == "actor_q2":
        model.q2.encoder = model.actor.encoder
    else:
        model.q2.encoder = model.q1.encoder
    with pytest.raises(ValueError, match="disjoint"):
        _learner(model)


def test_distinct_model_parameter_owners_are_accepted() -> None:
    learner = _learner()
    actor = {id(p) for group in learner.actor_optimizer.param_groups for p in group["params"]}
    critic = [id(p) for group in learner.critic_optimizer.param_groups for p in group["params"]]
    assert not actor.intersection(critic)
    assert len(critic) == len(set(critic))


class _NonfiniteValues(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.new_full((len(observation), 3), float("nan"))


def test_true_terminal_zero_discount_ignores_nonfinite_continuation() -> None:
    learner = _learner()
    original = _batch(BatchKind.DISCRETE)
    size = len(original.transition_ids)
    batch = replace(
        original,
        terminated=torch.ones(size, dtype=torch.bool),
        truncated=torch.zeros(size, dtype=torch.bool),
        bootstrap_discounts=torch.zeros(size),
    )
    learner.target_model.q1 = _NonfiniteValues()
    learner.target_model.q2 = _NonfiniteValues()
    targets = learner._critic_targets(discrete_batch(batch), torch.tensor(0.2))
    assert torch.equal(targets, batch.rewards)


@pytest.mark.parametrize("case", [(False, 0.0), (False, 0.995), (True, 0.995)])
def test_nonterminal_or_nonzero_discount_nonfinite_target_rejected(
    case: tuple[bool, float],
) -> None:
    learner = _learner()
    original = _batch(BatchKind.DISCRETE)
    size = len(original.transition_ids)
    terminal, discount = case
    batch = replace(
        original,
        terminated=torch.full((size,), terminal, dtype=torch.bool),
        bootstrap_discounts=torch.full((size,), discount),
    )
    learner.target_model.q1 = _NonfiniteValues()
    learner.target_model.q2 = _NonfiniteValues()
    with pytest.raises(ValueError, match="non-finite"):
        learner._critic_targets(discrete_batch(batch), torch.tensor(0.2))


@pytest.mark.parametrize(
    "fault",
    [
        "short_rewards",
        "short_actions",
        "short_discounts",
        "short_terminated",
        "short_truncated",
        "fractional_actions",
        "boolean_actions",
        "negative_actions",
        "out_of_range_actions",
        "negative_weights",
        "zero_weights",
        "nan_weights",
        "inf_weights",
        "scalar_weights",
        "matrix_weights",
        "list_weights",
        "nan_rewards",
        "overflow_rewards",
        "negative_discounts",
        "inf_discounts",
        "float_terminated",
        "float_truncated",
        "complex_weights",
        "overflow_weight_sum",
    ],
)
def test_bad_batch_rejected_before_any_optimizer_mutation(fault: str) -> None:
    learner = _learner()
    batch = _batch(BatchKind.DISCRETE)
    size = len(batch.transition_ids)
    fields = {
        "short_rewards": {"rewards": torch.ones(1)},
        "short_actions": {"actions": torch.zeros(1, dtype=torch.long)},
        "short_discounts": {"bootstrap_discounts": torch.ones(1)},
        "short_terminated": {"terminated": torch.zeros(1, dtype=torch.bool)},
        "short_truncated": {"truncated": torch.zeros(1, dtype=torch.bool)},
        "fractional_actions": {"actions": torch.full((size,), 0.75)},
        "boolean_actions": {"actions": torch.zeros(size, dtype=torch.bool)},
        "negative_actions": {"actions": torch.full((size,), -1, dtype=torch.long)},
        "out_of_range_actions": {"actions": torch.full((size,), 3, dtype=torch.long)},
        "negative_weights": {"importance_weights": -torch.ones(size)},
        "zero_weights": {"importance_weights": torch.zeros(size)},
        "nan_weights": {"importance_weights": torch.full((size,), float("nan"))},
        "inf_weights": {"importance_weights": torch.full((size,), float("inf"))},
        "scalar_weights": {"importance_weights": torch.ones(1)},
        "matrix_weights": {"importance_weights": torch.ones(size // 2, 2)},
        "list_weights": {"importance_weights": [1.0] * size},
        "nan_rewards": {"rewards": torch.full((size,), float("nan"))},
        "overflow_rewards": {"rewards": torch.full((size,), 1e100, dtype=torch.float64)},
        "negative_discounts": {"bootstrap_discounts": -torch.ones(size)},
        "inf_discounts": {"bootstrap_discounts": torch.full((size,), float("inf"))},
        "float_terminated": {"terminated": torch.zeros(size)},
        "float_truncated": {"truncated": torch.zeros(size)},
        "complex_weights": {"importance_weights": torch.ones(size, dtype=torch.complex64)},
        "overflow_weight_sum": {"importance_weights": torch.full((size,), 1e38)},
    }
    before = deepcopy(learner.model.state_dict())
    with (
        patch.object(learner, "_optimize", side_effect=AssertionError("optimizer reached")),
        pytest.raises((ValueError, TypeError), match=r"SD-SAC|action"),
    ):
        learner.update(replace(batch, **fields[fault]))
    assert all(
        torch.equal(value, before[name]) for name, value in learner.model.state_dict().items()
    )
    assert not learner.actor_optimizer.state
    assert not learner.critic_optimizer.state


def test_column_vectors_are_explicitly_normalized_without_changing_loss() -> None:
    learner = _learner()
    batch = _batch(BatchKind.DISCRETE)
    weights = torch.linspace(0.1, 1.0, len(batch.transition_ids))
    batch = replace(batch, importance_weights=weights)
    column = replace(
        batch,
        actions=batch.actions[:, None],
        rewards=batch.rewards[:, None],
        bootstrap_discounts=batch.bootstrap_discounts[:, None],
        terminated=batch.terminated[:, None],
        truncated=batch.truncated[:, None],
        importance_weights=weights[:, None],
    )
    # Compare complete updates on independent identical learner states.
    saved = deepcopy(learner.state_dict())
    vector_metrics, _ = learner.update(batch)
    restored = _learner()
    restored.load_state_dict(saved)
    column_metrics, _ = restored.update(column)
    assert column_metrics["loss/critic"] == pytest.approx(vector_metrics["loss/critic"])
    assert column_metrics["loss/actor"] == pytest.approx(vector_metrics["loss/actor"])


class _WrongWidthActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def probabilities(self, observation: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(len(observation), 1).softmax(1)


def test_actor_critic_width_mismatch_rejected_before_update() -> None:
    model = DiscreteSacModel()
    model.actor = _WrongWidthActor()
    learner = _learner(model)
    with (
        patch.object(learner, "_optimize", side_effect=AssertionError("optimizer reached")),
        pytest.raises(ValueError, match=r"SD-SAC|shape|action"),
    ):
        learner.update(_batch(BatchKind.DISCRETE))


@pytest.mark.parametrize("fault", ["missing", "shape", "nan", "negative", "above_maximum"])
def test_invalid_behavior_entropy_rejected_before_optimizer(fault: str) -> None:
    learner = _learner()
    learner.entropy_penalty_coefficient = 0.5
    learner.entropy_penalty_reference = "behavior"
    batch = _batch(BatchKind.DISCRETE)
    size = len(batch.transition_ids)
    values = {
        "shape": torch.ones(1),
        "nan": torch.full((size,), float("nan")),
        "negative": -torch.ones(size),
        "above_maximum": torch.full((size,), 2.0),
    }
    metadata = {} if fault == "missing" else {"behavior_entropies": values[fault]}
    with (
        patch.object(learner, "_optimize", side_effect=AssertionError("optimizer reached")),
        pytest.raises(ValueError, match=r"entropy|entropies"),
    ):
        learner.update(replace(batch, metadata=metadata))
    assert not learner.critic_optimizer.state


class _MixedValues(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        result = observation.new_ones((len(observation), 3))
        result[[0, 2]] = float("nan")
        return result


def test_only_true_terminal_rows_ignore_invalid_continuation_in_mixed_batch() -> None:
    learner = _learner()
    original = _batch(BatchKind.DISCRETE)
    terminal = torch.zeros(len(original.transition_ids), dtype=torch.bool)
    terminal[[0, 2]] = True
    discounts = original.bootstrap_discounts.clone()
    discounts[terminal] = 0
    batch = replace(original, terminated=terminal, bootstrap_discounts=discounts)
    learner.target_model.q1 = _MixedValues()
    learner.target_model.q2 = _MixedValues()
    targets = learner._critic_targets(discrete_batch(batch), torch.tensor(0.2))
    assert torch.isfinite(targets).all()
    assert torch.equal(targets[terminal], batch.rewards[terminal])
    assert (targets[~terminal] > batch.rewards[~terminal]).all()


@pytest.mark.parametrize("owner", ["q1", "q2", "target_q1", "target_q2"])
def test_critic_matrix_shape_rejected_before_optimizer(owner: str) -> None:
    learner = _learner()
    module = learner.target_model if owner.startswith("target_") else learner.model
    getattr(module, owner.removeprefix("target_")).head = nn.Linear(16, 2)
    with (
        patch.object(learner, "_optimize", side_effect=AssertionError("optimizer reached")),
        pytest.raises(ValueError, match=r"SD-SAC|shape"),
    ):
        learner.update(_batch(BatchKind.DISCRETE))


class _MappingEncoder(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(torch.cat((observation["left"], observation["right"]), dim=1))


def test_batch_count_does_not_assume_tensor_observations() -> None:
    model = DiscreteSacModel()
    for module in (model.actor, model.q1, model.q2):
        module.encoder = _MappingEncoder(module.encoder)
    learner = _learner(model)
    batch = _batch(BatchKind.DISCRETE)
    batch = replace(
        batch,
        observations={"left": batch.observations[:, :2], "right": batch.observations[:, 2:]},
        next_observations={
            "left": batch.next_observations[:, :2],
            "right": batch.next_observations[:, 2:],
        },
    )
    metrics, priorities = learner.update(batch)
    assert torch.isfinite(torch.tensor(metrics["loss/critic"]))
    assert priorities.transition_ids == batch.transition_ids


def test_cached_actor_graph_uses_updated_critics_for_loss_and_gradient() -> None:
    learner = _learner()
    reference = _learner()
    reference.load_state_dict(deepcopy(learner.state_dict()))
    batch = _batch(BatchKind.DISCRETE)
    prepared = discrete_batch(reference._batch(batch))
    alpha = torch.tensor(0.2)
    critic = reference._critic_step(prepared, alpha)
    stale_actor = reference._actor_step(prepared, alpha)
    reference._optimize(critic.loss, reference.critic_optimizer)
    actor = reference._actor_step(prepared, alpha)
    reference._optimize(actor.loss, reference.actor_optimizer)
    metrics, _ = learner.update(batch)
    assert abs(actor.loss.item() - stale_actor.loss.item()) > 1e-8
    assert metrics["loss/actor"] == pytest.approx(actor.loss.item())
    for actual, expected in zip(
        learner.model.actor.parameters(), reference.model.actor.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_cached_policy_preserves_dropout_and_batchnorm_forward_order() -> None:
    model = DiscreteSacModel()
    for module in (model.actor, model.q1, model.q2):
        module.encoder.layers.append(nn.BatchNorm1d(16))
        module.encoder.layers.append(nn.Dropout(0.2))
    learner = _learner(model)
    reference = _learner(deepcopy(model))
    reference.load_state_dict(deepcopy(learner.state_dict()))
    batch = _batch(BatchKind.DISCRETE)
    rng = torch.get_rng_state()
    prepared = discrete_batch(reference._batch(batch))
    alpha = torch.tensor(0.2)
    critic = reference._critic_step(prepared, alpha)
    reference._optimize(critic.loss, reference.critic_optimizer)
    actor = reference._actor_step(prepared, alpha)
    reference._optimize(actor.loss, reference.actor_optimizer)
    torch.set_rng_state(rng)
    metrics, _ = learner.update(batch)
    assert metrics["loss/actor"] == pytest.approx(actor.loss.item())
    for name, actual in learner.model.state_dict().items():
        torch.testing.assert_close(actual, reference.model.state_dict()[name], rtol=0, atol=1e-7)


@pytest.mark.parametrize("distribution", ["equal", "unbalanced"])
@pytest.mark.parametrize("terminal_coefficient", [0.0, 0.5])
def test_importance_weight_scale_does_not_change_updates(
    distribution: str, terminal_coefficient: float
) -> None:
    learner = _learner()
    learner.terminal_value_loss_coefficient = terminal_coefficient
    scaled = _learner()
    scaled.terminal_value_loss_coefficient = terminal_coefficient
    scaled.load_state_dict(deepcopy(learner.state_dict()))
    batch = _batch(BatchKind.DISCRETE)
    size = len(batch.transition_ids)
    weights = (
        torch.ones(size)
        if distribution == "equal"
        else torch.tensor([0.01, 0.1, 0.25, 1.0, 2.0, 8.0, 16.0, 32.0])
    )
    terminated = torch.arange(size) < 2
    batch = replace(
        batch,
        terminated=terminated,
        bootstrap_discounts=batch.bootstrap_discounts * ~terminated,
        importance_weights=weights,
    )
    scaled_batch = replace(batch, importance_weights=weights * 1e-12)
    for _ in range(3):
        expected, _ = learner.update(batch)
        actual, _ = scaled.update(scaled_batch)
        for name in (
            "loss/critic",
            "loss/actor",
            "loss/entropy",
            "state/alpha",
            "loss/terminal_value",
        ):
            assert actual[name] == pytest.approx(expected[name], rel=2e-6, abs=1e-7)
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, scaled.model.state_dict()[name], rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(learner.log_alpha, scaled.log_alpha, rtol=0, atol=1e-7)
