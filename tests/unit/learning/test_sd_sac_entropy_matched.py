"""Entropy-matched actor targets are scale-free and fail closed on invalid setups."""

import math
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch
from trackmaniarl.algorithms.sd_sac_objectives import entropy_matched_target
from trackmaniarl.models.actors import CategoricalActor

MATCHED: dict[str, object] = {
    "actor_objective": "soft_q_forward_kl",
    "actor_temperature": "entropy_matched",
    "target_entropy": 0.8,
    "learn_entropy_coefficient": False,
    "backup_entropy_coefficient": 0.0,
    "entropy_penalty_coefficient": 0.0,
    "execution": {"device": "cpu", "torch_threads": 1},
}


class FixedValues(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.values.expand(len(observations), -1)


def _learner(model: nn.Module, **overrides: object) -> StableDiscreteSoftActorCritic:
    learner = StableDiscreteSoftActorCritic(model, **{**MATCHED, **overrides})
    learner.setup({"seed": 17})
    return learner


def _fixed_value_model(values: list[float]) -> nn.Module:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, len(values))
    model.q1, model.q2 = FixedValues(values), FixedValues(values)
    return model


def _entropy(logs: torch.Tensor) -> torch.Tensor:
    return -(logs.exp() * logs).sum(1)


@pytest.mark.parametrize("scale", [1e-9, 1e-4, 1e-2, 1.0, 100.0, 1e6])
def test_target_has_requested_entropy_at_any_value_scale(scale: float) -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(64, 78, generator=generator, dtype=torch.float64) * scale
    target = entropy_matched_target(values, 0.8)
    temperature = target.temperature
    assert bool(torch.isfinite(temperature).all() & (temperature > 0).all())
    assert bool(torch.isfinite(target.log_probabilities).all())
    assert not bool(target.unreachable.any())
    assert float(target.entropy_error) < 1e-8
    torch.testing.assert_close(
        _entropy(target.log_probabilities), torch.full((64,), 0.8, dtype=torch.float64)
    )


def test_random_gap_structures_reach_the_requested_entropy() -> None:
    generator = torch.Generator().manual_seed(43)
    exponents = torch.empty(256, 78, dtype=torch.float64).uniform_(-12, 3, generator=generator)
    values = -(10.0**exponents)
    values[:, 0] = 0.0
    target = entropy_matched_target(values, 0.8)
    assert not bool(target.unreachable.any())
    assert float(target.entropy_error) < 1e-8
    assert float((_entropy(target.log_probabilities) - 0.8).abs().max()) < 1e-8


@pytest.mark.parametrize(("scale", "shift"), [(1e-3, -5.0), (7.0, 100.0), (1e3, 0.0)])
def test_target_is_invariant_to_positive_affine_value_changes(scale: float, shift: float) -> None:
    generator = torch.Generator().manual_seed(29)
    values = torch.randn(32, 78, generator=generator, dtype=torch.float64)
    reference = entropy_matched_target(values, 0.8).log_probabilities
    changed = entropy_matched_target(values * scale + shift, 0.8).log_probabilities
    torch.testing.assert_close(changed.exp(), reference.exp(), rtol=0, atol=1e-9)


def test_flat_rows_and_tied_maxima_above_the_target_are_reported() -> None:
    flat = entropy_matched_target(torch.full((2, 78), 3.25), 0.8)
    torch.testing.assert_close(
        flat.log_probabilities.exp(), torch.full((2, 78), 1 / 78, dtype=torch.float64)
    )
    assert bool(flat.unreachable.all())
    tied = torch.full((1, 78), -1.0)
    tied[0, :3] = 0.0
    target = entropy_matched_target(tied, 0.8)
    # Three equal maxima keep log(3) nats however small the temperature gets.
    assert bool(target.unreachable.all())
    assert float(_entropy(target.log_probabilities)[0]) == pytest.approx(math.log(3), abs=1e-9)
    assert bool(torch.isfinite(target.log_probabilities).all())
    assert float(target.entropy_error) == 0


def test_tied_maxima_below_the_target_are_solved() -> None:
    values = torch.full((1, 78), -1.0)
    values[0, :2] = 0.0
    target = entropy_matched_target(values, 0.8)
    assert not bool(target.unreachable.any())
    assert float(_entropy(target.log_probabilities)[0]) == pytest.approx(0.8, abs=1e-8)


@pytest.mark.parametrize("target", [0.0, -0.1, math.log(78), 5.0])
def test_out_of_range_target_entropy_rejected(target: float) -> None:
    with pytest.raises(ValueError, match="target_entropy"):
        entropy_matched_target(torch.zeros(1, 78), target)


def test_nonfinite_values_rejected() -> None:
    values = torch.zeros(1, 78)
    values[0, 5] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        entropy_matched_target(values, 0.8)


def test_actor_target_ignores_the_scalar_alpha() -> None:
    values = [0.0] * 78
    values[35] = 0.0021
    model = _fixed_value_model(values)
    learner = _learner(model)
    batch = replace(_batch(BatchKind.DISCRETE), observations=torch.zeros(8, 4))
    prepared = discrete_batch(learner._batch(batch))
    expected = entropy_matched_target(torch.tensor([values] * 8), 0.8).log_probabilities
    logs = model.actor.log_probabilities(prepared.observations).detach()
    expected_loss = float(-(expected.exp() * logs).sum(1).mean())
    for alpha in (0.2, 0.01):
        step = learner._actor_step(prepared, torch.tensor(alpha))
        assert float(step.loss.detach()) == pytest.approx(expected_loss, rel=1e-6)
        assert float(step.diagnostics["policy/soft_q_entropy"]) == pytest.approx(0.8, abs=1e-6)
        assert float(step.diagnostics["policy/matched_alpha_mean"]) > 0
        assert float(step.diagnostics["policy/matched_unreachable_fraction"]) == 0
        assert float(step.diagnostics["policy/matched_entropy_error_max"]) < 1e-8


def test_start_like_action_gap_sharpens_only_with_entropy_matched_target() -> None:
    """CP33856 start: mean-Q top-2 gap 0.0021 under alpha 0.01 leaves the target flat."""
    values = [0.0] * 78
    values[35] = 0.0021
    batch = replace(_batch(BatchKind.DISCRETE), observations=torch.zeros(8, 4))
    final = {}
    for temperature in ("learned", "entropy_matched"):
        torch.manual_seed(17)
        model = _fixed_value_model(values)
        overrides: dict[str, object] = {"actor_learning_rate": 0.05}
        if temperature == "learned":
            overrides.update(actor_temperature="learned", backup_entropy_coefficient=None)
        learner = _learner(model, **overrides)
        prepared = discrete_batch(learner._batch(batch))
        for _ in range(300):
            step = learner._actor_step(prepared, torch.tensor(0.01))
            learner._optimize(step.loss, learner.actor_optimizer)
        final[temperature] = model.actor.log_probabilities(prepared.observations).detach()
    assert float(_entropy(final["learned"]).mean()) > 4.2
    assert float(_entropy(final["entropy_matched"]).mean()) < 1.5
    assert bool((final["entropy_matched"].argmax(1) == 35).all())


@pytest.mark.parametrize("coefficient", [None, 0.0, 0.05])
def test_critic_backup_uses_configured_entropy_coefficient(coefficient: float | None) -> None:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(),
        backup_entropy_coefficient=coefficient,
        entropy_penalty_coefficient=0,
        execution={"device": "cpu", "torch_threads": 1},
    )
    learner.setup({"seed": 17})
    batch = discrete_batch(learner._batch(_batch(BatchKind.DISCRETE)))
    used = 0.2 if coefficient is None else coefficient
    with torch.no_grad():
        logs = learner.model.actor.log_probabilities(batch.next_observations).float()
        target = learner.target_model
        q = 0.5 * target.q1(batch.next_observations) + 0.5 * target.q2(batch.next_observations)
        continuation = (logs.exp() * (q - used * logs)).sum(1)
    expected = batch.rewards + batch.discounts * continuation
    torch.testing.assert_close(learner._critic_targets(batch, torch.tensor(0.2)), expected)


def test_entropy_matched_update_is_finite_without_alpha_optimizer() -> None:
    learner = _learner(DiscreteSacModel())
    metrics, priorities = learner.update(_batch(BatchKind.DISCRETE))
    assert learner.log_alpha is None
    assert learner.alpha_optimizer is None
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["policy/soft_q_entropy"] == pytest.approx(0.8, abs=1e-6)
    assert metrics["policy/matched_alpha_mean"] > 0
    assert metrics["policy/matched_unreachable_fraction"] == 0
    assert metrics["policy/matched_entropy_error_max"] < 1e-8
    assert priorities.transition_ids == list(range(8))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actor_objective": "sac"}, "soft_q_forward_kl"),
        ({"target_entropy": None}, "target_entropy"),
        ({"learn_entropy_coefficient": True}, "learn_entropy_coefficient"),
        ({"backup_entropy_coefficient": None}, "backup_entropy_coefficient"),
        ({"backup_entropy_coefficient": -0.1}, "backup_entropy_coefficient"),
        ({"backup_entropy_coefficient": float("nan")}, "backup_entropy_coefficient"),
        ({"actor_temperature": "fixed"}, "actor_temperature"),
    ],
)
def test_invalid_entropy_matched_configuration_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StableDiscreteSoftActorCritic(DiscreteSacModel(), **{**MATCHED, **overrides})


def test_default_options_keep_the_canonical_checkpoint_contract() -> None:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(), execution={"device": "cpu", "torch_threads": 1}
    )
    learner.setup({"seed": 17})
    options = learner.state_dict()["sd_sac_options"]
    assert "actor_temperature" not in options
    assert "backup_entropy_coefficient" not in options


def test_checkpoint_pins_entropy_matched_and_backup_options() -> None:
    state = _learner(DiscreteSacModel()).state_dict()
    _learner(DiscreteSacModel()).load_state_dict(state)
    learned = _learner(
        DiscreteSacModel(), actor_temperature="learned", backup_entropy_coefficient=None
    )
    with pytest.raises(ValueError, match="options do not match"):
        learned.load_state_dict(state)
    backup_only = StableDiscreteSoftActorCritic(
        DiscreteSacModel(),
        backup_entropy_coefficient=0.0,
        entropy_penalty_coefficient=0,
        execution={"device": "cpu", "torch_threads": 1},
    )
    backup_only.setup({"seed": 17})
    legacy = dict(backup_only.state_dict())
    del legacy["sd_sac_options"]
    with pytest.raises(ValueError, match="legacy checkpoint"):
        backup_only.load_state_dict(legacy)


def test_near_tied_maxima_still_reach_the_requested_entropy() -> None:
    """Review case: a fixed log-temperature bracket stopped at log(3) for these gaps."""
    values = torch.tensor([[0.0, -1e-7, -2e-7] + [-1.0] * 75], dtype=torch.float64)
    target = entropy_matched_target(values, 0.8)
    assert not bool(target.unreachable.any())
    assert float(_entropy(target.log_probabilities)[0]) == pytest.approx(0.8, abs=1e-8)
    assert float(target.temperature[0, 0]) == pytest.approx(9.2944095e-8, rel=1e-6)


def test_target_entropy_at_log_action_count_rejected_before_any_weight_changes() -> None:
    """Review case: the critic Adam step ran before the actor solver rejected ln(A)."""
    learner = _learner(DiscreteSacModel(), target_entropy=math.log(3))
    before = deepcopy(learner.model.state_dict())
    with pytest.raises(ValueError, match="target_entropy"):
        learner.update(_batch(BatchKind.DISCRETE))
    after = learner.model.state_dict()
    assert all(torch.equal(after[name], value) for name, value in before.items())
    assert not learner.critic_optimizer.state
    assert not learner.actor_optimizer.state
