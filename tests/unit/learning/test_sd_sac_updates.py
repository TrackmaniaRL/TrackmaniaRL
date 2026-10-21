from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from tests.unit.learning.test_sd_sac_objectives import FixedValues
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch
from trackmaniarl.algorithms.sd_sac_objectives import (
    categorical_statistics,
    soft_q_log_probabilities,
)
from trackmaniarl.algorithms.stable_discrete_soft_actor_critic import _DiscreteActorStep
from trackmaniarl.core.runtime import _TrainingComponents, _validate_training_contract
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.models.actors import CategoricalActor
from trackmaniarl.models.backbones import HypersphericalLinear, SimbaV2Backbone
from trackmaniarl.project.scaffold_run_templates import _trackmania_actor_critic_config


def test_optimizer_rates_and_evaluated_checkpoint_round_trip() -> None:
    options = {"actor_learning_rate": 9e-4, "entropy_learning_rate": 1e-4}
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel(), **options)
    learner.setup({"seed": 17})
    state = learner.state_dict_for_policy(learner.policy().export_state())
    restored = StableDiscreteSoftActorCritic(DiscreteSacModel(), **options)
    restored.setup({"seed": 17})
    restored.load_state_dict(state)
    assert restored.actor_optimizer.param_groups[0]["lr"] == 9e-4
    assert restored.critic_optimizer.param_groups[0]["lr"] == 3e-4
    assert restored.alpha_optimizer.param_groups[0]["lr"] == 1e-4
    assert restored.state_dict()["sd_sac_options"] == state["sd_sac_options"]


@pytest.mark.parametrize("entropy", [0.0, 1.0])
def test_temperature_moves_in_the_direction_of_its_entropy_target(entropy: float) -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel(), target_entropy=0.8)
    learner.setup({"seed": 17})
    initial = float(learner.log_alpha.detach())
    learner._entropy_loss(_DiscreteActorStep(torch.tensor(0.0), torch.tensor([entropy]), 3))
    difference = float(learner.log_alpha.detach()) - initial
    assert difference > 0 if entropy < 0.8 else difference < 0


@pytest.mark.parametrize("entropy", [0.0, 1.0])
def test_temperature_respects_explicit_bounds_and_checkpoint_restores(entropy: float) -> None:
    options = {
        "entropy_coefficient": 0.01,
        "entropy_coefficient_min": 0.01,
        "entropy_coefficient_max": 0.01,
        "target_entropy": 0.8,
    }
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel(), **options)
    learner.setup({"seed": 17})
    learner._entropy_loss(_DiscreteActorStep(torch.tensor(0.0), torch.tensor([entropy]), 3))
    assert float(learner.log_alpha.detach().exp()) == pytest.approx(0.01)
    restored = StableDiscreteSoftActorCritic(DiscreteSacModel(), **options)
    restored.setup({"seed": 17})
    restored.load_state_dict(deepcopy(learner.state_dict()))
    assert torch.equal(restored.log_alpha, learner.log_alpha)


@pytest.mark.parametrize(
    "options",
    [
        {"entropy_learning_rate": 0},
        {"entropy_learning_rate": float("nan")},
        {"entropy_coefficient_min": -1},
        {"entropy_coefficient_max": float("inf")},
        {"entropy_coefficient_min": 0.3},
        {"entropy_coefficient_max": 0.1},
        {"q_clip_epsilon": float("nan")},
        {"entropy_penalty_coefficient": float("inf")},
        {"target_entropy": -1},
        {"target_entropy": float("nan")},
        {"actor_objective": "unknown"},
    ],
)
def test_stability_options_reject_invalid_values(options: dict) -> None:
    with pytest.raises(ValueError, match=r"SD-SAC|entropy_coefficient|actor_objective"):
        StableDiscreteSoftActorCritic(DiscreteSacModel(), **options)


def test_impossible_entropy_target_rejected_before_any_optimizer_step() -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel(), target_entropy=2)
    learner.setup({"seed": 17})
    before = deepcopy(learner.state_dict())
    with pytest.raises(ValueError, match=r"log\(action_count\)"):
        learner.update(_batch(BatchKind.DISCRETE))
    assert all(
        torch.equal(value, before["model"][key])
        for key, value in learner.model.state_dict().items()
    )


def test_saturated_forward_kl_has_a_gradient_toward_the_missed_best_action() -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 3)
    model.q1, model.q2 = FixedValues([0, 0, 1.0]), FixedValues([0, 0, 1.0])
    learner = StableDiscreteSoftActorCritic(
        model, actor_objective="soft_q_forward_kl", entropy_penalty_coefficient=0
    )
    learner.setup({"seed": 17})
    with torch.no_grad():
        model.actor.logits.weight.zero_()
        model.actor.logits.bias.copy_(torch.tensor([1000.0, 0.0, -1000.0]))
    batch = replace(_batch(BatchKind.DISCRETE), observations=torch.zeros(2, 4))
    step = learner._actor_step(discrete_batch(learner._batch(batch)), torch.tensor(0.01))
    gradient = torch.autograd.grad(step.loss, model.actor.logits.bias)[0]
    assert torch.isfinite(step.loss)
    assert torch.allclose(gradient, torch.tensor([1.0, 0.0, -1.0]), atol=1e-6)
    assert model.q1.values.grad is None
    assert not model.q1.saw_grad_enabled
    assert float(step.diagnostics["policy/q_greedy_regret"]) == 1
    assert float(step.diagnostics["policy/q_greedy_agreement"]) == 0
    assert all(torch.isfinite(v) for v in step.diagnostics.values())


def test_soft_q_target_is_invariant_to_a_shared_value_baseline() -> None:
    values = torch.tensor([[0.0, 0.25, 0.5]])
    alpha = torch.tensor(0.01)
    assert torch.equal(
        soft_q_log_probabilities(values, alpha), soft_q_log_probabilities(values + 1000, alpha)
    )


def test_exact_soft_q_policy_has_zero_distance_and_consistent_entropy() -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 3)
    model.q1, model.q2 = FixedValues([0, 0.1, 0.2]), FixedValues([0, 0.1, 0.2])
    learner = StableDiscreteSoftActorCritic(model, entropy_penalty_coefficient=0)
    learner.setup({"seed": 17})
    with torch.no_grad():
        model.actor.logits.weight.zero_()
        model.actor.logits.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
    batch = replace(_batch(BatchKind.DISCRETE), observations=torch.zeros(2, 4))
    step = learner._actor_step(discrete_batch(learner._batch(batch)), torch.tensor(0.1))
    assert float(step.diagnostics["policy/soft_q_reverse_kl"]) == pytest.approx(0, abs=1e-6)
    assert float(step.diagnostics["policy/soft_q_forward_kl"]) == pytest.approx(0, abs=1e-6)
    assert float(step.diagnostics["policy/soft_q_entropy"]) == pytest.approx(
        float(step.entropy.detach().mean())
    )


def test_probabilities_only_custom_actor_retains_canonical_contract() -> None:
    class Actor:
        def probabilities(self, observations: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0]])

    p, logp = categorical_statistics(Actor(), torch.zeros(1, 4))
    assert torch.isfinite(logp).all()
    assert float(-(p * logp).sum()) == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_cannot_silently_switch_actor_objectives(*, legacy: bool) -> None:
    original = StableDiscreteSoftActorCritic(DiscreteSacModel())
    original.setup({"seed": 17})
    state = deepcopy(original.state_dict())
    if legacy:
        state.pop("sd_sac_options")
    variant = StableDiscreteSoftActorCritic(DiscreteSacModel(), actor_objective="soft_q_forward_kl")
    variant.setup({"seed": 17})
    before = deepcopy(variant.model.state_dict())
    with pytest.raises(ValueError, match="SD-SAC options"):
        variant.load_state_dict(state)
    assert all(torch.equal(v, before[k]) for k, v in variant.model.state_dict().items())


def test_checkpoint_rejects_temperature_outside_configured_bounds() -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel(), entropy_coefficient_min=0.01)
    learner.setup({"seed": 17})
    state = deepcopy(learner.state_dict())
    state["log_alpha"] = torch.tensor(math.log(0.001))
    with pytest.raises(ValueError, match="outside configured bounds"):
        learner.load_state_dict(state)


def test_generic_n_step_rejected_by_runtime_and_direct_update() -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel())
    spec = RunSpec.model_validate(yaml.safe_load(_trackmania_actor_critic_config("sd-sac")))
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"n_step": 3})})
    with pytest.raises(ValueError, match="n_step=1"):
        _validate_training_contract(
            spec, _TrainingComponents(learner, SimpleNamespace(), SimpleNamespace())
        )
    learner.setup({"seed": 17})
    with pytest.raises(ValueError, match="intermediate entropy"):
        learner.update(replace(_batch(BatchKind.DISCRETE), metadata={"n_step": 3}))


def test_clipping_reports_when_it_blocks_a_gradient() -> None:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, 3)
    model.q1, model.q2 = FixedValues([0, 0, 0.0]), FixedValues([0, 0, 0.0])
    learner = StableDiscreteSoftActorCritic(model, q_clip_epsilon=0.5)
    learner.setup({"seed": 17})
    batch = replace(
        _batch(BatchKind.DISCRETE),
        observations=torch.zeros(2, 4),
        actions=torch.zeros(2, dtype=torch.long),
        terminated=torch.tensor([True, False]),
    )
    q1, q2 = torch.full((2,), 2.0, requires_grad=True), torch.full((2,), 2.0, requires_grad=True)
    losses, metrics = learner._critic_losses(
        discrete_batch(learner._batch(batch)), (q1, q2, torch.full((2,), 2.0))
    )
    losses.sum().backward()
    assert float(metrics["critic/clipped_fraction"]) == 1
    assert float(metrics["critic/clip_blocked_gradient_fraction"]) == 1
    assert float(metrics["critic/terminal_samples"]) == 1
    assert float(metrics["critic/terminal_td_mae"]) == 0
    assert torch.equal(q1.grad, torch.zeros(2))


def test_terminal_prediction_error_is_reported_separately() -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel())
    learner.setup({"seed": 17})
    batch = replace(
        _batch(BatchKind.DISCRETE),
        actions=torch.tensor([0, 1]),
        terminated=torch.tensor([True, False]),
    )
    _, metrics = learner._critic_losses(
        discrete_batch(learner._batch(batch)),
        (torch.tensor([1.5, 100.0]), torch.tensor([2.5, 100.0]), torch.tensor([-2.0, 100.0])),
    )
    assert float(metrics["critic/terminal_td_mae"]) == 4
    assert float(metrics["critic/terminal_samples"]) == 1


@pytest.mark.parametrize("optimizer", ["actor_optimizer", "critic_optimizer", "alpha_optimizer"])
def test_checkpoint_rejects_misleading_optimizer_rates_before_loading_weights(
    optimizer: str,
) -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel(), actor_learning_rate=9e-4)
    learner.setup({"seed": 17})
    state = deepcopy(learner.state_dict())
    state[optimizer]["param_groups"][0]["lr"] = 0.01
    before = deepcopy(learner.model.state_dict())
    with pytest.raises(ValueError, match="learning rate does not match"):
        learner.load_state_dict(state)
    assert all(torch.equal(v, before[k]) for k, v in learner.model.state_dict().items())


@pytest.mark.parametrize("log_alpha", [-1000.0, 1000.0, float("nan")])
def test_checkpoint_rejects_nonrepresentable_temperature(log_alpha: float) -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel())
    learner.setup({"seed": 17})
    state = deepcopy(learner.state_dict())
    state["log_alpha"] = torch.tensor(log_alpha)
    with pytest.raises(ValueError, match=r"alpha must|log_alpha must"):
        learner.load_state_dict(state)


def spherical_model() -> nn.Module:
    model = nn.Module()
    model.actor = CategoricalActor(SimbaV2Backbone(4, 8, block_count=1), 8, 3)
    model.q1 = nn.Sequential(SimbaV2Backbone(4, 8, block_count=1), nn.Linear(8, 3))
    model.q2 = nn.Sequential(SimbaV2Backbone(4, 8, block_count=1), nn.Linear(8, 3))
    return model


@pytest.mark.parametrize("objective", ["sac", "soft_q_forward_kl"])
def test_all_online_branches_project_after_adam_and_before_target_averaging(objective: str) -> None:
    torch.manual_seed(17)
    model = spherical_model()
    learner = StableDiscreteSoftActorCritic(
        model, entropy_penalty_coefficient=0, actor_objective=objective
    )
    learner.setup({"seed": 17})
    initial_target = deepcopy(learner.target_model.state_dict())
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, HypersphericalLinear):
                layer.weight.mul_(4)
    learner.update(_batch(BatchKind.DISCRETE))
    for name, layer in model.named_modules():
        if isinstance(layer, HypersphericalLinear):
            assert torch.allclose(
                layer.weight.norm(dim=1), torch.ones(len(layer.weight)), atol=2e-6
            )
            key = f"{name}.weight"
            expected = initial_target[key].lerp(layer.weight, learner.target_tau)
            assert torch.allclose(learner.target_model.state_dict()[key], expected, atol=2e-6)
    assert learner.actor_optimizer.state
    assert learner.critic_optimizer.state
    state = deepcopy(learner.state_dict())
    restored = StableDiscreteSoftActorCritic(
        spherical_model(), entropy_penalty_coefficient=0, actor_objective=objective
    )
    restored.setup({"seed": 17})
    restored.load_state_dict(state)
    assert restored.state_dict()["sd_sac_options"] == state["sd_sac_options"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_unprojected_saved_adam_contract_is_rejected_before_restore() -> None:
    learner = StableDiscreteSoftActorCritic(spherical_model())
    learner.setup({"seed": 17})
    state = deepcopy(learner.state_dict())
    assert state["sd_sac_options"]["hyperspherical_projection"] is True
    del state["sd_sac_options"]["hyperspherical_projection"]
    with pytest.raises(ValueError, match="options do not match"):
        learner.load_state_dict(state)
    del state["sd_sac_options"]
    with pytest.raises(ValueError, match="legacy checkpoint"):
        learner.load_state_dict(state)


def test_plain_mlp_optimizer_contract_is_unchanged() -> None:
    learner = StableDiscreteSoftActorCritic(DiscreteSacModel())
    learner.setup({"seed": 17})
    assert "hyperspherical_projection" not in learner.state_dict()["sd_sac_options"]


@pytest.mark.parametrize("actor_rate", [None, 9e-4])
def test_actor_rate_does_not_change_critic_or_temperature_rate(actor_rate: float | None) -> None:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(), learning_rate=3e-4, actor_learning_rate=actor_rate
    )
    learner.setup({"seed": 17})
    assert learner.actor_optimizer.param_groups[0]["lr"] == (
        3e-4 if actor_rate is None else actor_rate
    )
    assert learner.critic_optimizer.param_groups[0]["lr"] == 3e-4
    assert learner.alpha_optimizer is not None
    assert learner.alpha_optimizer.param_groups[0]["lr"] == 3e-4


@pytest.mark.parametrize("actor_rate", [0, -1, float("nan"), float("inf")])
def test_invalid_actor_learning_rate_is_rejected(actor_rate: float) -> None:
    with pytest.raises(ValueError, match="actor_learning_rate"):
        StableDiscreteSoftActorCritic(DiscreteSacModel(), actor_learning_rate=actor_rate)
