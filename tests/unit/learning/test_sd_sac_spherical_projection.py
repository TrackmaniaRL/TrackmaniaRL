"""SD-SAC must preserve the optimizer geometry of hyperspherical backbones."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.models.actors import CategoricalActor
from trackmaniarl.models.backbones import HypersphericalLinear, SimbaV2Backbone


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
