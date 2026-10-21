from __future__ import annotations

import torch

from trackmaniarl.algorithms.proximal_policy_optimization import ProximalPolicyOptimization
from trackmaniarl.trackmania.baseline import TelemetryPpoModel


def test_ppo_warmup_preserves_rng_weights_and_first_behavior_sample() -> None:
    learner = ProximalPolicyOptimization(
        TelemetryPpoModel(hidden_dim=8),
        execution={"device": "cpu", "precision": "float32"},
        normalize_observations=False,
        normalize_rewards=False,
    )
    learner.setup({"seed": 17})
    policy = learner.policy()
    reference = learner.policy()
    observation = torch.zeros(33)
    rng = torch.get_rng_state()
    weights = {key: value.clone() for key, value in policy.export_state().items()}
    policy.warm_up(observation)
    policy.warm_up(observation)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, policy.export_state()[key]) for key, value in weights.items())
    action, info = policy.act_with_info(observation)
    torch.set_rng_state(rng)
    expected_action, expected_info = reference.act_with_info(observation)
    assert (action == expected_action).all()
    assert (
        info["_trackmaniarl_behavior_log_probability"]
        == expected_info["_trackmaniarl_behavior_log_probability"]
    )
    assert info["_trackmaniarl_behavior_value"] == expected_info["_trackmaniarl_behavior_value"]
