from __future__ import annotations

import numpy as np
import pytest
import torch

from trackmaniarl.algorithms.proximal_policy_optimization import ProximalPolicyOptimization
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.trackmania.baseline import TelemetryPpoModel


def _learner(*, normalize: bool = True) -> ProximalPolicyOptimization:
    learner = ProximalPolicyOptimization(
        TelemetryPpoModel(hidden_dim=8),
        execution={"device": "cpu", "precision": "float32"},
        normalize_observations=normalize,
    )
    learner.setup({"seed": 17})
    return learner


def test_ppo_policy_round_trip_transfers_observation_statistics() -> None:
    source = _learner()
    target = _learner()
    source.observation_normalizer.update(
        torch.stack((torch.full((33,), 4.0), torch.full((33,), 8.0))), 1
    )
    original = source.policy()
    replica = target.policy()
    observation = torch.linspace(-2.0, 2.0, 33)
    expected, expected_info = original.act_with_info(observation, PolicyMode.EVALUATION)
    before = replica.act(observation, PolicyMode.EVALUATION)
    assert not np.allclose(before, expected)

    codec = WireCodec(max_message_bytes=1_000_000)
    replica.load_state(codec.decode(codec.encode(original.export_state())))
    actual, info = replica.act_with_info(observation, PolicyMode.EVALUATION)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert info["_trackmaniarl_behavior_value"] == pytest.approx(
        expected_info["_trackmaniarl_behavior_value"]
    )


def test_ppo_policy_rejects_mismatched_normalizer_configuration() -> None:
    source = _learner().policy()
    target = _learner(normalize=False).policy()
    with pytest.raises(ValueError, match="normalizer configuration"):
        target.load_state(source.export_state())
