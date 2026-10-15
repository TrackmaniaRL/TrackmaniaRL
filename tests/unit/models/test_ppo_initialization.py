from __future__ import annotations

import pytest
import torch
from torch import nn

from trackmaniarl.experiments.graph_iqn_v3 import CONTEXT_V3_DIM
from trackmaniarl.experiments.graph_iqn_v5 import RECOVERY_V5_DIM
from trackmaniarl.experiments.graph_iqn_v6 import IncidentGatedTrackGnnSimbaEncoderV6
from trackmaniarl.models.actors import GaussianActorConfig, PpoGaussianActor


@pytest.mark.parametrize("bias", [False, True])
def test_ppo_preserves_trainable_zero_projection_and_orthogonal_hidden_layer(*, bias: bool) -> None:
    torch.manual_seed(17)
    hidden = nn.Linear(4, 8, bias=False)
    projection = nn.Linear(8, 8, bias=bias)
    nn.init.zeros_(projection.weight)
    if projection.bias is not None:
        nn.init.zeros_(projection.bias)
    encoder = nn.Sequential(hidden, nn.Tanh(), projection)

    actor = PpoGaussianActor(encoder, GaussianActorConfig(8, 3))
    observations = torch.randn(4, 4)

    torch.testing.assert_close(encoder(observations), torch.zeros(4, 8), atol=0.0, rtol=0.0)
    torch.testing.assert_close(hidden.weight.T @ hidden.weight, 2.0 * torch.eye(4))
    log_probability, _ = actor.evaluate_latent_actions(observations, torch.ones(4, 3))
    log_probability.sum().backward()
    assert projection.weight.grad is not None
    assert torch.count_nonzero(projection.weight.grad) > 0


def test_ppo_incident_encoder_residuals_are_initially_inert() -> None:
    torch.manual_seed(17)
    encoder = IncidentGatedTrackGnnSimbaEncoderV6(
        context_residual_scale=0.02,
        spatial_residual_scale=0.02,
        recovery_residual_scale=0.01,
    )
    PpoGaussianActor(encoder, GaussianActorConfig(192, 3))
    context = torch.randn(3, CONTEXT_V3_DIM)
    track = torch.randn(3, 3, 88)
    recovery = torch.ones(3, RECOVERY_V5_DIM)

    corrections = (
        encoder._context_correction(context),
        encoder._spatial_correction(track),
        encoder.incident_gate(recovery) * encoder.recovery_adapter(recovery),
    )

    for correction in corrections:
        torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)
