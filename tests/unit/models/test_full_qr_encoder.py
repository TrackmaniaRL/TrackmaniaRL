"""Verify the historical shape adapter preserves learned V6 computations."""

import pytest
import torch

from trackmaniarl.experiments.full_qr.components import BatchedIncidentEncoder
from trackmaniarl.experiments.graph_iqn_v3 import CONTEXT_V3_DIM
from trackmaniarl.experiments.graph_iqn_v6 import IncidentGatedTrackGnnSimbaEncoderV6


@pytest.mark.parametrize("leading", [(2,), (2, 3), (2, 1, 3)])
def test_adapter_preserves_active_residuals_and_gradients(leading: tuple[int, ...]) -> None:
    torch.manual_seed(17)
    adapter = BatchedIncidentEncoder(
        context_residual_scale=0.02,
        spatial_residual_scale=0.02,
        recovery_residual_scale=0.01,
    )
    # Exercise learned residuals too; their initial zero outputs hide shape errors.
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.uniform_(-0.05, 0.05)
    baseline = IncidentGatedTrackGnnSimbaEncoderV6(
        context_residual_scale=0.02,
        spatial_residual_scale=0.02,
        recovery_residual_scale=0.01,
    )
    baseline.load_state_dict(adapter.state_dict(), strict=True)
    shapes = {"physics": (60,), "track": (3, 88), "context": (CONTEXT_V3_DIM,), "recovery": (8,)}
    observation = {key: torch.rand(*leading, *shape) for key, shape in shapes.items()}
    observation["recovery"][..., 6:] = 1.0
    flattened = {key: value.reshape(-1, *shapes[key]) for key, value in observation.items()}
    expected = baseline(flattened)
    actual = adapter(observation)
    torch.testing.assert_close(actual, expected.reshape(*leading, 192), rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, parameter), (other_name, other) in zip(
        adapter.named_parameters(), baseline.named_parameters(), strict=True
    ):
        assert name == other_name
        assert parameter.grad is not None
        assert other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
