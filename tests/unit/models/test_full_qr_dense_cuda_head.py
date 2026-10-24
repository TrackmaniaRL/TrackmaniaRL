"""Check CUDA execution against the original sparse formula, including gradients."""

from copy import deepcopy

import pytest
import torch

from trackmaniarl.models.contracts import ValuePhase
from trackmaniarl.models.heads import FixedQuantileHead, FixedQuantileHeadConfig
from trackmaniarl.models.strategies import FixedQuantileStrategy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical execution test")
@pytest.mark.parametrize("leading", [(64,), (16, 8)])
@pytest.mark.parametrize("dueling", [False, True])
def test_dense_cuda_selected_preserves_sparse_outputs_and_gradients(
    leading: tuple[int, ...], *, dueling: bool
) -> None:
    torch.manual_seed(17)
    reference = FixedQuantileHead(FixedQuantileHeadConfig(192, 78, 64, dueling=dueling)).cuda()
    optimized = deepcopy(reference)
    optimized.dense_cuda_selected = True
    left = torch.randn(*leading, 192, device="cuda", requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    actions = torch.randint(78, leading, device="cuda")
    support = FixedQuantileStrategy(64).support(left, ValuePhase.TRAIN, None)
    expected = reference.evaluate_actions(left, support, actions)
    actual = optimized.evaluate_actions(right, support, actions)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-6)
    for (name, parameter), (other_name, other) in zip(
        reference.named_parameters(), optimized.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-3, atol=1e-6)
    assert tuple(reference.state_dict()) == tuple(optimized.state_dict())
