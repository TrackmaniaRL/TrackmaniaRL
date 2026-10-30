"""Check gradient magnitude/direction and that autograd does not update weights."""

import pytest
import torch

from .gradients import compare_gradients, gradients, model_digest


def test_weighted_opposing_gradients_and_no_parameter_update() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    parameter = next(model.parameters())
    before = model_digest(model)
    td = parameter.sum()
    bc = -3 * parameter.sum()
    named = {"head.weight": parameter}
    result = compare_gradients(gradients(td, named), gradients(bc, named))
    assert result["all"]["weighted_bc_to_td_ratio"] == pytest.approx(3)
    assert result["head"]["cosine"] == pytest.approx(-1)
    assert model_digest(model) == before
    assert parameter.grad is None


def test_zero_gradient_has_no_invented_angle() -> None:
    value = torch.nn.Parameter(torch.zeros(2))
    named = {"encoder.weight": value}
    output = compare_gradients(gradients(value.sum(), named), gradients(0 * value.sum(), named))
    assert output["all"]["cosine"] is None
    assert output["all"]["weighted_bc_to_td_ratio"] == 0
