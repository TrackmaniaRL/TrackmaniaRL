"""CPU numerical boundary checks; no environment, controller, or checkpoint IO."""

import math

import pytest
import torch

from trackmaniarl.algorithms.sac_config import SDSACConfig
from trackmaniarl.algorithms.sd_sac_objectives import (
    actor_diagnostics,
    soft_q_log_probabilities,
    terminal_value_loss,
)


@pytest.mark.parametrize("width", [0.0, -0.5, float("nan"), float("inf")])
def test_q_clipping_requires_a_positive_finite_width(width: float) -> None:
    with pytest.raises(ValueError, match="q_clip_epsilon must be finite and positive"):
        SDSACConfig(q_clip_epsilon=width).validate()


@pytest.mark.parametrize(
    "name", ["entropy_coefficient", "entropy_coefficient_min", "entropy_coefficient_max"]
)
@pytest.mark.parametrize("value", [1e-50, 1e50])
@pytest.mark.parametrize("learned", [False, True])
def test_temperature_must_survive_float32_conversion(
    name: str, value: float, *, learned: bool
) -> None:
    options = {name: value, "learn_entropy_coefficient": learned}
    with pytest.raises(ValueError, match=f"{name}.*float32"):
        SDSACConfig(**options).validate()


@pytest.mark.parametrize(
    "name", ["entropy_coefficient", "entropy_coefficient_min", "entropy_coefficient_max"]
)
def test_learned_temperature_must_survive_its_log_exp_roundtrip(name: str) -> None:
    maximum = torch.finfo(torch.float32).max
    assert math.isfinite(maximum)
    assert not bool(torch.isfinite(torch.tensor(maximum).log().exp()))
    with pytest.raises(ValueError, match=f"{name}.*float32 log/exp"):
        SDSACConfig(**{name: maximum}).validate()


@pytest.mark.parametrize("learned", [False, True])
def test_representable_small_temperature_and_existing_defaults_remain_valid(
    *, learned: bool
) -> None:
    smallest = float(torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))
    SDSACConfig(
        entropy_coefficient=smallest,
        entropy_coefficient_min=smallest,
        entropy_coefficient_max=smallest,
        learn_entropy_coefficient=learned,
    ).validate()
    SDSACConfig().validate()


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf"), 1e-50, 1e50])
def test_soft_q_rejects_nonrepresentable_temperature(alpha: float) -> None:
    with pytest.raises(ValueError, match="finite positive float32 alpha"):
        soft_q_log_probabilities(
            torch.tensor([[0.0, 1.0]]), torch.tensor(alpha, dtype=torch.float64)
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e50])
def test_soft_q_rejects_nonfinite_float32_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite float32 values"):
        soft_q_log_probabilities(
            torch.tensor([[0.0, value]], dtype=torch.float64), torch.tensor(0.2)
        )


@pytest.mark.parametrize("shape", [(2,), (1, 2, 1), (0, 2), (2, 0)])
def test_soft_q_requires_a_nonempty_batch_action_matrix(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"nonempty \[batch, actions\]"):
        soft_q_log_probabilities(torch.zeros(shape), torch.tensor(0.2))


def test_soft_q_requires_one_temperature() -> None:
    with pytest.raises(ValueError, match="scalar alpha"):
        soft_q_log_probabilities(torch.zeros(2, 2), torch.tensor([0.1, 0.2]))


def test_small_temperature_keeps_entropy_and_forward_kl_gradient_finite() -> None:
    values = torch.tensor([[0.0, 1.0]])
    alpha = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    target = soft_q_log_probabilities(values, alpha)
    logits = torch.tensor([[10.0, -10.0]], requires_grad=True)
    logp = logits.log_softmax(1)
    loss = -(target.exp() * logp).sum()
    gradient = torch.autograd.grad(loss, logits)[0]
    metrics = actor_diagnostics((logp.exp(), logp), (values, values), target)
    assert torch.isfinite(target).all()
    assert torch.isfinite(loss)
    assert torch.allclose(gradient, torch.tensor([[1.0, -1.0]]), atol=1e-6)
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    assert float(metrics["policy/soft_q_entropy"]) == 0


def test_soft_q_preserves_extreme_advantages_without_clipping() -> None:
    maximum = torch.finfo(torch.float32).max
    values = torch.tensor([[-maximum, 0.0, maximum]])
    alpha = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    target = soft_q_log_probabilities(values, alpha)
    expected = (values.double() - float(maximum)) / alpha.double()
    assert torch.isfinite(target).all()
    assert torch.equal(target, expected)
    assert torch.equal(target.exp(), torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))


def test_ordinary_soft_q_distribution_is_preserved() -> None:
    values = torch.tensor([[0.0, 0.25, 0.5], [-1.0, 1.0, 2.0]])
    alpha = torch.tensor(0.2)
    actual = soft_q_log_probabilities(values, alpha)
    expected = (values / alpha).log_softmax(1)
    assert torch.allclose(actual.float(), expected, rtol=1e-6, atol=1e-6)
    assert torch.allclose(actual.exp().sum(1), torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize("weight", [1.0, 1e-40, torch.finfo(torch.float32).smallest_normal / 2])
def test_tiny_terminal_weight_keeps_its_loss_and_finite_gradient(weight: float) -> None:
    q1 = torch.tensor([2.0, 0.0], requires_grad=True)
    q2 = torch.tensor([3.0, 0.0], requires_grad=True)
    loss = terminal_value_loss(
        (q1, q2), torch.zeros(2), torch.tensor([True, False]), torch.tensor([weight, 1.0])
    )
    gradients = torch.autograd.grad(loss, (q1, q2))
    assert float(loss.detach()) == 13.0
    assert torch.equal(gradients[0], torch.tensor([4.0, 0.0]))
    assert torch.equal(gradients[1], torch.tensor([6.0, 0.0]))


@pytest.mark.parametrize("scale", [1.0, 1e-40, 1e30])
def test_terminal_weighted_mean_is_scale_invariant(scale: float) -> None:
    q = torch.tensor([1.0, 3.0, 10.0], requires_grad=True)
    loss = terminal_value_loss(
        (q, q),
        torch.zeros(3),
        torch.tensor([True, True, False]),
        torch.tensor([1.0, 3.0, 2.0]) * scale,
    )
    loss.backward()
    assert float(loss.detach()) == pytest.approx(14.0)
    assert torch.allclose(q.grad, torch.tensor([1.0, 9.0, 0.0]))


def test_zero_terminal_weight_mass_has_zero_loss_and_gradients() -> None:
    q = torch.tensor([1.0, 3.0, 10.0], requires_grad=True)
    loss = terminal_value_loss(
        (q, q),
        torch.zeros(3),
        torch.tensor([True, True, False]),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.equal(q.grad, torch.zeros(3))


def test_diagnostic_critic_average_avoids_overflow_for_finite_values() -> None:
    values = torch.full((1, 2), torch.finfo(torch.float32).max)
    logp = torch.full((1, 2), -math.log(2))
    target = soft_q_log_probabilities(values, torch.tensor(0.2))
    metrics = actor_diagnostics((logp.exp(), logp), (values, values), target)
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    assert float(metrics["policy/q_greedy_regret"]) == 0.0
    assert float(metrics["policy/q_expected_regret"]) == 0.0
    assert float(metrics["critic/action_value_spread"]) == 0.0
