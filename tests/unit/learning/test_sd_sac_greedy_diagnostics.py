"""Expose incorrect greedy maxima even when categorical entropy stays high."""

import math

import pytest
import torch

from trackmaniarl.algorithms.sd_sac_objectives import actor_diagnostics, soft_q_log_probabilities


def test_broad_policy_reports_wrong_greedy_maximum_and_small_margin() -> None:
    probabilities = torch.full((1, 78), 1 / 78)
    probabilities[0, 0] += 0.001
    probabilities[0, 57] -= 0.001
    values = torch.zeros_like(probabilities)
    values[0, 57] = 0.1
    target = soft_q_log_probabilities(values, torch.tensor(0.01))
    metrics = actor_diagnostics((probabilities, probabilities.log()), (values, values), target)

    assert float(metrics["policy/entropy_min"]) > math.log(78) - 0.01
    assert float(metrics["policy/q_greedy_agreement"]) == 0
    assert float(metrics["policy/greedy_probability_margin"]) == pytest.approx(0.001)
    assert float(metrics["policy/critic_greedy_probability"]) == pytest.approx(1 / 78 - 0.001)
    assert float(metrics["critic/greedy_value_margin"]) == pytest.approx(0.1)
    assert float(metrics["policy/soft_q_cross_entropy"]) > math.log(78)
    assert all(torch.isfinite(value) and not value.requires_grad for value in metrics.values())


def test_tied_actor_and_critic_maxima_report_zero_margins() -> None:
    probabilities = torch.full((3, 2), 0.5)
    values = torch.ones_like(probabilities)
    target = soft_q_log_probabilities(values, torch.tensor(0.01))
    metrics = actor_diagnostics((probabilities, probabilities.log()), (values, values), target)

    assert float(metrics["policy/greedy_probability_margin"]) == 0
    assert float(metrics["critic/greedy_value_margin"]) == 0
    assert float(metrics["policy/critic_greedy_probability"]) == 0.5
    assert float(metrics["policy/soft_q_cross_entropy"]) == pytest.approx(math.log(2))
