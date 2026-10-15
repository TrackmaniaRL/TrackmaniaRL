from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tests.unit._composite_value_fixtures import _assert_nested_state_equal, _value_model
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models.contracts import ValuePhase
from trackmaniarl.models.heads import FixedQuantileHead, FixedQuantileHeadConfig
from trackmaniarl.models.strategies import FixedQuantileStrategy


@pytest.mark.parametrize("quantiles", [2, 4, 64])
@pytest.mark.parametrize("target_samples", [1, 7])
def test_qr_objective_sums_prediction_quantiles_and_averages_target_samples(
    quantiles: int, target_samples: int
) -> None:
    strategy = FixedQuantileStrategy(quantiles)
    support = strategy.support(torch.zeros(1, 3), ValuePhase.TRAIN, None)
    predictions = torch.zeros(1, quantiles, requires_grad=True)
    targets = torch.ones(1, target_samples)

    loss = strategy.regression_loss(predictions, targets, support)

    # For delta=1, each quantile contributes tau/2 and gradient -tau.
    torch.testing.assert_close(loss, torch.tensor([quantiles / 4]))
    loss.sum().backward()
    torch.testing.assert_close(predictions.grad, -support.points)


def test_qr_loss_matches_scalar_pairwise_reference_for_sequence_positions() -> None:
    strategy = FixedQuantileStrategy(4)
    support = strategy.support(torch.zeros(2, 3, 5), ValuePhase.TRAIN, None)
    predictions = torch.linspace(-2, 2, 24).reshape(2, 3, 4).requires_grad_()
    targets = torch.linspace(2, -1, 18).reshape(2, 3, 3)
    references = []
    for batch_index in range(2):
        positions = []
        for time_index in range(3):
            terms = []
            for quantile_index in range(4):
                for target_index in range(3):
                    delta = (
                        targets[batch_index, time_index, target_index]
                        - predictions[batch_index, time_index, quantile_index]
                    )
                    huber = 0.5 * delta.square() if abs(delta) <= 1 else abs(delta) - 0.5
                    tau = (quantile_index + 0.5) / 4
                    weight = abs(tau - float(delta.detach() < 0))
                    terms.append(weight * huber / 3)
            positions.append(torch.stack(terms).sum())
        references.append(torch.stack(positions))
    expected = torch.stack(references)
    actual = strategy.regression_loss(predictions, targets, support)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), predictions, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), predictions)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("dueling", [False, True])
def test_fixed_head_selected_actions_match_all_action_values_and_gradients(
    *, dueling: bool
) -> None:
    torch.manual_seed(17)
    head = FixedQuantileHead(FixedQuantileHeadConfig(5, 4, 6, dueling))
    reference_head = deepcopy(head)
    features = torch.randn(2, 3, 5, requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    support = FixedQuantileStrategy(6).support(features, ValuePhase.TRAIN, None)
    actions = torch.tensor([[0, 1, 3], [2, 3, 0]])
    selected = head.evaluate_actions(features, support, actions)
    gathered = (
        reference_head.evaluate_all(reference_features, support)
        .gather(-1, actions[..., None, None].expand(2, 3, 6, 1))
        .squeeze(-1)
    )
    torch.testing.assert_close(selected, gathered)
    weights = torch.linspace(-1, 1, selected.numel()).reshape_as(selected)
    (weights * selected).sum().backward()
    (weights * gathered).sum().backward()
    torch.testing.assert_close(features.grad, reference_features.grad)
    for actual_parameter, expected_parameter in zip(
        head.parameters(), reference_head.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_qr_learns_terminal_bandit_and_resumes_after_zstd_roundtrip(tmp_path: Path) -> None:
    def learner() -> DiscreteValueLearner:
        result = DiscreteValueLearner(
            _value_model("qr"),
            learning_rate=0.02,
            target_update_interval=5,
            target_tau=0.0,
            diagnostics_interval_updates=1,
            execution={"device": "cpu", "torch_threads": 2},
        )
        result.setup({"seed": 17})
        return result

    batch = TrainingBatch(
        data={},
        observations=torch.zeros(3, 4),
        actions=torch.arange(3),
        rewards=torch.tensor([-1.0, 0.0, 1.0]),
        next_observations=torch.zeros(3, 4),
        terminated=torch.ones(3, dtype=torch.bool),
        truncated=torch.zeros(3, dtype=torch.bool),
        bootstrap_discounts=torch.zeros(3),
        transition_ids=[10, 11, 12],
    )
    source = learner()
    for _ in range(100):
        metrics, priorities = source.update(batch)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        assert priorities.transition_ids == batch.transition_ids
        assert metrics["debug/bootstrap_zero_fraction"] == 1.0
    assert source.policy().act(torch.zeros(4), PolicyMode.EVALUATION) == 2
    codec = TorchCheckpointCodec()
    checkpoint = tmp_path / "qr.pt"
    codec.save(source.state_dict(), checkpoint)
    assert checkpoint.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
    restored = learner()
    restored.load_state_dict(codec.load(checkpoint))
    _assert_nested_state_equal(restored.state_dict(), source.state_dict())
    source_metrics, source_priorities = source.update(batch)
    restored_metrics, restored_priorities = restored.update(batch)
    for name, value in source_metrics.items():
        if not name.startswith("timing/"):
            assert restored_metrics[name] == value
    assert restored_priorities == source_priorities
    _assert_nested_state_equal(restored.state_dict(), source.state_dict())
