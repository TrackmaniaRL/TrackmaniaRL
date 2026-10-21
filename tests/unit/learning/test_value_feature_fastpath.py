from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from tests.unit._composite_value_fixtures import _batch, _value_model
from trackmaniarl.algorithms.value_based import DiscreteValueLearner, update_helpers
from trackmaniarl.algorithms.value_based.batches import ValueBatchView
from trackmaniarl.models.composite import BatchLayout


def _legacy_features(
    learner: DiscreteValueLearner, view: ValueBatchView
) -> update_helpers._EncodedFeatures:
    layout = BatchLayout.SEQUENCE if view.sequence else BatchLayout.FRAMES
    online = learner.model.encode_masked_sequence(
        view.batch.observations, layout, learner.burn_in, masks=view.masks
    )
    target = learner.target_model.encode_masked_sequence(
        view.batch.observations, layout, learner.burn_in, masks=view.masks
    )
    final_online = learner.model.encode_masked_sequence(
        view.batch.next_observations, layout, learner.burn_in, masks=None
    )[:, -1]
    final_target = learner.target_model.encode_masked_sequence(
        view.batch.next_observations, layout, learner.burn_in, masks=None
    )[:, -1]
    return update_helpers._EncodedFeatures(online, target, final_online, final_target)


@pytest.mark.parametrize("kind", ["scalar", "qr"])
def test_single_step_optimization_preserves_update_and_gradients(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(17)
    learner = DiscreteValueLearner(_value_model(kind))
    learner.setup({"seed": 17})
    state = deepcopy(learner.state_dict())
    reference = DiscreteValueLearner(_value_model(kind))
    reference.setup({"seed": 17})
    reference.load_state_dict(state)
    batch = _batch()
    actual, _ = learner.update(batch)
    monkeypatch.setattr(update_helpers, "_encoded_features", _legacy_features)
    expected, _ = reference.update(batch)

    assert actual["loss/total"] == expected["loss/total"]
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, reference.model.state_dict()[name], rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(
        learner.model.parameters(), reference.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)


def test_single_step_uses_three_encoder_calls_without_bootstrap_autograd() -> None:
    learner = DiscreteValueLearner(_value_model("qr"))
    learner.setup({"seed": 17})
    online: list[bool] = []
    target: list[bool] = []
    first = learner.model.encoder.register_forward_hook(
        lambda *_: online.append(torch.is_grad_enabled())
    )
    second = learner.target_model.encoder.register_forward_hook(
        lambda *_: target.append(torch.is_grad_enabled())
    )
    try:
        view = ValueBatchView.from_batch(learner._batch(_batch()))
        current, online_next, target_next = update_helpers.features(
            learner, update_helpers.FeatureInputs(view, [0])
        )
    finally:
        first.remove()
        second.remove()
    assert online == [True, False]
    assert target == [False]
    assert current.requires_grad
    assert not online_next.requires_grad
    assert not target_next.requires_grad


def test_hard_gradient_clipping_metrics_match_actual_preclip_norm() -> None:
    learner = DiscreteValueLearner(_value_model("qr"), gradient_clip_norm=0.001)
    learner.setup({"seed": 17})
    metrics, _ = learner.update(_batch())
    assert metrics["debug/gradient_clipped_fraction"] == 1.0
    assert metrics["debug/gradient_clip_coefficient"] == pytest.approx(
        0.001 / (metrics["gradients/norm"] + 1e-6)
    )
    norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in learner.model.parameters() if p.grad is not None])
    )
    assert float(norm) <= 0.00101
