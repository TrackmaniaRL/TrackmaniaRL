"""Rare maneuvers retain weight; supervised fitting changes actual greedy choices."""

import pytest
import torch

from trackmaniarl.experiments.shortcut_pretraining import (
    BehaviorCloningPretrainer,
    balanced_action_weights,
)
from trackmaniarl.models.factory import CompositeValueModelFactory


def test_classes_have_equal_total_weight_and_invalid_labels_are_rejected() -> None:
    actions = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    weights = balanced_action_weights(actions, 3)
    assert weights[0] * 3 == weights[1]
    assert weights[2] == 0
    with pytest.raises(ValueError, match="action space"):
        balanced_action_weights(torch.tensor([3]), 3)
    with pytest.raises(ValueError, match="integer"):
        balanced_action_weights(torch.tensor([0.0]), 3)


def test_supervised_update_fits_rare_action_and_rejects_nonfinite_inputs() -> None:
    torch.manual_seed(17)
    model = CompositeValueModelFactory(
        encoder={
            "class_path": "trackmaniarl.models.encoders:StructuredMlpSensorEncoder",
            "kwargs": {"input_shapes": {"sensor": [2]}, "output_dim": 8, "hidden_dim": 8},
        },
        temporal={
            "class_path": "trackmaniarl.models.temporal:IdentityTemporalCore",
            "kwargs": {"input_dim": 8},
        },
        head={
            "class_path": "trackmaniarl.models.heads:FixedQuantileHead",
            "kwargs": {"config": {"feature_dim": 8, "action_count": 3, "quantile_count": 64}},
        },
        strategy={
            "class_path": "trackmaniarl.models.strategies:FixedQuantileStrategy",
            "kwargs": {"quantile_count": 64},
        },
    ).build()
    observations = {"sensor": torch.tensor([[1.0, 0.0]] * 24 + [[0.0, 1.0]] * 4)}
    actions = torch.tensor([0] * 24 + [2] * 4)
    trainer = BehaviorCloningPretrainer(model, actions, learning_rate=0.01)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for _ in range(80):
        trainer.update(observations, actions)
    model.eval()
    assert torch.equal(trainer.logits(observations).argmax(-1), actions)
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    after = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="nonfinite"):
        trainer.update({"sensor": torch.full((28, 2), float("nan"))}, actions)
    assert trainer.updates == 80
    assert all(torch.equal(after[name], value) for name, value in model.state_dict().items())
