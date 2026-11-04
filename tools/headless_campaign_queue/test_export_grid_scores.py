import pytest
import torch

from tools.headless_campaign_queue.export_grid_scores import pair_scores
from tools.headless_campaign_queue.policy_stability import analyze


def test_real_tensor_export_feeds_analysis() -> None:
    before = torch.tensor([[1.0, 0.5], [2.0, 1.0]])
    after = torch.tensor([[0.4, 1.0], [3.0, 1.0]])
    paired = pair_scores(before, after)
    assert paired["actions"] == ["action-0", "action-1"]
    assert analyze(paired)["changed_greedy_action_fraction"] == 0.5


def test_misaligned_or_nonfinite_arrays_fail() -> None:
    with pytest.raises(ValueError, match="equally shaped"):
        pair_scores(torch.zeros(2, 3), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="finite"):
        pair_scores(torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 2))
