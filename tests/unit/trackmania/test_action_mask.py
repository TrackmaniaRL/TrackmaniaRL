from __future__ import annotations

import pytest
import torch

from trackmaniarl.core.contracts import ActionSelectionRequest, PolicyMode
from trackmaniarl.trackmania.actions import TrackmaniaActionSelector, select_brake_tap_actions


@pytest.mark.parametrize("global_probability", [0.0, 1.0])
def test_exploration_respects_each_rows_policy_mask(global_probability: float) -> None:
    torch.manual_seed(7)
    selector = TrackmaniaActionSelector({"global_exploration_probability": global_probability})
    values = torch.full((2048, 78), -torch.inf)
    # Arbitrary exclusions also cover invalid neighbors, not just brake pulses.
    values[:1024, [3, 39]] = 0
    values[1024:, [15, 75]] = 0
    chosen = selector.select(
        values, values.argmax(-1), ActionSelectionRequest(PolicyMode.ONLINE, 1.0)
    )
    assert torch.isfinite(values.gather(-1, chosen[:, None])).all()


def test_pulse_actions_are_never_explored_when_disabled() -> None:
    _, table = select_brake_tap_actions(None)
    excluded = torch.tensor([action[1] < 0 for action in table])
    values = torch.zeros(10000, 78)
    values[:, excluded] = -torch.inf
    selector = TrackmaniaActionSelector()
    selected = selector.select(
        values, values.argmax(-1), ActionSelectionRequest(PolicyMode.ONLINE, 1.0)
    )
    assert not excluded[selected].any()


def test_compact_exploration_obeys_mask_and_rejects_empty_support() -> None:
    selector = TrackmaniaActionSelector({"action_ids": (3, 39, 75)})
    values = torch.tensor([[-torch.inf, 0.0, -torch.inf]])
    request = ActionSelectionRequest(PolicyMode.ONLINE, 1.0)
    assert selector.select(values, values.argmax(-1), request).item() == 1
    with pytest.raises(ValueError, match="allowed action"):
        selector.select(torch.full_like(values, -torch.inf), values.argmax(-1), request)
