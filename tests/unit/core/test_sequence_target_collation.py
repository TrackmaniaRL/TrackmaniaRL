"""Sequence bootstrap histories honor the feature pipeline's collation."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.unit.core._replay_sampler_support import _store
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay import PrioritizedSampler, SequenceSampler
from trackmaniarl.core.replay.sequence_sampler import _collate_target_histories


class _ScalingPipeline:
    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return {
            "_trackmaniarl_batch_collated": True,
            "observations": {
                "scaled": torch.tensor([item.observation * 10 for item in transitions])
            },
            "next_observations": {
                "scaled": torch.tensor([item.next_observation * 10 for item in transitions])
            },
            "actions": torch.tensor([item.action for item in transitions]),
            "rewards": torch.tensor([item.reward for item in transitions]),
            "terminated": torch.tensor([item.terminated for item in transitions]),
            "truncated": torch.tensor([item.truncated for item in transitions]),
        }


@pytest.mark.parametrize("prioritized", [False, True])
def test_sequence_bootstrap_history_uses_pipeline_collation(*, prioritized: bool) -> None:
    store = _store(episodes=1, steps=6)
    pipeline = _ScalingPipeline()
    sampler = (
        PrioritizedSampler(pipeline, seed=7, allow_padded_history=True)
        if prioritized
        else SequenceSampler(pipeline, sequence_length=3, seed=7)
    )
    batch = sampler.sample(store, BatchRequest(batch_size=3, sequence_length=3, n_step=2))
    for row, anchor in enumerate(batch.metadata["priority_transition_ids"]):
        window = list(batch.transition_ids[row * 3 : (row + 1) * 3])
        history = store.get(window)
        horizon = store.get(store.n_step_ids(anchor, 2))
        current = [10 * item.observation for item in history]
        target = ([item.observation for item in history] + [
            item.next_observation for item in horizon
        ])[-3:]
        assert batch.observations["scaled"][row].tolist() == pytest.approx(current)
        assert batch.next_observations["scaled"][row].tolist() == pytest.approx(
            [10 * item for item in target]
        )


def test_target_history_uses_available_transition_metadata() -> None:
    class StepPipeline(_ScalingPipeline):
        def collate(self, transitions: list[Transition]) -> dict[str, Any]:
            output = super().collate(transitions)
            output["observations"]["scaled"] += torch.tensor(
                [100 * int(item.step) for item in transitions]
            )
            return output

    store = _store(episodes=1, steps=5)
    history = store.get([0, 1, 2])
    horizon = store.get([2, 3])
    result = _collate_target_histories(StepPipeline(), [history], [horizon], 3)
    assert result["scaled"].tolist() == [220, 330, 340]
