"""Replay caches belong to one store, even when local IDs and revisions overlap."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.unit.core._replay_sampler_support import _BasicReplayStore
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, PrioritizedSampler


def _transition(step: int) -> Transition:
    return Transition(
        observation=float(step),
        action=0,
        reward=1.0,
        next_observation=float(step + 1),
        terminated=True,
        truncated=False,
        episode_id=f"episode-{step}",
        step=0,
    )


def test_switching_store_rechecks_n_step_eligibility() -> None:
    completed = InMemoryReplayStore(capacity=4)
    live = InMemoryReplayStore(capacity=4)
    completed.append(_transition(0))
    live.append(replace(_transition(0), terminated=False))
    sampler = PrioritizedSampler(IdentityFeaturePipeline())
    request = BatchRequest(batch_size=1, n_step=3)
    sampler.sample(completed, request)

    with pytest.raises(RuntimeError, match="replay has 0"):
        sampler.sample(live, request)


@pytest.mark.parametrize("store_factory", [InMemoryReplayStore, _BasicReplayStore])
def test_switching_store_does_not_reuse_priorities(store_factory: type) -> None:
    source = store_factory()
    target = store_factory()
    for step in range(4):
        source.append(_transition(step))
        target.append(_transition(step))
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), alpha=1.0, beta=1.0, seed=3)
    request = BatchRequest(batch_size=4)
    sampler.sample(source, request)
    sampler.update_priorities(PriorityUpdate([0, 1, 2, 3], [1.0, 3.0, 9.0, 27.0]))

    batch = sampler.sample(target, request)

    assert torch.equal(batch.importance_weights, torch.ones(4))


def test_switching_from_incremental_to_fallback_keeps_priority_updates_effective() -> None:
    source = InMemoryReplayStore(capacity=4)
    target = _BasicReplayStore()
    for step in range(4):
        source.append(_transition(step))
        target.append(_transition(step))
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), alpha=1.0, seed=3)
    request = BatchRequest(batch_size=4)
    sampler.sample(source, request)
    sampler.sample(target, request)

    sampler.update_priorities(PriorityUpdate([0], [100.0]))

    assert sampler.state_dict()["fallback_priorities"][0] == pytest.approx(100.0 + 1e-6)
