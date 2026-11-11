from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay import DemoMixSampler, InMemoryReplayStore, UniformSampler


def _transition(value: float) -> Transition:
    return Transition(
        observation=np.array([value], dtype=np.float32),
        action=torch.tensor([value]),
        reward=value,
        next_observation=np.array([value + 1], dtype=np.float32),
        terminated=True,
        truncated=False,
        info={"nested": {"array": np.array([value]), "tensor": torch.tensor([value])}},
        episode_id=str(value),
        step=0,
    )


@pytest.mark.parametrize("sampler_type", [UniformSampler, DemoMixSampler])
def test_collator_cannot_mutate_replay_info(sampler_type: type) -> None:
    class MutatingPipeline(IdentityFeaturePipeline):
        def collate(self, transitions: list[Transition]) -> dict[str, Any]:
            transitions[0].info["nested"]["array"][0] = 7
            transitions[0].info["nested"]["tensor"][0] = 8
            return super().collate(transitions)

    store = InMemoryReplayStore(capacity=1)
    store.append(_transition(1))
    sampler_type(MutatingPipeline()).sample(store, BatchRequest(batch_size=1))
    result = store.get([0])[0]
    assert result.info["nested"]["array"].tolist() == [1]
    assert result.info["nested"]["tensor"].tolist() == [1]


def test_get_returns_owned_transition_and_does_not_mutate_replay() -> None:
    store = InMemoryReplayStore(capacity=2)
    store.append(_transition(1))
    result = store.get([0])[0]
    result.observation[0] = 4
    result.action[0] = 5
    result.next_observation[0] = 6
    result.info["nested"]["array"][0] = 7
    result.info["nested"]["tensor"][0] = 8
    fresh = store.get([0])[0]
    assert fresh.observation.tolist() == [1]
    assert fresh.action.tolist() == [1]
    assert fresh.next_observation.tolist() == [2]
    assert fresh.info["nested"]["array"].tolist() == [1]
    assert fresh.info["nested"]["tensor"].tolist() == [1]


def test_get_preserves_subclass_transition_override_and_copies_it() -> None:
    class CustomStore(InMemoryReplayStore):
        def _transition(self, transition_id: int) -> Transition:
            original = super()._transition(transition_id)
            return Transition(
                observation=original.observation,
                action=original.action,
                reward=123.0,
                next_observation=original.next_observation,
                terminated=original.terminated,
                truncated=original.truncated,
                info=original.info,
                episode_id=original.episode_id,
                step=original.step,
            )

    store = CustomStore(capacity=1)
    store.append(_transition(1))
    value = store.get([0])[0]
    assert value.reward == 123.0
    value.observation[0] = 50
    assert store.get([0])[0].observation.tolist() == [1]


@pytest.mark.parametrize("sampler_type", [UniformSampler, DemoMixSampler])
def test_atomic_selection_and_collation_does_not_hold_lock(sampler_type: type) -> None:
    store = InMemoryReplayStore(capacity=1)
    store.append(_transition(1))
    materializing, proceed, appended = threading.Event(), threading.Event(), threading.Event()
    original = store.materialize_n_step

    def materialize(ids: list[int], request: BatchRequest) -> tuple[list[Transition], list[float]]:
        materializing.set()
        assert proceed.wait(5)
        return original(ids, request)

    store.materialize_n_step = materialize

    class BlockingPipeline(IdentityFeaturePipeline):
        def collate(self, transitions: list[Transition]) -> dict[str, Any]:
            assert appended.wait(5), "collator ran while replay lock was held"
            return super().collate(transitions)

    sampler = sampler_type(BlockingPipeline())
    with ThreadPoolExecutor(max_workers=2) as pool:
        sample = pool.submit(sampler.sample, store, BatchRequest(batch_size=1))
        assert materializing.wait(5)

        def append() -> None:
            store.append(_transition(2))
            appended.set()

        writer = pool.submit(append)
        assert not appended.wait(0.05)
        proceed.set()
        batch = sample.result(timeout=10)
        writer.result(timeout=10)
    assert batch.transition_ids == [0]
    assert batch.observations.tolist() == [[1]]
    assert batch.next_observations.tolist() == [[2]]
