"""Parity checks for the incremental uniform sequence index."""

from __future__ import annotations

from random import Random
from threading import Event, Thread
from typing import Never

import numpy as np
import pytest

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay.sequence_index import SequenceIndex
from trackmaniarl.core.replay.sequence_sampler import SequenceSampler
from trackmaniarl.core.replay.store import InMemoryReplayStore


def _eligible(store: InMemoryReplayStore, length: int, n_step: int) -> set[int]:
    return {
        item
        for item in store.available_ids()
        if store.is_n_step_eligible(item, n_step)
        and len(set(store.history_ids(item, length))) == length
    }


@pytest.mark.parametrize(("length", "n_step"), [(2, 1), (3, 3), (5, 2)])
def test_incremental_index_matches_full_scan_with_interleaved_eviction(
    length: int, n_step: int
) -> None:
    rng = Random(18)
    store = InMemoryReplayStore(capacity=31)
    index = SequenceIndex(store, length, n_step)
    steps = [0, 0, 0]
    episodes = [0, 0, 0]
    for iteration in range(400):
        actor = rng.randrange(3)
        ended = rng.random() < 0.18
        step = steps[actor]
        store.append(
            Transition(
                observation=float(step),
                action=0,
                reward=1.0,
                next_observation=float(step + 1),
                terminated=ended,
                truncated=False,
                episode_id=f"{actor}/{episodes[actor]}",
                step=step,
            )
        )
        steps[actor] = 0 if ended else step + 1
        episodes[actor] += int(ended)
        # Multiple mutations between samples and skipped history are both common.
        if iteration % 4 == 0:
            index.synchronize()
            expected = _eligible(store, length, n_step)
            assert {item for item in index.ids if item is not None} == expected
            assert int(index.tree.total) == len(expected)
            if expected:
                assert set(index.draw(rng, len(expected))) == expected


def test_index_rebuild_has_same_slot_order_after_restore() -> None:
    store = InMemoryReplayStore(capacity=7)
    index = SequenceIndex(store, 2, 1)
    for step in range(13):
        store.append(
            Transition(
                observation=float(step),
                action=0,
                reward=1.0,
                next_observation=float(step + 1),
                terminated=False,
                truncated=False,
                episode_id="episode",
                step=step,
            )
        )
        index.synchronize()
    rebuilt = SequenceIndex(store, 2, 1)
    rebuilt.synchronize()
    assert index.draw(Random(42), 3) == rebuilt.draw(Random(42), 3)


def test_external_store_subclass_does_not_assume_modulo_slot_contract() -> None:
    class CustomStore(InMemoryReplayStore):
        def sampling_transaction(self) -> Never:
            raise AssertionError("custom store transaction must not be assumed reentrant")

    store = CustomStore(capacity=8)
    for step in range(4):
        store.append(
            Transition(
                observation=float(step),
                action=0,
                reward=1.0,
                next_observation=float(step + 1),
                terminated=False,
                truncated=False,
                episode_id="episode",
                step=step,
            )
        )
    sampler = SequenceSampler(IdentityFeaturePipeline(), sequence_length=2)
    sampler.sample(store, BatchRequest(batch_size=1))
    assert sampler._incremental_index is None


def test_user_collation_runs_outside_lock_with_owned_replay_snapshot() -> None:
    store = InMemoryReplayStore(capacity=3)

    def add(step: int) -> None:
        store.append(
            Transition(
                observation=np.array([step], dtype=np.float32),
                action=0,
                reward=1.0,
                next_observation=np.array([step + 1], dtype=np.float32),
                terminated=False,
                truncated=False,
                episode_id="episode",
                step=step,
            )
        )

    for step in range(3):
        add(step)
    entered, finished = Event(), Event()

    class WaitingPipeline(IdentityFeaturePipeline):
        def collate(self, transitions: list[Transition]) -> dict:
            entered.set()
            if not finished.wait(3):
                raise TimeoutError("producer blocked by replay lock during user collation")
            return super().collate(transitions)

    def producer() -> None:
        if entered.wait(3):
            for step in range(3, 6):
                add(step)
            finished.set()

    thread = Thread(target=producer)
    thread.start()
    try:
        batch = SequenceSampler(WaitingPipeline(), 2).sample(store, BatchRequest(batch_size=1))
        assert max(batch.observations.flatten().tolist()) < 3
        assert max(batch.next_observations.flatten().tolist()) <= 3
    finally:
        thread.join(timeout=4)
    assert not thread.is_alive()
