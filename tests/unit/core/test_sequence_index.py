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


class _NoForwardResolver:
    def __init__(self) -> None:
        self.store = InMemoryReplayStore(capacity=8)

    def __getattr__(self, name: str) -> object:
        if name == "n_step_ids":
            raise AttributeError(name)
        return getattr(self.store, name)


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


def test_index_rejects_history_linked_back_across_terminal_boundary() -> None:
    store = InMemoryReplayStore(capacity=3)
    for episode, step, terminated in (("x", 9, True), ("x", 10, False)):
        store.append(
            Transition(
                float(step),
                0,
                1.0,
                float(step + 1),
                terminated,
                False,
                episode_id=episode,
                step=step,
            )
        )
    index = SequenceIndex(store, 2, 1)
    index.synchronize()
    assert index.tree.total == 0
    for episode in ("y", "z"):
        store.append(Transition(0.0, 0, 1.0, 1.0, False, False, episode_id=episode, step=0))
    index.synchronize()
    assert index.tree.total == 0
    with pytest.raises(RuntimeError, match="Need 1 valid sequences"):
        index.draw(Random(0), 1)


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


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_linked_external_store_rejects_history_crossing_episode_boundary(boundary: str) -> None:
    class CustomStore(InMemoryReplayStore):
        pass

    store = CustomStore(capacity=3)
    store.append(
        Transition(
            9.0,
            0,
            1.0,
            10.0,
            boundary == "terminated",
            boundary == "truncated",
            episode_id="episode",
            step=9,
        )
    )
    store.append(Transition(10.0, 0, 1.0, 11.0, False, False, episode_id="episode", step=10))
    with pytest.raises(RuntimeError, match="Need 1 valid sequences"):
        SequenceSampler(IdentityFeaturePipeline(), 2).sample(store, BatchRequest(batch_size=1))


def test_linked_external_store_keeps_interleaved_episode_history() -> None:
    class CustomStore(InMemoryReplayStore):
        pass

    store = CustomStore(capacity=3)
    for episode, step in (("a", 0), ("b", 0), ("a", 1)):
        store.append(
            Transition(
                float(step), 0, 1.0, float(step + 1), False, False, episode_id=episode, step=step
            )
        )
    batch = SequenceSampler(IdentityFeaturePipeline(), 2).sample(store, BatchRequest(batch_size=1))
    assert batch.transition_ids == [0, 2]


@pytest.mark.parametrize("interleaved", [False, True])
def test_linked_store_without_horizon_resolver_checks_numeric_fallback(
    *, interleaved: bool
) -> None:
    store = _NoForwardResolver()
    for step in range(3):
        store.store.append(
            Transition(
                float(step), 0, 1.0, float(step + 1), False, False, episode_id="a", step=step
            )
        )
        if interleaved:
            store.store.append(
                Transition(
                    float(step), 0, 1.0, float(step + 1), False, False, episode_id="b", step=step
                )
            )
    sampler = SequenceSampler(IdentityFeaturePipeline(), 2)
    request = BatchRequest(batch_size=1, n_step=2)
    if interleaved:
        with pytest.raises(RuntimeError, match="needs n_step_ids"):
            sampler.sample(store, request)  # type: ignore[arg-type]
    else:
        batch = sampler.sample(store, request)  # type: ignore[arg-type]
        assert batch.transition_ids == [0, 1]


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_linked_numeric_horizon_can_end_early_at_boundary(boundary: str) -> None:
    store = _NoForwardResolver()
    store.store.append(Transition(0.0, 0, 1.0, 1.0, False, False, episode_id="a", step=0))
    store.store.append(
        Transition(
            1.0,
            0,
            2.0,
            2.0,
            boundary == "terminated",
            boundary == "truncated",
            episode_id="a",
            step=1,
        )
    )
    batch = SequenceSampler(IdentityFeaturePipeline(), 2).sample(
        store,
        BatchRequest(batch_size=1, n_step=3, gamma=0.5),  # type: ignore[arg-type]
    )
    assert batch.transition_ids == [0, 1]
    assert batch.rewards[0, -1].item() == pytest.approx(2.0)
    expected_discount = 0.0 if boundary == "terminated" else 0.5
    assert batch.bootstrap_discounts[0, -1].item() == pytest.approx(expected_discount)


def test_linked_interleaved_horizons_follow_resolver() -> None:
    class CustomStore(InMemoryReplayStore):
        pass

    store = CustomStore(capacity=6)
    for step in range(3):
        for episode in ("a", "b"):
            store.append(
                Transition(
                    float(step),
                    0,
                    float(step + 1),
                    float(step + 1),
                    False,
                    False,
                    episode_id=episode,
                    step=step,
                )
            )
    batch = SequenceSampler(IdentityFeaturePipeline(), 2).sample(
        store, BatchRequest(batch_size=2, n_step=2, gamma=0.5)
    )
    assert {tuple(batch.transition_ids[i : i + 2]) for i in (0, 2)} == {(0, 2), (1, 3)}
    assert batch.rewards[:, -1].tolist() == pytest.approx([3.5, 3.5])
    assert batch.bootstrap_discounts[:, -1].tolist() == pytest.approx([0.25, 0.25])


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
