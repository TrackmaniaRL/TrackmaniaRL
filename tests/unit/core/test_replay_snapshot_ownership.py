from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay import InMemoryReplayStore
from trackmaniarl.core.replay.sequence_sampler import SequenceSampler


def test_snapshot_and_restored_replay_own_nested_metadata_and_overrides() -> None:
    store = InMemoryReplayStore(capacity=4)
    store.append(
        Transition(
            observation=np.array([1.0]),
            action=0,
            reward=1.0,
            next_observation={"pixels": np.array([2.0])},
            terminated=True,
            truncated=False,
            episode_id="episode",
            step=0,
            info={"nested": {"values": [np.array([3.0])]}},
        )
    )
    snapshot = store.state_dict()
    restored = InMemoryReplayStore(capacity=4)
    restored.load_state_dict(snapshot)

    snapshot["next_overrides"][0]["pixels"][0] = 20.0
    snapshot["info"][0]["nested"]["values"][0][0] = 30.0
    for replay in (store, restored):
        transition = replay.get([0])[0]
        assert transition.next_observation["pixels"][0] == 2.0
        assert transition.info["nested"]["values"][0][0] == 3.0

    fresh = restored.state_dict()
    fresh["info"][0]["nested"]["values"].append(4)
    assert len(restored.get([0])[0].info["nested"]["values"]) == 1


def test_object_array_elements_are_owned_on_append_snapshot_and_restore() -> None:
    source = np.empty(1, dtype=object)
    source[0] = {"values": [3]}
    next_source = np.empty(1, dtype=object)
    next_source[0] = [2]
    store = InMemoryReplayStore(capacity=4)
    store.append(
        Transition(
            observation=np.array([1.0]),
            action=0,
            reward=1.0,
            next_observation=next_source,
            terminated=True,
            truncated=False,
            episode_id="episode",
            step=0,
            info={"objects": source},
        )
    )
    source[0]["values"].append(4)
    next_source[0].append(4)
    snapshot = store.state_dict()
    restored = InMemoryReplayStore(capacity=4)
    restored.load_state_dict(snapshot)
    snapshot["info"][0]["objects"][0]["values"].append(5)
    snapshot["next_overrides"][0][0].append(5)
    for replay in (store, restored):
        assert replay.get([0])[0].info["objects"][0]["values"] == [3]
        assert replay.get([0])[0].next_observation[0] == [2]
    restored_value = restored.get([0])[0].info["objects"]
    restored_value[0]["values"].append(6)
    assert restored.get([0])[0].info["objects"][0]["values"] == [3]


def test_append_sample_snapshot_and_restore_interleave() -> None:
    store = InMemoryReplayStore(capacity=16)

    def append(step: int) -> None:
        store.append(
            Transition(
                float(step), 0, 1.0, float(step + 1), False, False, episode_id="episode", step=step
            )
        )

    for step in range(4):
        append(step)
    barrier = Barrier(2)

    def producer() -> None:
        for step in range(4, 24):
            barrier.wait(timeout=5)
            append(step)
            barrier.wait(timeout=5)

    def consumer() -> None:
        sampler = SequenceSampler(IdentityFeaturePipeline(), sequence_length=2, seed=7)
        for _ in range(20):
            barrier.wait(timeout=5)
            sampler.sample(store, BatchRequest(batch_size=1))
            state = store.state_dict()
            restored = InMemoryReplayStore(capacity=16)
            restored.load_state_dict(state)
            ids = restored.available_ids()
            assert ids == list(range(state["next_index"] - state["size"], state["next_index"]))
            assert [item.observation for item in restored.get(ids)] == [float(i) for i in ids]
            SequenceSampler(IdentityFeaturePipeline(), 2).sample(
                restored, BatchRequest(batch_size=1)
            )
            barrier.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        producer_future = pool.submit(producer)
        consumer_future = pool.submit(consumer)
        producer_future.result(timeout=30)
        consumer_future.result(timeout=30)
