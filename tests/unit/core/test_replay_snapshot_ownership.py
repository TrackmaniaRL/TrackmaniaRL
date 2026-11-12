from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.pytree import policy_observation_snapshot, tree_snapshot
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


@pytest.mark.parametrize("structured", [False, True])
def test_object_array_with_non_leaf_tensor_survives_replay_roundtrip(*, structured: bool) -> None:
    tensor = torch.tensor([1.0], requires_grad=True) * 2
    array = np.empty(1, dtype=[("value", object), ("count", "i4")] if structured else object)
    objects = array["value"] if structured else array
    objects[0] = {"nested": [tensor]}
    store = InMemoryReplayStore(capacity=2)
    store.append(
        Transition(
            np.array([1.0]),
            0,
            1.0,
            array,
            True,
            False,
            episode_id="episode",
            step=0,
            info={"array": array},
        )
    )
    snapshot = store.state_dict()
    restored = InMemoryReplayStore(capacity=2)
    restored.load_state_dict(snapshot)
    tensor.detach().fill_(9)
    for saved in (snapshot["info"][0]["array"], snapshot["next_overrides"][0]):
        cell = saved["value"][0] if structured else saved[0]
        cell["nested"][0].fill_(8)
    for replay in (store, restored):
        transition = replay.get([0])[0]
        for copied in (transition.next_observation, transition.info["array"]):
            cell = copied["value"][0] if structured else copied[0]
            observed = cell["nested"][0]
            assert observed.item() == 2.0
            assert not observed.requires_grad
            observed.fill_(7)
        fresh = replay.get([0])[0].info["array"]
        cell = fresh["value"][0] if structured else fresh[0]
        assert cell["nested"][0].item() == 2.0


@pytest.mark.parametrize("snapshot", [tree_snapshot, policy_observation_snapshot])
def test_self_referential_object_array_is_owned(snapshot: Callable[[Any], Any]) -> None:
    array = np.empty(1, dtype=object)
    array[0] = array
    copied = snapshot(array)
    assert copied is not array
    assert copied[0] is copied


@pytest.mark.parametrize("snapshot", [tree_snapshot, policy_observation_snapshot])
def test_ndarray_subclass_tensor_metadata_is_owned(snapshot: Callable[[Any], Any]) -> None:
    class MetadataArray(np.ndarray):
        __slots__ = ("__dict__", "slot_tensor")

    tensor = torch.tensor([1.0], requires_grad=True) * 2
    array = np.empty(1, dtype=object).view(MetadataArray)
    array[0] = {"tensor": tensor}
    array.attribute = {"tensor": tensor}
    array.slot_tensor = tensor
    copied = snapshot(array)
    assert isinstance(copied, MetadataArray)
    assert copied.attribute["tensor"] is copied.slot_tensor
    assert copied[0]["tensor"] is copied.slot_tensor
    assert not copied.slot_tensor.requires_grad
    tensor.detach().fill_(4)
    assert copied.slot_tensor.item() == 2.0


def test_policy_snapshot_traverses_numeric_ndarray_subclass_attributes() -> None:
    class MetadataArray(np.ndarray):
        pass

    array = np.array([1.0]).view(MetadataArray)
    array.tensor = torch.tensor([1.0], requires_grad=True) * 2
    copied = policy_observation_snapshot(array)
    assert copied.tensor.item() == 2.0
    assert not copied.tensor.requires_grad
    array.tensor.detach().fill_(3)
    assert copied.tensor.item() == 2.0


@pytest.mark.parametrize("snapshot", [tree_snapshot, policy_observation_snapshot])
def test_object_array_with_structured_scalar_containing_tensor_is_owned(
    snapshot: Callable[[Any], Any],
) -> None:
    tensor = torch.tensor([1.0], requires_grad=True) * 2
    records = np.empty(1, dtype=[("x", object)])
    records["x"][0] = {"nested": [tensor]}
    array = np.empty(1, dtype=object)
    array[0] = records[0]
    copied = snapshot(array)
    copied_tensor = copied[0]["x"]["nested"][0]
    assert not copied_tensor.requires_grad
    tensor.detach().fill_(9)
    assert copied_tensor.item() == 2.0
    copied_tensor.fill_(4)
    assert tensor.item() == 9.0


def test_standalone_structured_scalar_survives_replay_snapshot_and_restore() -> None:
    tensor = torch.tensor([1.0], requires_grad=True) * 2
    records = np.empty(1, dtype=[("x", object)])
    records["x"][0] = {"tensor": tensor}
    store = InMemoryReplayStore(capacity=2)
    store.append(
        Transition(
            0, 0, 1.0, 1, True, False, episode_id="episode", step=0, info={"record": records[0]}
        )
    )
    state = store.state_dict()
    restored = InMemoryReplayStore(capacity=2)
    restored.load_state_dict(state)

    tensor.detach().fill_(9)
    state["info"][0]["record"]["x"]["tensor"].fill_(8)
    for replay in (store, restored):
        actual = replay.get([0])[0].info["record"]["x"]["tensor"]
        assert actual.item() == 2.0
        assert not actual.requires_grad


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
