from __future__ import annotations

from collections import UserDict, namedtuple
from dataclasses import dataclass
from threading import Event
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.core.collector import (
    EpisodeCollector,
    FixedStepRolloutCollector,
    RolloutCollectionConfig,
)
from trackmaniarl.core.data import Transition
from trackmaniarl.core.pytree import policy_observation_snapshot
from trackmaniarl.core.replay.store import InMemoryReplayStore
from trackmaniarl.distributed.actor_collection import (
    CollectionContext,
    TrainingEpisode,
    _finish_training_episode,
    _run_training_episode,
)
from trackmaniarl.distributed.actor_metrics import EpisodeMetrics


class ReusingEnvironment:
    def __init__(self, *, tensor: bool, stop: Event | None = None) -> None:
        self.buffer = torch.zeros(1) if tensor else np.zeros(1, dtype=np.float32)
        self.stop = stop
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.buffer[...] = 0
        self.steps = 0
        return self.buffer, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.buffer[...] += 1
        self.steps += 1
        if self.stop is not None and self.steps == 2:
            self.stop.set()
        return self.buffer, 1.0, False, False, {}


class IdentityPipeline:
    def transform_observation(self, value: Any) -> Any:
        return value


class ZeroPolicy:
    def act(self, observation: Any) -> int:
        return 0


class MutatingObservationPolicy:
    def act(self, observation: Any) -> Any:
        observation[...] += 10
        return observation


class ReusingActionPolicy:
    def __init__(self, *, tensor: bool) -> None:
        self.action = torch.zeros(1) if tensor else np.zeros(1, dtype=np.float32)
        self.calls = 0

    def act(self, observation: Any) -> Any:
        self.calls += 1
        self.action[...] = self.calls
        return self.action


class MutatingActionEnvironment(ReusingEnvironment):
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action[...] = -10
        return super().step(action)


class ReusingInfoPolicy:
    def __init__(self, *, tensor: bool) -> None:
        self.info_buffer = torch.zeros(1) if tensor else np.zeros(1, dtype=np.float32)
        self.calls = 0

    def act_with_info(self, observation: Any) -> tuple[int, dict[str, Any]]:
        self.calls += 1
        self.info_buffer[...] = self.calls
        return 0, {"policy_buffer": self.info_buffer}


class ReusingInfoEnvironment(ReusingEnvironment):
    def __init__(
        self, *, tensor: bool, policy: ReusingInfoPolicy, stop: Event | None = None
    ) -> None:
        super().__init__(tensor=tensor, stop=stop)
        self.info_buffer = torch.zeros(1) if tensor else np.zeros(1, dtype=np.float32)
        self.policy = policy

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.policy.info_buffer[...] = -1
        observation, reward, terminated, truncated, _ = super().step(action)
        self.info_buffer[...] = self.steps
        return observation, reward, terminated, truncated, {"environment_buffer": self.info_buffer}


class ReferenceStore:
    """A valid minimal store that keeps transitions without copying their leaves."""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def append(self, transition: Transition) -> int:
        self.transitions.append(transition)
        return len(self.transitions) - 1


def test_policy_input_snapshot_preserves_custom_containers_and_nonleaf_tensors() -> None:
    observation_type = namedtuple("Observation", ["speed"])
    original_tensor = torch.ones(1, requires_grad=True) * 2
    original = UserDict({"observation": observation_type(original_tensor)})
    copied = policy_observation_snapshot(original)
    assert isinstance(copied, UserDict)
    assert isinstance(copied["observation"], observation_type)
    copied["observation"].speed[...] = 99
    assert float(original_tensor.detach()[0]) == 2.0


def test_policy_input_snapshot_preserves_dataclass_with_nonleaf_tensor() -> None:
    @dataclass(slots=True)
    class Observation:
        speed: torch.Tensor

    original = Observation(torch.ones(1, requires_grad=True) * 2)
    copied = policy_observation_snapshot(original)
    assert isinstance(copied, Observation)
    copied.speed[...] = 99
    assert float(original.speed.detach()[0]) == 2.0


@pytest.mark.parametrize("slots", [False, True])
def test_policy_input_snapshot_preserves_object_with_nonleaf_tensor(slots: bool) -> None:  # noqa: FBT001
    class OrdinaryObservation:
        def __init__(self, speed: torch.Tensor) -> None:
            self.speed = speed

    class SlottedObservation:
        __slots__ = ("speed",)

        def __init__(self, speed: torch.Tensor) -> None:
            self.speed = speed

    observation_type = SlottedObservation if slots else OrdinaryObservation
    original = observation_type(torch.ones(1, requires_grad=True) * 2)
    copied = policy_observation_snapshot(original)
    assert type(copied) is observation_type
    copied.speed[...] = 99
    assert float(original.speed.detach()[0]) == 2.0


@pytest.mark.parametrize("readonly", [False, True])
def test_policy_input_snapshot_copies_memoryview(readonly: bool) -> None:  # noqa: FBT001
    data = bytearray(b"abc")
    original = memoryview(bytes(data) if readonly else data)
    copied = policy_observation_snapshot(original)
    assert isinstance(copied, memoryview)
    assert copied.readonly == readonly
    assert copied.tobytes() == b"abc"
    if not readonly:
        copied[0] = ord("z")
        assert original.tobytes() == b"abc"


def test_policy_input_snapshot_copies_readonly_mapping() -> None:
    original = MappingProxyType({"speed": np.array([1.0])})
    copied = policy_observation_snapshot(original)
    assert isinstance(copied, MappingProxyType)
    copied["speed"][...] = 2.0
    assert float(original["speed"][0]) == 1.0


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_policy_snapshot_preserves_references_but_not_distinct_view_aliases(backend: str) -> None:
    base = np.arange(3, dtype=np.float32) if backend == "numpy" else torch.arange(3.0)
    copied = policy_observation_snapshot({"base": base, "same": base, "view": base[1:]})
    assert copied["base"] is copied["same"]
    copied["base"][1] = 99
    assert float(copied["same"][1]) == 99.0
    assert float(copied["view"][0]) == 1.0
    assert float(base[1]) == 1.0


def _assert_transition_values(transitions: list[Transition]) -> None:
    assert len(transitions) == 2
    for index, item in enumerate(transitions):
        assert float(item.observation[0]) == index
        assert float(item.next_observation[0]) == index + 1


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_episode_collector_snapshots_before_environment_mutates_buffer(kind: str) -> None:
    store = InMemoryReplayStore(10)
    EpisodeCollector(store, IdentityPipeline(), ZeroPolicy()).collect(
        ReusingEnvironment(tensor=kind == "torch"), "episode", 2
    )
    _assert_transition_values(store.get(store.available_ids()))


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_rollout_collector_snapshots_before_environment_mutates_buffer(kind: str) -> None:
    store = InMemoryReplayStore(10)
    collector = FixedStepRolloutCollector(
        store,
        IdentityPipeline(),
        RolloutCollectionConfig(ZeroPolicy(), ReusingEnvironment(tensor=kind == "torch"), 5),
    )
    collector.collect(2, "rollout")
    _assert_transition_values(store.get(store.available_ids()))


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("collector_kind", ["episode", "rollout"])
def test_custom_reference_store_keeps_next_observations(kind: str, collector_kind: str) -> None:
    store = ReferenceStore()
    environment = ReusingEnvironment(tensor=kind == "torch")
    if collector_kind == "episode":
        EpisodeCollector(store, IdentityPipeline(), ZeroPolicy()).collect(environment, "episode", 2)
    else:
        FixedStepRolloutCollector(
            store, IdentityPipeline(), RolloutCollectionConfig(ZeroPolicy(), environment, 5)
        ).collect(2, "rollout")
    environment.buffer[...] = 99
    _assert_transition_values(store.transitions)


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("collector_kind", ["episode", "rollout"])
def test_local_collectors_keep_issued_actions_in_store_and_artifact(
    kind: str, collector_kind: str
) -> None:
    store = ReferenceStore()
    environment = MutatingActionEnvironment(tensor=kind == "torch")
    policy = ReusingActionPolicy(tensor=kind == "torch")
    if collector_kind == "episode":
        result = EpisodeCollector(store, IdentityPipeline(), policy).collect(
            environment, "episode", 2
        )
    else:
        result = FixedStepRolloutCollector(
            store, IdentityPipeline(), RolloutCollectionConfig(policy, environment, 5)
        ).collect(2, "rollout")
    assert [float(t.action[0]) for t in store.transitions] == [1.0, 2.0]
    assert [float(action[0]) for action in result.artifact.actions] == [1.0, 2.0]
    result.artifact.actions[0][...] = 77
    assert float(store.transitions[0].action[0]) == 1.0


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_distributed_collection_keeps_issued_action_through_stop(kind: str) -> None:
    stop = Event()
    sent = []
    policy = ReusingActionPolicy(tensor=kind == "torch")
    runtime = SimpleNamespace(
        stop=stop,
        spec=SimpleNamespace(training=SimpleNamespace(max_episode_steps=5)),
        _should_flush=lambda *_: False,
        _policy=lambda: (policy, 0.0, 0),
        _spool=sent.append,
        _summary=lambda reward, info, steps: {},
    )
    environment = MutatingActionEnvironment(tensor=kind == "torch", stop=stop)
    observation, _ = environment.reset()
    state = TrainingEpisode(
        observation, policy, 0.0, 0, "episode", EpisodeMetrics.from_policy(policy)
    )
    context = CollectionContext(runtime, environment, IdentityPipeline())
    _run_training_episode(context, state)
    _finish_training_episode(context, state)
    assert [float(t.action[0]) for t in sent[0].transitions] == [1.0, 2.0]


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("collector_kind", ["episode", "rollout"])
def test_local_collectors_snapshot_reused_policy_and_environment_info(
    kind: str, collector_kind: str
) -> None:
    tensor = kind == "torch"
    store = ReferenceStore()
    policy = ReusingInfoPolicy(tensor=tensor)
    environment = ReusingInfoEnvironment(tensor=tensor, policy=policy)
    if collector_kind == "episode":
        result = EpisodeCollector(store, IdentityPipeline(), policy).collect(
            environment, "episode", 2
        )
    else:
        result = FixedStepRolloutCollector(
            store, IdentityPipeline(), RolloutCollectionConfig(policy, environment, 5)
        ).collect(2, "rollout")
    assert [float(t.info["policy_buffer"][0]) for t in store.transitions] == [1.0, 2.0]
    assert [float(t.info["environment_buffer"][0]) for t in store.transitions] == [1.0, 2.0]
    assert [float(t["environment_buffer"][0]) for t in result.artifact.telemetry] == [1.0, 2.0]
    result.artifact.telemetry[0]["environment_buffer"][...] = 99
    assert float(store.transitions[0].info["environment_buffer"][0]) == 1.0


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_distributed_collection_snapshots_reused_policy_and_environment_info(kind: str) -> None:
    tensor = kind == "torch"
    stop = Event()
    sent = []
    policy = ReusingInfoPolicy(tensor=tensor)
    runtime = SimpleNamespace(
        stop=stop,
        spec=SimpleNamespace(training=SimpleNamespace(max_episode_steps=5)),
        _should_flush=lambda *_: False,
        _policy=lambda: (policy, 0.0, 0),
        _spool=sent.append,
        _summary=lambda reward, info, steps: {},
    )
    environment = ReusingInfoEnvironment(tensor=tensor, policy=policy, stop=stop)
    observation, _ = environment.reset()
    state = TrainingEpisode(
        observation, policy, 0.0, 0, "episode", EpisodeMetrics.from_policy(policy)
    )
    context = CollectionContext(runtime, environment, IdentityPipeline())
    _run_training_episode(context, state)
    _finish_training_episode(context, state)
    assert [float(t.info["policy_buffer"][0]) for t in sent[0].transitions] == [1.0, 2.0]
    assert [float(t.info["environment_buffer"][0]) for t in sent[0].transitions] == [1.0, 2.0]


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_distributed_collection_snapshots_before_step_and_flushes_on_stop(kind: str) -> None:
    stop = Event()
    sent = []
    runtime = SimpleNamespace(
        stop=stop,
        spec=SimpleNamespace(training=SimpleNamespace(max_episode_steps=5)),
        _should_flush=lambda *_: False,
        _policy=lambda: (policy, 0.0, 0),
        _spool=sent.append,
        _summary=lambda reward, info, steps: {},
    )
    environment = ReusingEnvironment(tensor=kind == "torch", stop=stop)
    policy = ZeroPolicy()
    observation, _ = environment.reset()
    state = TrainingEpisode(
        observation, policy, 0.0, 0, "episode", EpisodeMetrics.from_policy(policy)
    )
    context = CollectionContext(runtime, environment, IdentityPipeline())
    _run_training_episode(context, state)
    _finish_training_episode(context, state)
    assert stop.is_set()
    _assert_transition_values(sent[0].transitions)
    assert sent[0].transitions[-1].truncated


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("collector_kind", ["episode", "rollout", "distributed"])
def test_policy_cannot_mutate_stored_or_environment_observations(
    kind: str, collector_kind: str
) -> None:
    stop = Event()
    environment = ReusingEnvironment(tensor=kind == "torch", stop=stop)
    policy = MutatingObservationPolicy()
    if collector_kind == "distributed":
        sent = []
        runtime = SimpleNamespace(
            stop=stop,
            spec=SimpleNamespace(training=SimpleNamespace(max_episode_steps=5)),
            _should_flush=lambda *_: False,
            _policy=lambda: (policy, 0.0, 0),
            _spool=sent.append,
            _summary=lambda reward, info, steps: {},
        )
        observation, _ = environment.reset()
        state = TrainingEpisode(
            observation, policy, 0.0, 0, "episode", EpisodeMetrics.from_policy(policy)
        )
        context = CollectionContext(runtime, environment, IdentityPipeline())
        _run_training_episode(context, state)
        _finish_training_episode(context, state)
        transitions = sent[0].transitions
    else:
        store = ReferenceStore()
        if collector_kind == "episode":
            EpisodeCollector(store, IdentityPipeline(), policy).collect(environment, "episode", 2)
        else:
            FixedStepRolloutCollector(
                store, IdentityPipeline(), RolloutCollectionConfig(policy, environment, 5)
            ).collect(2, "rollout")
        transitions = store.transitions
    _assert_transition_values(transitions)
    assert [float(t.action[0]) for t in transitions] == [10.0, 11.0]
