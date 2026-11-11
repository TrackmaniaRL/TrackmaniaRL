from __future__ import annotations

from threading import Event
from types import SimpleNamespace
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


class ReferenceStore:
    """A valid minimal store that keeps transitions without copying their leaves."""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def append(self, transition: Transition) -> int:
        self.transitions.append(transition)
        return len(self.transitions) - 1


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
