"""The TrackMania collector must work against a minimal Gymnasium-like adapter."""

from __future__ import annotations

import gc
from typing import Any

import pytest

from trackmaniarl.core.builtins import IdentityFeaturePipeline, ZeroPolicy
from trackmaniarl.core.collector import (
    FixedStepRolloutCollector,
    RolloutCollectionConfig,
)
from trackmaniarl.core.replay import InMemoryReplayStore


class FakeTrackmania:
    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, float], dict[str, Any]]:
        del seed
        self.step_index = 0
        return {"speed": 0.0}, {}

    def step(self, action: Any) -> tuple[dict[str, float], float, bool, bool, dict[str, Any]]:
        del action
        self.step_index += 1
        return (
            {"speed": float(self.step_index)},
            1.0,
            self.step_index == 2,
            False,
            {"observation_ref": f"frame-{self.step_index}"},
        )


def test_fixed_rollout_continues_episode_across_collection_boundaries() -> None:
    environment = FakeTrackmania()
    store = InMemoryReplayStore()
    collector = FixedStepRolloutCollector(
        store,
        IdentityFeaturePipeline(),
        RolloutCollectionConfig(ZeroPolicy(), environment, max_episode_steps=10),
    )

    first = collector.collect(1, "rollout-0")
    second = collector.collect(2, "rollout-1")
    transitions = store.get(store.available_ids())
    assert first.transitions == 1
    assert second.transitions == 2
    episode_ids = ["episode-00000000", "episode-00000000", "episode-00000001"]
    assert [item.episode_id for item in transitions] == episode_ids
    assert [item.step for item in transitions] == [0, 1, 0]


def test_realtime_rollout_truncates_then_releases_before_learning() -> None:
    environment = FakeTrackmania()
    released = []
    environment.release_controls = lambda: released.append(True)
    store = InMemoryReplayStore()
    collector = FixedStepRolloutCollector(
        store,
        IdentityFeaturePipeline(),
        RolloutCollectionConfig(
            ZeroPolicy(), environment, max_episode_steps=10, reset_at_rollout_end=True
        ),
    )
    collector.collect(1, "first")
    collector.collect(1, "second")
    transitions = store.get(store.available_ids())
    assert all(item.truncated and not item.terminated for item in transitions)
    assert all(item.info["termination_reason"] == "rollout_boundary" for item in transitions)
    assert [item.episode_id for item in transitions] == ["episode-00000000", "episode-00000001"]
    assert [item.step for item in transitions] == [0, 0]
    assert released == [True, True]


def test_rollout_warms_policy_and_collects_cycles_before_reset() -> None:
    events: list[str] = []

    class Pipeline(IdentityFeaturePipeline):
        def synthetic_observation(self) -> dict[str, float]:
            return {"speed": 0.0}

    class Policy(ZeroPolicy):
        def warm_up(self, observation: Any) -> None:
            del observation
            events.append("warm")

    class Environment(FakeTrackmania):
        def reset(self, *, seed: int | None = None) -> tuple[dict[str, float], dict[str, Any]]:
            events.append("reset")
            assert not gc.isenabled()
            return super().reset(seed=seed)

    collector = FixedStepRolloutCollector(
        InMemoryReplayStore(),
        Pipeline(),
        RolloutCollectionConfig(Policy(), Environment(), 10, reset_at_rollout_end=True),
    )
    collector.collect(1, "first")
    assert gc.isenabled()
    assert events == ["warm", "reset"]


@pytest.mark.parametrize("previously_enabled", [True, False])
def test_rollout_restores_gc_state_after_environment_failure(*, previously_enabled: bool) -> None:
    class Environment(FakeTrackmania):
        def step(self, action: Any) -> tuple[dict[str, float], float, bool, bool, dict[str, Any]]:
            del action
            assert not gc.isenabled()
            raise RuntimeError("step failed")

    original = gc.isenabled()
    (gc.enable if previously_enabled else gc.disable)()
    try:
        collector = FixedStepRolloutCollector(
            InMemoryReplayStore(),
            IdentityFeaturePipeline(),
            RolloutCollectionConfig(ZeroPolicy(), Environment(), 10, reset_at_rollout_end=True),
        )
        with pytest.raises(RuntimeError, match="step failed"):
            collector.collect(1, "first")
        assert gc.isenabled() == previously_enabled
    finally:
        (gc.enable if original else gc.disable)()
