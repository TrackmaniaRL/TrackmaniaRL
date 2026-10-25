"""Synthetic SDK integration, never native source qualification."""

from dataclasses import replace

import numpy as np
import pytest

from trackmaniarl.distributed.actor_metrics import EpisodeMetrics
from trackmaniarl.trackmania.diagnostics import ProgressBinDiagnostics
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrEnvironment,
    FullQrEnvironmentSettings,
    FullQrReset,
    FullQrStep,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import NativeGraphFrame
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError
from trackmaniarl.trackmania.reward_config import RewardConfig


def frame(clock: int) -> NativeGraphFrame:
    return NativeGraphFrame(
        (clock, clock / 20, 0, 0, 50, 0, 0, 1, 0, 0, 50, 5000, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0)
    )


class Pipeline:
    def reset_episode(self) -> None:
        self.times = []

    def transform_observation(self, value: NativeGraphFrame) -> dict[str, np.ndarray]:
        self.times.append(value.channels[0])
        return {"synthetic": np.asarray(value.channels, dtype=np.float32)}


class Provider:
    def __init__(self) -> None:
        self.closed = 0
        self.generation = 0
        self.clock = 0
        self.coverage = True
        self.foreign = False
        self.eof = False
        self.reuse = False
        self.finish = False

    def reset(self, *, seed: int | None = None) -> FullQrReset:
        if not self.reuse:
            self.generation += 1
        self.clock = 0
        self.identity = EventStreamIdentity(
            "foreign" if self.foreign else "actor0", str(self.generation)
        )
        self.events = RewardEventCollector(self.identity)
        return FullQrReset(frame(0), self.events)

    def advance(self, action: int) -> FullQrStep | None:
        self.clock += 50
        if self.eof:
            return None
        self.events.record_motor(
            self.identity, sequence=self.clock // 50, race_time_ms=self.clock, large_motor=255
        )
        self.events.record_ui_boundary(
            self.identity, race_time_ms=self.clock, finish_active=self.finish
        )
        if self.coverage:
            self.events.seal_motor_through(self.identity, self.clock)
        return FullQrStep(frame(self.clock), 0.0)

    def close(self) -> None:
        self.closed += 1


def setup() -> tuple[FullQrEnvironment, Provider, Pipeline]:
    provider, pipeline = Provider(), Pipeline()
    trajectory = np.zeros((10_001, 3), dtype=np.float32)
    trajectory[:, 0] = np.arange(len(trajectory))
    environment = FullQrEnvironment(
        provider,
        pipeline,
        FullQrEnvironmentSettings(
            trajectory,
            RewardConfig(maximum_race_time_s=150, collision_penalty=0.15, collision_cooldown_s=0),
            actor="actor0",
        ),
    )
    return environment, provider, pipeline


def test_sdk_mapping_collision_delivery_and_fresh_reset_history() -> None:
    environment, provider, pipeline = setup()
    initial, info = environment.reset(seed=17)
    assert initial["synthetic"].shape == (23,)
    assert info["episode"] == "1"
    observation, reward, terminated, truncated, info = environment.step(77)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["collision_detected"]
    assert info["reward_collision"] == -0.15
    assert observation["synthetic"][0] == 50
    assert pipeline.times == [0, 50]
    assert provider.events.consumed_through_ms == 50
    environment.reset()
    assert pipeline.times == [0]
    assert provider.events.consumed_through_ms == 0
    assert environment.step(0)[4]["episode"] == "2"
    environment.close()
    environment.close()
    assert provider.closed == 1


@pytest.mark.parametrize("finish", [False, True])
def test_actor_metrics_retain_actual_reward_components_and_collisions(*, finish: bool) -> None:
    environment, provider, _ = setup()
    # Let this short synthetic trace exercise the actual Finish reward path.
    environment.reward_config = replace(
        environment.reward_config, minimum_finish_steps=1, finish_progress=0.0001
    )
    metrics = EpisodeMetrics(ProgressBinDiagnostics(78, bin_count=20))
    try:
        environment.reset()
        provider.finish = finish
        _, reward, terminated, _, info = environment.step(41)
        metrics.record_reward(reward, info)
        components = {
            key: value
            for key, value in metrics._reward_totals().items()
            if key.startswith("reward_")
        }
        assert sum(components.values()) == pytest.approx(reward)
        assert components["reward_time"] < 0
        assert bool(components["reward_pbrs"]) is (not finish)
        assert components["reward_progress"] > 0
        assert components["reward_terminal"] == (30 if finish else 0)
        assert terminated is finish
        assert info["collision"] is True
        assert info["collision_detected"] is True
        assert metrics.collision_count == metrics.collision_detected_count == 1
        assert info["accepted_progress_delta_m"] > 0
        assert info["projected_velocity_mps"] > 0
    finally:
        environment.close()


@pytest.mark.parametrize("fault", ["coverage", "eof"])
def test_missing_source_or_eof_is_fatal_without_transition(fault: str) -> None:
    environment, provider, _ = setup()
    environment.reset()
    if fault == "coverage":
        provider.coverage = False
    else:
        provider.eof = True
    with pytest.raises(NativeLifecycleError, match="no transition returned"):
        environment.step(0)
    assert provider.closed == 1
    assert provider.events.consumed_through_ms == 0
    with pytest.raises(NativeLifecycleError):
        environment.reset()


@pytest.mark.parametrize("fault", ["foreign", "reuse"])
def test_reset_rejects_foreign_or_reused_episode(fault: str) -> None:
    environment, provider, _ = setup()
    environment.reset()
    setattr(provider, fault, True)
    with pytest.raises(NativeLifecycleError, match="reset failed"):
        environment.reset()
    assert provider.closed == 1


def test_bad_action_does_not_advance_engine_or_close_valid_episode() -> None:
    environment, provider, _ = setup()
    environment.reset()
    for action in (True, -1, 78, 1.0):
        with pytest.raises(ValueError, match="78 actions"):
            environment.step(action)
    assert provider.clock == 0
    assert provider.closed == 0
    environment.step(0)
    assert provider.clock == 50
    environment.close()


def test_interrupt_preserves_graceful_sdk_shutdown_and_closes_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, provider, _ = setup()
    environment.reset()

    def interrupt(action: int) -> FullQrStep:
        raise KeyboardInterrupt

    monkeypatch.setattr(provider, "advance", interrupt)
    with pytest.raises(KeyboardInterrupt):
        environment.step(0)
    assert provider.closed == 1
    assert provider.events.consumed_through_ms == 0
