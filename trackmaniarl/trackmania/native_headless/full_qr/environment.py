"""SDK adapter for a full native provider with explicit event coverage.

The separately qualified native provider supplies explicit source coverage.
An unqualified diagnostic ImmediateSession cannot satisfy this contract.
In particular, EOF is never converted into a transition.
Prepared graph observations allow the SDK to collate/replay ordinary mappings
without serializing a native frame or recalculating its causal feature history.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol, cast

import numpy as np

from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    NativeCommittedGoalFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGoalTerminalFrame,
    NativeGraphFeaturePipeline,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.reward_episode import NativeRewardEpisode
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError
from trackmaniarl.trackmania.reward_config import RewardConfig


@dataclass(frozen=True, slots=True)
class FullQrReset:
    frame: NativeGraphFrame
    events: RewardEventCollector


@dataclass(frozen=True, slots=True)
class FullQrStep:
    frame: NativeGraphFrame
    # Historical reward uses the issued command, before controller calibration.
    issued_steering: float


@dataclass(frozen=True, slots=True)
class FullQrTerminalStep:
    frame: NativeGoalTerminalFrame | NativeCommittedGoalFrame
    issued_steering: float


@dataclass(frozen=True, slots=True)
class FullQrEnvironmentSettings:
    trajectory: np.ndarray
    reward_config: RewardConfig
    actor: str

    def __post_init__(self) -> None:
        if not self.actor or self.reward_config.maximum_race_time_s != 150:
            raise ValueError("an actor identity and the full 150s horizon are required")


class FullQrProvider(Protocol):
    """One actor's engine and fully sourced motor/UI delivery, not a raw tape.

    reset must establish a new episode identity; advance must fill that stream
    through the returned boundary before returning. Unknown/incomplete sources
    must raise. Source qualification remains a separate deployment requirement.
    """

    def reset(self, *, seed: int | None = None) -> FullQrReset: ...

    def advance(self, action: int) -> FullQrStep | FullQrTerminalStep: ...

    def close(self) -> None: ...


class FullQrEnvironment:
    """Join full features, per-episode delivery and unchanged historical reward.

    Any failure after starting engine work closes the provider and is fatal to
    this adapter. It cannot silently turn into a terminal replay row or reset.
    There is deliberately no default backend or factory in the training preset.
    """

    def __init__(
        self,
        provider: FullQrProvider,
        pipeline: NativeGraphFeaturePipeline,
        settings: FullQrEnvironmentSettings,
    ) -> None:
        self.provider, self.pipeline = provider, pipeline
        self.trajectory = settings.trajectory
        self.reward_config, self.actor = settings.reward_config, settings.actor
        self._events: RewardEventCollector | None = None
        self._reward: NativeRewardEpisode | None = None
        self._seen: set[EventStreamIdentity] = set()
        self._ended, self._closed = True, False
        self._episode_started_monotonic = 0.0
        self._last_physics_time_ms = 0.0

    def _fail(self, error: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            error.add_note(f"full QR provider cleanup failed: {cleanup!r}")

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise NativeLifecycleError("create a fresh full QR environment after close/failure")
        try:
            reset = self.provider.reset(seed=seed)
            if type(reset) is not FullQrReset or type(reset.events) is not RewardEventCollector:
                raise TypeError("a full typed reset and event stream are required")
            identity = reset.events.identity
            if identity.actor != self.actor or identity in self._seen:
                raise ValueError("reset requires a fresh episode bound to this actor")
            if reset.events.consumed_through_ms != 0:
                raise ValueError("reset event stream is already consumed")
            reward = NativeRewardEpisode(
                self.trajectory, self.reward_config, event_identity=identity
            )
            reward.reset(reset.frame)
            self.pipeline.reset_episode()
            observation = self.pipeline.transform_observation(reset.frame)
            self._events, self._reward = reset.events, reward
            self._seen.add(identity)
            self._ended = False
            self._episode_started_monotonic = monotonic()
            self._last_physics_time_ms = float(reset.frame.channels[0])
            return observation, {"actor": identity.actor, "episode": identity.episode}
        except BaseException as error:
            self._fail(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise NativeLifecycleError("full QR reset failed; provider closed") from error

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if type(action) is not int or not 0 <= action < 78:
            raise ValueError("one of the historical 78 actions is required")
        if self._closed or self._ended or self._events is None or self._reward is None:
            raise NativeLifecycleError("an active full QR reset is required")
        try:
            step = self.provider.advance(action)
            if type(step) not in (FullQrStep, FullQrTerminalStep):
                raise TypeError("EOF or missing native data is not a full QR step")
            # Validate causal features before consuming event coverage. A failed
            # feature computation invalidates the whole adapter, never replay.
            committed = (
                type(step) is FullQrTerminalStep and type(step.frame) is NativeCommittedGoalFrame
            )
            if committed:
                committed_frame = cast(NativeCommittedGoalFrame, step.frame)
                observation = self.pipeline.transform_committed_terminal(committed_frame)
                result = self._reward.step_committed_from_event_stream(
                    committed_frame, collector=self._events, issued_steering=step.issued_steering
                )
            elif type(step) is FullQrTerminalStep:
                terminal_frame = cast(NativeGoalTerminalFrame, step.frame)
                observation = self.pipeline.transform_terminal(terminal_frame)
                result = self._reward.step_terminal_from_event_stream(
                    terminal_frame, collector=self._events, issued_steering=step.issued_steering
                )
            else:
                regular_frame = cast(NativeGraphFrame, step.frame)
                observation = self.pipeline.transform_observation(regular_frame)
                result = self._reward.step_from_event_stream(
                    regular_frame, collector=self._events, issued_steering=step.issued_steering
                )
            self._ended = result.terminated
            physics_time_ms = float(step.frame.channels[0])
            step_race_ms = physics_time_ms - self._last_physics_time_ms
            self._last_physics_time_ms = physics_time_ms
            episode_observed_monotonic = monotonic()
            return (
                observation,
                result.reward,
                result.terminated,
                False,
                {
                    "actor": self.actor,
                    "episode": self._events.identity.episode,
                    "race_time_ms": (
                        cast(NativeCommittedGoalFrame, step.frame).score_race_time_ms
                        if committed
                        else step.frame.channels[0]
                    ),
                    "observed_physics_time_ms": step.frame.channels[0],
                    "step_race_time_ms": step_race_ms,
                    "episode_elapsed_s": (
                        episode_observed_monotonic - self._episode_started_monotonic
                    ),
                    "episode_started_monotonic": self._episode_started_monotonic,
                    "episode_observed_monotonic": episode_observed_monotonic,
                    "termination_reason": result.reason,
                    "progress_m": self._reward.progress_m,
                    "progress_pct": self._reward.progress_pct,
                    "collision_detected": result.collision_detected,
                    "reward_collision": result.collision_reward,
                    "control_steer": step.issued_steering,
                },
            )
        except BaseException as error:
            self._fail(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise NativeLifecycleError("full QR step failed; no transition returned") from error

    def close(self) -> None:
        if not self._closed:
            self._closed, self._ended = True, True
            self.provider.close()
