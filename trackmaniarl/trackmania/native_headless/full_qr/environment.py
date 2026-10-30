"""SDK adapter for a full native provider with explicit event coverage.

The separately qualified native provider supplies explicit source coverage.
An unqualified diagnostic ImmediateSession cannot satisfy this contract.
In particular, EOF is never converted into a transition.
Prepared graph observations allow the SDK to collate/replay ordinary mappings
without serializing a native frame or recalculating its causal feature history.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from time import monotonic
from typing import Any, Protocol, cast

import numpy as np
import torch

from trackmaniarl.core.pytree import tree_snapshot
from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    NativeCommittedGoalFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGoalTerminalFrame,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.reward_episode import NativeRewardEpisode
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_info import reward_result_info
from trackmaniarl.trackmania.reward_types import RewardResult


def _checked_observation(value: dict[str, Any]) -> dict[str, Any]:
    """Reject invalid custom features before consuming reward events."""
    if not isinstance(value, dict) or not value:
        raise TypeError("native observation pipeline must return a nonempty sensor mapping")

    def check(leaf: Any) -> None:
        if isinstance(leaf, (Mapping, list, tuple)):
            if not leaf:
                raise ValueError("native observation containers must not be empty")
            children = leaf.values() if isinstance(leaf, Mapping) else leaf
            for child in children:
                check(child)
            return
        if isinstance(leaf, torch.Tensor):
            if leaf.numel() == 0:
                raise ValueError("native observation sensors must not be empty")
            finite = bool(torch.isfinite(leaf).all())
        elif isinstance(leaf, (np.ndarray, np.number, np.bool_, int, float)):
            if isinstance(leaf, np.ndarray) and leaf.size == 0:
                raise ValueError("native observation sensors must not be empty")
            if isinstance(leaf, (np.ndarray, np.generic)) and leaf.dtype.kind not in "biufc":
                raise TypeError("native observation sensors must be numeric tensors or arrays")
            finite = bool(np.isfinite(leaf).all())
        else:
            raise TypeError("native observation sensors must be numeric tensors or arrays")
        if not finite:
            raise ValueError("native observation sensors must be finite")

    check(value)
    return cast(dict[str, Any], tree_snapshot(value))


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


class NativeObservationPipeline(Protocol):
    """Prepare model observations without taking ownership of physics or rewards.

    Each environment receives its own causal history. Implementations must handle
    all three qualified frame forms; EOF or absent telemetry is never an input.
    """

    def reset_episode(self) -> None: ...

    def transform_observation(self, observation: NativeGraphFrame) -> dict[str, Any]: ...

    def transform_terminal(self, observation: NativeGoalTerminalFrame) -> dict[str, Any]: ...

    def transform_committed_terminal(
        self, observation: NativeCommittedGoalFrame
    ) -> dict[str, Any]: ...


class FullQrEnvironment:
    """Join full features, per-episode delivery and unchanged historical reward.

    Any failure after starting engine work closes the provider and is fatal to
    this adapter. It cannot silently turn into a terminal replay row or reset.
    There is deliberately no default backend or factory in the training preset.
    """

    def __init__(
        self,
        provider: FullQrProvider,
        pipeline: NativeObservationPipeline,
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
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError("seed must be an integer or None")
            seed = int(seed)
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
            observation = _checked_observation(self.pipeline.transform_observation(reset.frame))
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
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 78:
            raise ValueError("one of the historical 78 actions is required")
        action = int(action)
        if self._closed or self._ended or self._events is None or self._reward is None:
            raise NativeLifecycleError("an active full QR reset is required")
        try:
            step = self.provider.advance(action)
            if type(step) not in (FullQrStep, FullQrTerminalStep):
                raise TypeError("EOF or missing native data is not a full QR step")
            observation, result = self._observe_and_reward(step)
            self._ended = result.terminated
            info = self._record_step_info(step, result)
            return observation, result.reward, result.terminated, False, info
        except BaseException as error:
            self._fail(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise NativeLifecycleError("full QR step failed; no transition returned") from error

    def _observe_and_reward(
        self, step: FullQrStep | FullQrTerminalStep
    ) -> tuple[dict[str, Any], RewardResult]:
        # Features must succeed before reward consumes the causal event stream.
        assert self._reward is not None
        assert self._events is not None
        if type(step) is FullQrTerminalStep and type(step.frame) is NativeCommittedGoalFrame:
            committed = step.frame
            observation = _checked_observation(
                self.pipeline.transform_committed_terminal(committed)
            )
            result = self._reward.step_committed_from_event_stream(
                committed, collector=self._events, issued_steering=step.issued_steering
            )
        elif type(step) is FullQrTerminalStep:
            terminal = cast(NativeGoalTerminalFrame, step.frame)
            observation = _checked_observation(self.pipeline.transform_terminal(terminal))
            result = self._reward.step_terminal_from_event_stream(
                terminal, collector=self._events, issued_steering=step.issued_steering
            )
        else:
            regular = cast(NativeGraphFrame, step.frame)
            observation = _checked_observation(self.pipeline.transform_observation(regular))
            result = self._reward.step_from_event_stream(
                regular, collector=self._events, issued_steering=step.issued_steering
            )
        return observation, result

    def _record_step_info(
        self, step: FullQrStep | FullQrTerminalStep, result: RewardResult
    ) -> dict[str, Any]:
        assert self._reward is not None
        assert self._events is not None
        frame = step.frame
        physics_time_ms = float(frame.channels[0])
        step_race_ms = physics_time_ms - self._last_physics_time_ms
        self._last_physics_time_ms = physics_time_ms
        observed_monotonic = monotonic()
        return {
            "actor": self.actor,
            "episode": self._events.identity.episode,
            "race_time_ms": (
                frame.score_race_time_ms
                if type(frame) is NativeCommittedGoalFrame
                else frame.channels[0]
            ),
            "observed_physics_time_ms": frame.channels[0],
            "step_race_time_ms": step_race_ms,
            "episode_elapsed_s": observed_monotonic - self._episode_started_monotonic,
            "episode_started_monotonic": self._episode_started_monotonic,
            "episode_observed_monotonic": observed_monotonic,
            "termination_reason": result.reason,
            "progress_m": self._reward.progress_m,
            "progress_pct": self._reward.progress_pct,
            **reward_result_info(result),
            "control_steer": step.issued_steering,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed, self._ended = True, True
            self.provider.close()
