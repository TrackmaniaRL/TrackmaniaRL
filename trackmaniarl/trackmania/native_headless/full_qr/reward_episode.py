"""Historical reward computations over typed native frames and explicit events.

This component calculates rewards; it does not qualify event sources. A deployment
must explicitly bind a separately qualified collision/UI source. Unknown events
remain None and must never become false by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    NativeCommittedGoalFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGoalTerminalFrame,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import RewardResult, TransitionInput

if TYPE_CHECKING:
    from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
        EventStreamIdentity,
        RewardEventCollector,
    )


@dataclass(frozen=True, slots=True)
class RewardEventWindow:
    """Caller-supplied events for one (start, end] decision interval.

    Collision means the historical >100 motor callback latch, consumed once at
    the boundary. Finish means the historical UI Finish state at that boundary.
    Contact counts, a native endpoint classification or a speed drop are not
    substitutes. Constructing this value does not establish event provenance.
    None explicitly represents an unavailable source, including missing negative
    coverage. Tests may supply synthetic values only as computation fixtures.
    """

    start_ms: int
    end_ms: int
    collision_detected: bool | None
    finish_ui_active: bool | None

    def __post_init__(self) -> None:
        if (
            type(self.start_ms) is not int
            or type(self.end_ms) is not int
            or not 0 <= self.start_ms < self.end_ms <= 150_000
            or self.start_ms % 50
            or self.end_ms - self.start_ms != 50
        ):
            raise ValueError("events require one exact 50ms interval inside the 150s domain")
        for value in (self.collision_detected, self.finish_ui_active):
            if value is not None and type(value) is not bool:
                raise TypeError("reward events must be bool or explicitly unavailable (None)")

    def require_complete(self) -> tuple[bool, bool]:
        if self.collision_detected is None or self.finish_ui_active is None:
            raise ValueError("collision and Finish UI sources both require complete coverage")
        return self.collision_detected, self.finish_ui_active


@dataclass(frozen=True, slots=True)
class GoalTerminalEventWindow:
    """Only the actual observed tail after an occurred, qualified Finish."""

    start_ms: int
    end_ms: int
    collision_detected: bool

    def __post_init__(self) -> None:
        if (
            type(self.start_ms) is not int
            or type(self.end_ms) is not int
            or not 0 <= self.start_ms < self.end_ms <= 150000
            or self.start_ms % 50
            or self.end_ms % 10
            or not 0 < self.end_ms - self.start_ms < 50
            or type(self.collision_detected) is not bool
        ):
            raise ValueError("one complete actual partial Goal event interval required")


class NativeRewardEpisode:
    """Own one actor's unmodified TrajectoryReward and strict episode clock.

    Rejected inputs do not consume a step. Errors inside reward computation
    invalidate the episode until reset, because its state may have changed.
    This component deliberately exposes no replay, learner or launch path.
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        config: RewardConfig,
        *,
        event_identity: EventStreamIdentity | None = None,
    ) -> None:
        if config.maximum_race_time_s != 150:
            raise ValueError("full QR requires the historical 150s reward horizon")
        self._reward = TrajectoryReward(trajectory, config)
        self._last_time_ms: int | None = None
        self._terminal = False
        self._event_identity = event_identity

    def step_from_event_stream(
        self,
        frame: NativeGraphFrame,
        *,
        collector: RewardEventCollector,
        issued_steering: float,
    ) -> RewardResult:
        """Commit one actor's events only after its reward step succeeds.

        This joins qualified delivery to computation. It does not qualify the
        provider: missing motor/UI coverage still raises before any mutation.
        The actor must serialize this call with its event delivery loop.
        """
        from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
            EventStreamIdentity,
            RewardEventCollector,
        )

        if (
            type(self._event_identity) is not EventStreamIdentity
            or type(collector) is not RewardEventCollector
            or collector.identity != self._event_identity
        ):
            raise ValueError("reward episode requires its independently bound event identity")
        end_ms = int(self._values(frame)[0])
        events = collector.peek(self._event_identity, end_ms)
        result = self.step(frame, events=events, issued_steering=issued_steering)
        collector.consume(self._event_identity, end_ms)
        return result

    def step_terminal_from_event_stream(
        self,
        frame: NativeGoalTerminalFrame,
        *,
        collector: RewardEventCollector,
        issued_steering: float,
    ) -> RewardResult:
        from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
            EventStreamIdentity,
            RewardEventCollector,
        )

        if (
            type(frame) is not NativeGoalTerminalFrame
            or self._terminal
            or self._last_time_ms != frame.capture.previous_native_t - 2200
            or type(self._event_identity) is not EventStreamIdentity
            or type(collector) is not RewardEventCollector
            or collector.identity != self._event_identity
        ):
            raise ValueError("actual final frame bound to the active reward episode required")
        values = np.asarray(frame.channels, dtype=np.float32)
        end_ms = int(values[0])
        events = collector.peek_terminal(self._event_identity, end_ms)
        if events.start_ms != self._last_time_ms:
            raise ValueError("terminal event coverage belongs to another reward interval")
        result = self._apply_transition(
            values,
            collision=events.collision_detected,
            finish=True,
            issued_steering=issued_steering,
        )
        if not result.terminated or result.reason != "finished":
            raise ValueError("qualified terminal Goal did not produce historical Finish reward")
        collector.consume_terminal(self._event_identity, end_ms)
        return result

    def step_committed_from_event_stream(
        self,
        frame: NativeCommittedGoalFrame,
        *,
        collector: RewardEventCollector,
        issued_steering: float,
    ) -> RewardResult:
        from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
            EventStreamIdentity,
            RewardEventCollector,
        )

        if (
            type(frame) is not NativeCommittedGoalFrame
            or self._terminal
            or self._last_time_ms != frame.capture.previous_native_t - 2200
            or type(self._event_identity) is not EventStreamIdentity
            or type(collector) is not RewardEventCollector
            or collector.identity != self._event_identity
        ):
            raise ValueError("committed frame bound to the active reward episode required")
        events = collector.peek_committed(self._event_identity, frame.score_race_time_ms)
        if (
            events.start_ms != self._last_time_ms
            or events.sampled_ms != int(frame.channels[0])
            or events.final_sample_completed
            != (
                frame.capture.final_boundary is not None
                and frame.capture.vehicle.native_t - frame.capture.previous_native_t == 50
            )
        ):
            raise ValueError("committed event/sample clocks belong to another reward interval")
        result = self._apply_transition(
            np.asarray(frame.channels, dtype=np.float32),
            collision=events.collision_detected,
            finish=True,
            issued_steering=issued_steering,
            race_time_ms=events.score_ms,
        )
        if not result.terminated or result.reason != "finished":
            raise ValueError("committed Goal did not produce historical Finish reward")
        collector.consume_committed(self._event_identity, events.score_ms)
        return result

    @property
    def progress_m(self) -> float:
        return self._reward.progress_m

    @property
    def progress_pct(self) -> float:
        return self._reward.progress_pct

    def reset(self, initial: NativeGraphFrame) -> None:
        values = self._values(initial)
        if values[0] != 0:
            raise ValueError("reward reset requires the initial native frame at time0")
        self._last_time_ms = None
        self._reward.reset(values[1:4], velocity=values[4:7], race_time_ms=0.0)
        self._terminal = False
        self._last_time_ms = 0

    def step(
        self,
        frame: NativeGraphFrame,
        *,
        events: RewardEventWindow,
        issued_steering: float,
    ) -> RewardResult:
        if self._last_time_ms is None or self._terminal:
            raise RuntimeError("reward episode must be reset before stepping")
        values = self._values(frame)
        end_ms = int(values[0])
        if end_ms != self._last_time_ms + 50:
            raise ValueError("reward episode must advance exactly50ms without gaps or repeats")
        if not isinstance(events, RewardEventWindow):
            raise TypeError("an explicit RewardEventWindow is required")
        if events.start_ms != self._last_time_ms or events.end_ms != end_ms:
            raise ValueError("event window must cover this exact reward decision")
        collision, finish = events.require_complete()
        return self._apply_transition(
            values, collision=collision, finish=finish, issued_steering=issued_steering
        )

    def _apply_transition(  # noqa: PLR0913 - actual values, two independent sourced events and issued action.
        self,
        values: np.ndarray,
        *,
        collision: bool,
        finish: bool,
        issued_steering: float,
        race_time_ms: int | None = None,
    ) -> RewardResult:
        # A committed score has an independently sourced clock. Keep the
        # observation's physical timestamp and coordinates unchanged.
        end_ms = int(values[0]) if race_time_ms is None else race_time_ms
        if not isfinite(issued_steering) or not -1 <= issued_steering <= 1:
            raise ValueError("issued steering must be finite and inside [-1,1]")
        # Historical environment_step uses the successfully issued command,
        # not the potentially delayed steering channel from telemetry.
        transition = TransitionInput(
            position=values[1:4],
            finish_ui_active=finish,
            velocity=values[4:7],
            race_time_ms=float(end_ms),
            collision=collision,
            steering=float(issued_steering),
        )
        self._last_time_ms = None
        result = self._reward.step(transition)
        self._last_time_ms = end_ms
        self._terminal = result.terminated
        return result

    @staticmethod
    def _values(frame: NativeGraphFrame) -> np.ndarray:
        if not isinstance(frame, NativeGraphFrame):
            raise TypeError("reward requires a typed NativeGraphFrame")
        # Match the historical float32 telemetry before reward computations.
        return np.asarray(frame.channels, dtype=np.float32)
