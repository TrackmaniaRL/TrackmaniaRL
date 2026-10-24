"""Strict event delivery between a qualified source and the reward adapter.

This implements transport/consumption, not a physics-to-motor or UI source.
Only the backend that observed complete delivery may advance its watermark.
An EOF, empty contact batch or endpoint classification cannot supply coverage.
"""

from collections import deque
from dataclasses import dataclass

from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    CommittedGoalEventWindow,
)
from trackmaniarl.trackmania.native_headless.full_qr.motor_feedback import MotorFeedbackLatch
from trackmaniarl.trackmania.native_headless.full_qr.reward_episode import (
    GoalTerminalEventWindow,
    RewardEventWindow,
)


@dataclass(frozen=True, slots=True)
class EventStreamIdentity:
    actor: str
    episode: str

    def __post_init__(self) -> None:
        if any(type(v) is not str or not v.strip() for v in (self.actor, self.episode)):
            raise ValueError("explicit actor and episode identity required")


class RewardEventCollector:
    """One episode's ordered notifications with explicit negative coverage.

    Times are integer race milliseconds, not callback receipt wall time. A
    source must resolve that clock before submission. Watermarks are promises
    that all notifications through that inclusive time have been delivered;
    late notifications are errors, never silently assigned to a later step.
    Boundary UI observations remain a separate required source. Instances are
    actor-owned and must be called serially by that actor's delivery loop.
    """

    def __init__(self, identity: EventStreamIdentity) -> None:
        if type(identity) is not EventStreamIdentity:
            raise TypeError("typed event stream identity required")
        self.identity = identity
        self._consumed = 0
        self._sealed = -1
        self._sequence = 0
        self._last_motor_time = -1
        self._motors: deque[tuple[int, int]] = deque()
        self._ui: dict[int, bool] = {}
        self._terminal_boundary: int | None = None
        self._terminal_consumed = False
        self._committed_sample_ms: int | None = None
        self._committed_final_sample_completed = False

    @property
    def consumed_through_ms(self) -> int:
        return self._consumed

    def _check_identity(self, identity: EventStreamIdentity) -> None:
        if self._terminal_consumed:
            raise ValueError("terminal event stream has already been consumed")
        if type(identity) is not EventStreamIdentity or identity != self.identity:
            raise ValueError("foreign actor or episode event stream")

    @staticmethod
    def _time(value: int) -> None:
        if type(value) is not int or not 0 <= value <= 150_000:
            raise ValueError("integer race clock inside the 150s domain required")

    def record_motor(  # noqa: PLR0913 - episode identity, sequence, clock and byte are independent source facts.
        self,
        identity: EventStreamIdentity,
        *,
        sequence: int,
        race_time_ms: int,
        large_motor: int,
    ) -> None:
        self._check_identity(identity)
        self._time(race_time_ms)
        if type(sequence) is not int or sequence != self._sequence + 1:
            raise ValueError("motor notification sequence gap or duplicate")
        if self._terminal_boundary is not None:
            raise ValueError("terminal source is already sealed for new motor notifications")
        if type(large_motor) is not int or not 0 <= large_motor <= 255:
            raise ValueError("unsigned callback motor byte required")
        if (
            race_time_ms <= self._consumed
            or race_time_ms <= self._sealed
            or race_time_ms < self._last_motor_time
        ):
            raise ValueError("late or reordered motor notification")
        # The historical latch only needs to retain positives. Low callbacks
        # still advance the delivery sequence and cannot clear a prior pulse.
        if large_motor > 100:
            if len(self._motors) >= 8192:
                raise ValueError("pending motor notification quota exceeded")
            self._motors.append((race_time_ms, large_motor))
        self._sequence = sequence
        self._last_motor_time = race_time_ms

    def seal_motor_through(self, identity: EventStreamIdentity, race_time_ms: int) -> None:
        self._check_identity(identity)
        self._time(race_time_ms)
        if race_time_ms <= self._sealed:
            raise ValueError("motor coverage must advance strictly")
        if self._terminal_boundary is not None and race_time_ms != self._terminal_boundary:
            raise ValueError("terminal motor coverage must end at the actual final boundary")
        self._sealed = race_time_ms

    def record_ui_boundary(
        self, identity: EventStreamIdentity, *, race_time_ms: int, finish_active: bool
    ) -> None:
        self._check_identity(identity)
        self._time(race_time_ms)
        if (
            race_time_ms <= self._consumed
            or race_time_ms % 50
            or race_time_ms in self._ui
            or type(finish_active) is not bool
            or self._terminal_boundary is not None
        ):
            raise ValueError("one explicit Finish UI observation per future 50ms boundary required")
        self._ui[race_time_ms] = finish_active

    def record_terminal_boundary(
        self, identity: EventStreamIdentity, *, race_time_ms: int, finish_active: bool
    ) -> None:
        self._check_identity(identity)
        self._time(race_time_ms)
        if (
            finish_active is not True
            or race_time_ms % 10
            or not 0 < race_time_ms - self._consumed < 50
            or self._terminal_boundary is not None
            or self._ui
            or self._last_motor_time > race_time_ms
        ):
            raise ValueError("one actual positive partial Finish boundary required")
        self._terminal_boundary = race_time_ms

    def peek_terminal(self, identity: EventStreamIdentity, end_ms: int) -> GoalTerminalEventWindow:
        self._check_identity(identity)
        self._time(end_ms)
        if (
            self._committed_sample_ms is not None
            or self._terminal_boundary != end_ms
            or self._sealed < end_ms
        ):
            raise ValueError("complete partial motor coverage and qualified Finish required")
        latch = MotorFeedbackLatch()
        for clock, motor in self._motors:
            if clock > end_ms:
                break
            latch.record_callback(motor)
        return GoalTerminalEventWindow(self._consumed, end_ms, latch.consume_collision())

    def consume_terminal(
        self, identity: EventStreamIdentity, end_ms: int
    ) -> GoalTerminalEventWindow:
        result = self.peek_terminal(identity, end_ms)
        self._motors.clear()
        self._consumed = end_ms
        self._terminal_consumed = True
        return result

    def record_committed_boundary(  # noqa: PLR0913 - identity and the two clocks plus explicit capture scope.
        self,
        identity: EventStreamIdentity,
        *,
        sampled_ms: int,
        score_ms: int,
        final_sample_completed: bool = False,
    ) -> None:
        """The qualified source supplies both clocks and later seals motor coverage."""
        self._check_identity(identity)
        CommittedGoalEventWindow(
            self._consumed, sampled_ms, score_ms, False, final_sample_completed
        )
        if self._terminal_boundary is not None or self._ui or self._last_motor_time > score_ms:
            raise ValueError("one new committed terminal boundary required")
        self._committed_sample_ms = sampled_ms
        self._committed_final_sample_completed = final_sample_completed
        self._terminal_boundary = score_ms

    def peek_committed(
        self, identity: EventStreamIdentity, score_ms: int
    ) -> CommittedGoalEventWindow:
        self._check_identity(identity)
        self._time(score_ms)
        if (
            self._committed_sample_ms is None
            or self._terminal_boundary != score_ms
            or self._sealed != score_ms
        ):
            raise ValueError("complete motor delivery through committed score required")
        latch = MotorFeedbackLatch()
        for clock, motor in self._motors:
            if clock > score_ms:
                break
            latch.record_callback(motor)
        return CommittedGoalEventWindow(
            self._consumed,
            self._committed_sample_ms,
            score_ms,
            latch.consume_collision(),
            self._committed_final_sample_completed,
        )

    def consume_committed(
        self, identity: EventStreamIdentity, score_ms: int
    ) -> CommittedGoalEventWindow:
        result = self.peek_committed(identity, score_ms)
        self._motors.clear()
        self._consumed = score_ms
        self._terminal_consumed = True
        return result

    def peek(self, identity: EventStreamIdentity, end_ms: int) -> RewardEventWindow:
        """Validate and preview a boundary without consuming reward history."""
        self._check_identity(identity)
        self._time(end_ms)
        if end_ms != self._consumed + 50:
            raise ValueError("reward events must be consumed once in consecutive 50ms windows")
        if self._sealed < end_ms or end_ms not in self._ui:
            raise ValueError("complete motor coverage and explicit Finish UI boundary required")
        latch = MotorFeedbackLatch()
        for clock, motor in self._motors:
            if clock > end_ms:
                break
            latch.record_callback(motor)
        return RewardEventWindow(
            self._consumed, end_ms, latch.consume_collision(), self._ui[end_ms]
        )

    def consume(self, identity: EventStreamIdentity, end_ms: int) -> RewardEventWindow:
        result = self.peek(identity, end_ms)
        while self._motors and self._motors[0][0] <= end_ms:
            self._motors.popleft()
        del self._ui[end_ms]
        self._consumed = end_ms
        return result
