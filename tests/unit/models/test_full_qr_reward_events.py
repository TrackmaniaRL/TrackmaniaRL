"""Event transport/latch tests; synthetic clocks do not qualify native sources."""

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)


def setup() -> tuple[EventStreamIdentity, RewardEventCollector]:
    identity = EventStreamIdentity("actor-0", "episode-1")
    return identity, RewardEventCollector(identity)


@pytest.mark.parametrize("end_ms", [10, 20, 30, 40])
def test_terminal_tail_requires_actual_finish_and_motor_coverage(end_ms: int) -> None:
    identity, collector = setup()
    collector.record_motor(identity, sequence=1, race_time_ms=end_ms, large_motor=101)
    with pytest.raises(ValueError, match="positive partial"):
        collector.record_terminal_boundary(identity, race_time_ms=end_ms, finish_active=False)
    collector.record_terminal_boundary(identity, race_time_ms=end_ms, finish_active=True)
    with pytest.raises(ValueError, match="complete partial motor coverage"):
        collector.consume_terminal(identity, end_ms)
    assert collector.consumed_through_ms == 0
    collector.seal_motor_through(identity, end_ms)
    # Regular windows still refuse this terminal-only interval.
    with pytest.raises(ValueError, match="50ms"):
        collector.peek(identity, end_ms)
    terminal = collector.consume_terminal(identity, end_ms)
    assert terminal.collision_detected is True
    assert terminal.end_ms == end_ms
    with pytest.raises(ValueError, match="already been consumed"):
        collector.consume_terminal(identity, end_ms)


def test_terminal_tail_rejects_future_notifications_and_foreign_episode() -> None:
    identity, collector = setup()
    collector.record_motor(identity, sequence=1, race_time_ms=45, large_motor=101)
    with pytest.raises(ValueError, match="positive partial"):
        collector.record_terminal_boundary(identity, race_time_ms=40, finish_active=True)
    with pytest.raises(ValueError, match="foreign"):
        collector.record_terminal_boundary(
            EventStreamIdentity(identity.actor, "other"), race_time_ms=40, finish_active=True
        )


def test_future_delivery_and_low_callbacks_do_not_erase_or_move_a_pulse() -> None:
    identity, collector = setup()
    for sequence, (clock, motor) in enumerate([(10, 101), (20, 0), (50, 100), (65, 255)], 1):
        collector.record_motor(identity, sequence=sequence, race_time_ms=clock, large_motor=motor)
    collector.seal_motor_through(identity, 150)
    for clock in (50, 100, 150):
        collector.record_ui_boundary(identity, race_time_ms=clock, finish_active=False)
    assert collector.consume(identity, 50).collision_detected is True
    assert collector.consume(identity, 100).collision_detected is True
    assert collector.consume(identity, 150).collision_detected is False


def test_boundary_pulse_is_consumed_once_and_empty_source_is_not_negative_evidence() -> None:
    identity, collector = setup()
    collector.record_motor(identity, sequence=1, race_time_ms=50, large_motor=101)
    collector.record_ui_boundary(identity, race_time_ms=50, finish_active=False)
    with pytest.raises(ValueError, match="complete motor coverage"):
        collector.consume(identity, 50)
    assert collector.consumed_through_ms == 0
    collector.seal_motor_through(identity, 50)
    assert collector.consume(identity, 50).collision_detected is True
    with pytest.raises(ValueError, match="once"):
        collector.consume(identity, 50)
    collector.record_ui_boundary(identity, race_time_ms=100, finish_active=False)
    collector.seal_motor_through(identity, 100)
    assert collector.consume(identity, 100).collision_detected is False


def test_motor_coverage_does_not_substitute_for_ui_coverage_or_finish_classification() -> None:
    identity, collector = setup()
    collector.seal_motor_through(identity, 100)
    with pytest.raises(ValueError, match="explicit Finish UI"):
        collector.consume(identity, 50)
    assert collector.consumed_through_ms == 0
    collector.record_ui_boundary(identity, race_time_ms=50, finish_active=False)
    collector.record_ui_boundary(identity, race_time_ms=100, finish_active=True)
    assert collector.consume(identity, 50).finish_ui_active is False
    assert collector.consume(identity, 100).finish_ui_active is True


@pytest.mark.parametrize("clock", [0, 50, 49])
def test_late_notifications_are_rejected_after_watermark(clock: int) -> None:
    identity, collector = setup()
    collector.seal_motor_through(identity, 50)
    with pytest.raises(ValueError, match="late"):
        collector.record_motor(identity, sequence=1, race_time_ms=clock, large_motor=101)


def test_wrong_actor_previous_episode_and_gaps_cannot_enter_another_stream() -> None:
    identity, collector = setup()
    for other in (
        EventStreamIdentity("actor-1", identity.episode),
        EventStreamIdentity(identity.actor, "episode-0"),
    ):
        with pytest.raises(ValueError, match="foreign"):
            collector.record_motor(other, sequence=1, race_time_ms=10, large_motor=101)
        with pytest.raises(ValueError, match="foreign"):
            collector.seal_motor_through(other, 50)
        with pytest.raises(ValueError, match="foreign"):
            collector.record_ui_boundary(other, race_time_ms=50, finish_active=False)
        with pytest.raises(ValueError, match="foreign"):
            collector.consume(other, 50)
    with pytest.raises(ValueError, match="sequence gap"):
        collector.record_motor(identity, sequence=2, race_time_ms=10, large_motor=101)
    collector.record_motor(identity, sequence=1, race_time_ms=10, large_motor=101)
    with pytest.raises(ValueError, match="sequence gap"):
        collector.record_motor(identity, sequence=1, race_time_ms=20, large_motor=101)


@pytest.mark.parametrize("clock", [10.0, True, -1, 150001])
def test_wall_clock_or_invalid_times_are_not_native_race_times(clock: object) -> None:
    identity, collector = setup()
    with pytest.raises(ValueError, match="integer race clock"):
        collector.seal_motor_through(identity, clock)


def test_failed_boundary_validation_leaves_pending_events_intact() -> None:
    identity, collector = setup()
    collector.record_motor(identity, sequence=1, race_time_ms=10, large_motor=255)
    collector.seal_motor_through(identity, 100)
    collector.record_ui_boundary(identity, race_time_ms=50, finish_active=False)
    with pytest.raises(ValueError, match="consecutive"):
        collector.consume(identity, 100)
    assert collector.consume(identity, 50).collision_detected is True
    with pytest.raises(ValueError, match="advance strictly"):
        collector.seal_motor_through(identity, 100)
    with pytest.raises(ValueError, match="future 50ms"):
        collector.record_ui_boundary(identity, race_time_ms=50, finish_active=False)


def test_ordering_failure_does_not_accept_a_reordered_positive_notification() -> None:
    identity, collector = setup()
    collector.record_motor(identity, sequence=1, race_time_ms=40, large_motor=0)
    with pytest.raises(ValueError, match="reordered"):
        collector.record_motor(identity, sequence=2, race_time_ms=30, large_motor=255)
    collector.record_motor(identity, sequence=2, race_time_ms=40, large_motor=0)
    collector.seal_motor_through(identity, 50)
    collector.record_ui_boundary(identity, race_time_ms=50, finish_active=False)
    assert collector.consume(identity, 50).collision_detected is False
