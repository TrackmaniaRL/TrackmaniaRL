"""Separate score/sample clocks, complete motor delivery and immutable regular guards."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.unit.experiments.test_graph_iqn_v5 import _geometry
from tests.unit.models.test_full_qr_committed_goal_capture import FIXTURE, _arguments
from tests.unit.models.test_full_qr_reward_episode import _config
from trackmaniarl.trackmania.native_headless.full_qr.adherence_capture import decode_adherence
from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    NativeCommittedGoalFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGraphFeaturePipeline,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.physical_rumble import PhysicalRumbleStep
from trackmaniarl.trackmania.native_headless.full_qr.render_rumble import Virtual150RumbleClock
from trackmaniarl.trackmania.native_headless.full_qr.reward_episode import NativeRewardEpisode
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import RumbleState
from trackmaniarl.trackmania.native_headless.full_qr.terminal_capture import (
    decode_committed_goal_capture,
)
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCaptureContext


def _actual_frame() -> NativeCommittedGoalFrame:
    arguments = _arguments()
    capture = decode_committed_goal_capture(**arguments)
    context = arguments["context"]
    adherence = decode_adherence(
        (FIXTURE / "adherence.bin").read_bytes(),
        VehicleCaptureContext(
            context.pid,
            context.meta["participant"],
            context.meta["vehicle"],
            context.meta["module_slide"],
            arguments["held_native_t"],
        ),
    )
    return NativeCommittedGoalFrame(capture, adherence)


def test_real_frame_keeps_physics_clock_and_actual_adherence() -> None:
    frame = _actual_frame()
    assert frame.channels[0] == 52790
    assert frame.score_race_time_ms == 52792
    with pytest.raises(ValueError, match="50ms"):
        NativeGraphFrame(frame.channels)
    with pytest.raises(ValueError, match="binding"):
        replace(frame, adherence=replace(frame.adherence, native_t=55000))


def test_score_after_an_already_consumed_regular_sample_does_not_repeat_feature_history(
    tmp_path: Path,
) -> None:
    # Unit clock variation, not a newly observed native trace. The real source
    # fixture above stays unchanged; actual replay qualification is separate.
    original = _actual_frame()
    capture = original.capture
    derived = replace(
        capture,
        previous_native_t=4700,
        vehicle=replace(capture.vehicle, native_t=4700),
        goal=replace(capture.goal, native_t=4700, last_goal_native_t=4702),
        fact=replace(
            capture.fact, event_native_t=4702, callback_native_t=4690, dispatch_native_t=4700
        ),
    )
    frame = NativeCommittedGoalFrame(derived, replace(original.adherence, native_t=4700))
    pipeline = NativeGraphFeaturePipeline(_geometry(tmp_path / "g.npz"), "test-map")
    goal = np.asarray(frame.channels[1:4], dtype=np.float32)
    trajectory = np.array([goal - [i, 0, 0] for i in range(100, -1, -1)], dtype=np.float32)
    identity = EventStreamIdentity("a", "unit-zero-tail")
    collector = RewardEventCollector(identity)
    reward = NativeRewardEpisode(trajectory, _config(), event_identity=identity)
    for index in range(51):
        channels = list(frame.channels)
        channels[0] = index * 50
        channels[1:4] = goal - [100 - index * 2, 0, 0]
        ordinary = NativeGraphFrame(tuple(channels))
        prepared = pipeline.transform_observation(ordinary)
        if index == 0:
            reward.reset(ordinary)
        else:
            collector.record_ui_boundary(identity, race_time_ms=index * 50, finish_active=False)
            collector.seal_motor_through(identity, index * 50)
            reward.step_from_event_stream(ordinary, collector=collector, issued_steering=0)
    assert pipeline.transform_committed_terminal(frame) is prepared
    collector.record_committed_boundary(identity, sampled_ms=2500, score_ms=2502)
    collector.seal_motor_through(identity, 2502)
    result = reward.step_committed_from_event_stream(frame, collector=collector, issued_steering=0)
    assert result.terminated
    assert result.reason == "finished"
    assert frame.channels[0] == 2500
    assert frame.score_race_time_ms == collector.consumed_through_ms == 2502


def test_commit_cannot_consume_missing_motor_delivery_or_enter_regular_reader() -> None:
    identity = EventStreamIdentity("a", "1")
    collector = RewardEventCollector(identity)
    collector.record_motor(identity, sequence=1, race_time_ms=41, large_motor=101)
    collector.record_committed_boundary(identity, sampled_ms=40, score_ms=42)
    with pytest.raises(ValueError, match="complete motor delivery"):
        collector.consume_committed(identity, 42)
    assert collector.consumed_through_ms == 0
    collector.seal_motor_through(identity, 42)
    with pytest.raises(ValueError, match="partial motor"):
        collector.peek_terminal(identity, 42)
    with pytest.raises(ValueError, match="consecutive"):
        collector.peek(identity, 42)
    with pytest.raises(ValueError, match="foreign"):
        collector.peek_committed(EventStreamIdentity("a", "2"), 42)
    event = collector.consume_committed(identity, 42)
    assert (event.start_ms, event.sampled_ms, event.score_ms) == (0, 40, 42)
    assert event.collision_detected is True
    with pytest.raises(ValueError, match="already been consumed"):
        collector.consume_committed(identity, 42)


@pytest.mark.parametrize(("sample", "score"), [(40, 50), (40, 40), (30, 42), (0, 10)])
def test_commit_refuses_a_gap_or_unbounded_future_score(sample: int, score: int) -> None:
    identity = EventStreamIdentity("a", "1")
    with pytest.raises(ValueError, match="separately timestamped"):
        RewardEventCollector(identity).record_committed_boundary(
            identity, sampled_ms=sample, score_ms=score
        )


def _step(clock: int, amplitude: float) -> PhysicalRumbleStep:
    return PhysicalRumbleStep(clock, RumbleState(1.0, amplitude), (0.0, 0.0, 0.0), 0, 0.0, 0)


@pytest.mark.parametrize("offset", range(1, 10))
def test_terminal_render_matches_prefix_with_no_additional_physical_endpoint(offset: int) -> None:
    first, reference = (Virtual150RumbleClock(RumbleState(0.0, 0.0)) for _ in range(2))
    for tick in range(2200, 2230, 10):
        assert first.advance(_step(tick, 0.7)) == reference.advance(_step(tick, 0.7))
    expected = tuple(
        change
        for change in reference.advance(_step(2230, 0.01))
        if change.race_time_ms <= 20 + offset
    )
    actual = first.finish_committed(2220 + offset)
    assert actual == expected
    assert first.native_t == 2220
    assert first.rendered_through_native_t == 2220 + offset
    with pytest.raises(ValueError, match="already closed"):
        first.advance(_step(2230, 0.7))


def test_rejected_terminal_render_does_not_mutate_clock_or_history() -> None:
    clock = Virtual150RumbleClock(RumbleState(0.0, 0.0))
    clock.advance(_step(2200, 0.2))
    for end in (2200, 2210, True):
        with pytest.raises(ValueError, match="bounded committed"):
            clock.finish_committed(end)
    assert clock.native_t == clock.rendered_through_native_t == 2200
    clock.advance(_step(2210, 0.2))
