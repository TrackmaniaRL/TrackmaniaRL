"""Offline synthetic reward integration; never physical event qualification."""

from dataclasses import replace

import numpy as np
import pytest

from trackmaniarl.trackmania.native_headless.full_qr.native_features import NativeGraphFrame
from trackmaniarl.trackmania.native_headless.full_qr.reward_episode import (
    NativeRewardEpisode,
    RewardEventWindow,
)
from trackmaniarl.trackmania.native_headless.full_qr.reward_events import (
    EventStreamIdentity,
    RewardEventCollector,
)
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import RewardResult, TransitionInput


def _config(**changes: object) -> RewardConfig:
    return replace(
        RewardConfig(
            crash_distance=15,
            no_progress_steps=100,
            slow_progress_window_steps=120,
            minimum_progress_per_window_m=3,
            terminal_failure_penalty=2,
            collision_penalty=0.15,
            collision_cooldown_s=0,
            nearest_forward_points=100,
            nearest_backward_points=40,
            limit_progress_by_kinematics=False,
            time_penalty_per_second=1,
            maximum_race_time_s=150,
            progress_reward_full_lap=200,
            finish_reward=70,
            potential_progress_weight=0,
            projected_velocity_scale=0.0576,
            time_attack_target_s=35,
            time_attack_linear_scale=2,
        ),
        **changes,
    )


def _trajectory(length: int = 10_000) -> np.ndarray:
    points = np.zeros((length + 1, 3), dtype=np.float32)
    points[:, 0] = np.arange(length + 1)
    return points


def _frame(time_ms: int, x: float = 0, y: float = 0) -> NativeGraphFrame:
    # Synthetic consumed channels; absent fields are never made into an API packet.
    speed = 50
    return NativeGraphFrame(
        (time_ms, x, y, 0, speed, 0, 0, 1, 0, 0, speed, 5000, 2, 0, 0, 0, 0, 0, 0, 1, -0.9, 1, 0)
    )


def _step(episode: NativeRewardEpisode, time_ms: int) -> RewardResult:
    return episode.step(
        _frame(time_ms, time_ms / 20),
        events=RewardEventWindow(time_ms - 50, time_ms, False, False),
        issued_steering=0,
    )


def test_event_stream_commits_with_reward_and_preserves_pulse_on_invalid_control() -> None:
    identity = EventStreamIdentity("actor-0", "episode-1")
    episode = NativeRewardEpisode(_trajectory(), _config(), event_identity=identity)
    episode.reset(_frame(0))
    collector = RewardEventCollector(identity)
    collector.record_motor(identity, sequence=1, race_time_ms=15, large_motor=101)
    collector.record_motor(identity, sequence=2, race_time_ms=20, large_motor=0)
    collector.record_ui_boundary(identity, race_time_ms=50, finish_active=False)
    with pytest.raises(ValueError, match="complete motor coverage"):
        episode.step_from_event_stream(_frame(50, 2.5), collector=collector, issued_steering=0)
    collector.seal_motor_through(identity, 100)
    with pytest.raises(ValueError, match="issued steering"):
        episode.step_from_event_stream(_frame(50, 2.5), collector=collector, issued_steering=2)
    assert episode.progress_m == 0
    assert collector.consumed_through_ms == 0
    result = episode.step_from_event_stream(_frame(50, 2.5), collector=collector, issued_steering=0)
    assert result.collision_reward == -0.15
    assert collector.consumed_through_ms == 50
    collector.record_ui_boundary(identity, race_time_ms=100, finish_active=False)
    result = episode.step_from_event_stream(_frame(100, 5), collector=collector, issued_steering=0)
    assert result.collision_reward == 0
    assert collector.consumed_through_ms == 100


def test_event_stream_cannot_cross_actor_or_reset_generation() -> None:
    identity = EventStreamIdentity("actor-0", "episode-1")
    episode = NativeRewardEpisode(_trajectory(), _config(), event_identity=identity)
    episode.reset(_frame(0))
    for other in (
        EventStreamIdentity("actor-1", identity.episode),
        EventStreamIdentity(identity.actor, "episode-0"),
    ):
        collector = RewardEventCollector(other)
        collector.seal_motor_through(other, 50)
        collector.record_ui_boundary(other, race_time_ms=50, finish_active=True)
        with pytest.raises(ValueError, match="independently bound"):
            episode.step_from_event_stream(_frame(50), collector=collector, issued_steering=0)
        assert collector.consumed_through_ms == 0
        assert episode.progress_m == 0


def test_reward_matches_historical_mapping_and_uses_issued_steering() -> None:
    config = _config(steering_delta_penalty=0.25)
    episode = NativeRewardEpisode(_trajectory(100), config)
    historical = TrajectoryReward(_trajectory(100), config)
    initial = _frame(0)
    episode.reset(initial)
    historical.reset(np.zeros(3, dtype=np.float32), velocity=np.array([50, 0, 0]), race_time_ms=0)
    outputs = []
    for index in range(1, 61):
        frame = _frame(index * 50, index * 100 / 60)
        values = np.asarray(frame.channels, dtype=np.float32)
        steering = 0.5 if index % 2 else -0.25
        events = RewardEventWindow((index - 1) * 50, index * 50, index in (11, 12), index == 60)
        actual = episode.step(frame, events=events, issued_steering=steering)
        expected = historical.step(
            TransitionInput(
                values[1:4], index == 60, values[4:7], index * 50, index in (11, 12), steering
            )
        )
        assert actual == expected
        assert episode.progress_m == historical.progress_m
        outputs.append(actual)
    assert outputs[-1].reason == "finished"
    assert outputs[-1].terminal_reward == 70
    assert outputs[-1].time_attack_terminal_reward == 64
    assert outputs[1].steering_delta_reward == -0.25 * 0.75
    assert outputs[10].collision_reward == outputs[11].collision_reward == -0.15
    assert outputs[12].collision_reward == 0
    with pytest.raises(RuntimeError, match="reset"):
        _step(episode, 3050)
    episode.reset(initial)
    assert episode.progress_m == 0
    assert _step(episode, 50).collision_reward == 0


def test_full_150s_horizon_without_earlier_progress_termination() -> None:
    episode = NativeRewardEpisode(_trajectory(), _config())
    episode.reset(_frame(0))
    for index in range(1, 3001):
        result = _step(episode, index * 50)
        assert result.terminated == (index == 3000)
        assert result.time_reward == pytest.approx(-0.05)
        assert result.projected_velocity_reward == pytest.approx(0.144)
    assert result.reason == "time_limit"
    assert result.terminal_reward == -2
    assert episode.progress_m == 7500


@pytest.mark.parametrize(("collision", "finish"), [(None, False), (False, None), (None, None)])
def test_missing_events_do_not_become_negative_evidence(
    *, collision: bool | None, finish: bool | None
) -> None:
    episode = NativeRewardEpisode(_trajectory(), _config())
    episode.reset(_frame(0))
    with pytest.raises(ValueError, match="complete coverage"):
        episode.step(
            _frame(50, 2.5),
            events=RewardEventWindow(0, 50, collision, finish),
            issued_steering=0,
        )
    # A rejected event window must not advance reward time/progress/history.
    result = _step(episode, 50)
    assert result.reward == pytest.approx(-0.05 + 0.04 + 0.144)


def test_wrong_windows_skips_duplicates_and_bad_control_preserve_history() -> None:
    episode = NativeRewardEpisode(_trajectory(), _config())
    with pytest.raises(RuntimeError, match="reset"):
        _step(episode, 50)
    with pytest.raises(ValueError, match="time0"):
        episode.reset(_frame(50))
    episode.reset(_frame(0))
    with pytest.raises(ValueError, match="without gaps"):
        _step(episode, 100)
    for events, steering, message in (
        (RewardEventWindow(50, 100, False, False), 0, "exact reward decision"),
        (RewardEventWindow(0, 50, False, False), float("nan"), "issued steering"),
        (RewardEventWindow(0, 50, False, False), 1.1, "issued steering"),
    ):
        with pytest.raises(ValueError, match=message):
            episode.step(_frame(50), events=events, issued_steering=steering)
    _step(episode, 50)
    with pytest.raises(ValueError, match="without gaps"):
        _step(episode, 50)
    _step(episode, 100)


@pytest.mark.parametrize("case", ["off_track", "no_progress", "slow_progress"])
def test_earlier_terminal_reasons_are_preserved(case: str) -> None:
    episode = NativeRewardEpisode(
        _trajectory(), _config(no_progress_steps=200 if case == "slow_progress" else 100)
    )
    episode.reset(_frame(0))
    for index in range(1, 201):
        result = episode.step(
            _frame(
                index * 50,
                x=index / 50 if case == "slow_progress" else 0,
                y=16 if case == "off_track" else 0,
            ),
            events=RewardEventWindow((index - 1) * 50, index * 50, False, False),
            issued_steering=0,
        )
        if result.terminated:
            break
    assert result.reason == case
    assert index == {"off_track": 1, "no_progress": 100, "slow_progress": 120}[case]


def test_finish_requires_historical_progress_and_minimum_steps() -> None:
    episode = NativeRewardEpisode(_trajectory(100), _config())
    episode.reset(_frame(0))
    for index in range(1, 51):
        result = episode.step(
            _frame(index * 50, 100 if index > 1 else 20),
            events=RewardEventWindow((index - 1) * 50, index * 50, False, True),
            issued_steering=0,
        )
        assert result.terminated == (index == 50)
    assert result.reason == "finished"


def test_two_actor_reward_histories_and_reset_are_independent() -> None:
    first = NativeRewardEpisode(_trajectory(), _config())
    second = NativeRewardEpisode(_trajectory(), _config())
    first.reset(_frame(0))
    second.reset(_frame(0))
    _step(first, 50)
    second.step(_frame(50), events=RewardEventWindow(0, 50, True, False), issued_steering=0)
    assert first.progress_m == 2
    assert second.progress_m == 0
    first.reset(_frame(0))
    result = _step(second, 100)
    assert not result.collision_detected
    assert second.progress_m == 5


@pytest.mark.parametrize(
    "args",
    [(False, 50, False, False), (0, 100, False, False), (0, 50, 0, False), (0, 50, False, 1)],
)
def test_bad_event_domains_and_boolean_lookalikes_are_rejected(
    args: tuple[int, int, bool | int, bool | int],
) -> None:
    with pytest.raises((ValueError, TypeError)):
        RewardEventWindow(*args)


def test_short_reward_horizon_cannot_silently_change_the_task() -> None:
    with pytest.raises(ValueError, match="150s"):
        NativeRewardEpisode(_trajectory(), _config(maximum_race_time_s=40))


def test_reward_failure_requires_reset_instead_of_reusing_partial_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = NativeRewardEpisode(_trajectory(), _config())
    episode.reset(_frame(0))
    original = TrajectoryReward.step

    def fail_after_scoring(reward: TrajectoryReward, transition: TransitionInput) -> RewardResult:
        original(reward, transition)
        raise RuntimeError("injected failure after reward history mutation")

    with monkeypatch.context() as patch:
        patch.setattr(TrajectoryReward, "step", fail_after_scoring)
        with pytest.raises(RuntimeError, match="history mutation"):
            _step(episode, 50)
    with pytest.raises(RuntimeError, match="reset"):
        _step(episode, 50)
    episode.reset(_frame(0))
    assert _step(episode, 50).reward == pytest.approx(0.134)
