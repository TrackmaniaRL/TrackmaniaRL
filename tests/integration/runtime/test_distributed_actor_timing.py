from __future__ import annotations

import pytest

from trackmaniarl.distributed.actor_metrics import (
    ControlUsageTracker,
    EpisodeMetrics,
    InferenceTimingTracker,
)


def test_control_usage_tracker_keeps_timing_when_control_metrics_are_absent() -> None:
    tracker = ControlUsageTracker()
    tracker.record(
        {
            "control_gas": 1.0,
            "control_brake": 0.0,
            "control_steer": 0.0,
            "step_race_time_ms": 50.0,
        }
    )
    tracker.record({"step_race_time_ms": 500.0})

    summary = tracker.summary()

    assert summary["step_race_time_ms_max"] == 500.0
    assert summary["step_race_time_measurement_count"] == 2.0
    assert summary["step_race_time_measurements_valid"] == 1.0


@pytest.mark.parametrize(
    "invalid_step",
    [
        {},
        {"step_race_time_ms": 0.0},
        {"step_race_time_ms": -1.0},
        {"step_race_time_ms": float("nan")},
        {"step_race_time_ms": float("inf")},
    ],
)
def test_control_usage_tracker_marks_invalid_individual_step_timing(
    invalid_step: dict[str, float],
) -> None:
    tracker = ControlUsageTracker()
    tracker.record({"step_race_time_ms": 50.0})
    tracker.record(invalid_step)

    summary = tracker.summary()

    assert summary["step_race_time_ms_max"] == 50.0
    assert summary["step_race_time_measurement_count"] == 1.0
    assert summary["step_race_time_measurements_valid"] == 0.0
    assert summary["step_race_time_invalid_measurements"] == 1.0


def test_timing_counts_distinguish_long_steps_from_missing_measurements() -> None:
    tracker = ControlUsageTracker()
    for value in (50.0, 100.0, 150.0, 0.0):
        tracker.record({"step_race_time_ms": value})
    summary = tracker.summary()
    assert summary["step_race_time_over_100ms_fraction"] == pytest.approx(1 / 3)
    assert summary["step_race_time_invalid_measurements"] == 1.0
    inference = InferenceTimingTracker()
    for value in (0.05, 0.1, 0.15):
        inference.record(value)
    assert inference.summary()["policy_inference_over_100ms_fraction"] == pytest.approx(1 / 3)


def test_actor_phase_timing_summarizes_without_changing_reward_or_actions() -> None:
    metrics = EpisodeMetrics.from_policy(object())
    metrics.record_phase("preprocessing", 0.002)
    metrics.record_phase("preprocessing", 0.006)
    metrics.record_phase("bookkeeping", 0.003)
    summary = metrics.summary_info(0.01, 17, 2)
    assert summary["actor_preprocessing_ms_mean"] == pytest.approx(4)
    assert summary["actor_preprocessing_ms_max"] == pytest.approx(6)
    assert summary["actor_bookkeeping_ms_mean"] == pytest.approx(3)
    assert metrics.total_reward == 0.0
