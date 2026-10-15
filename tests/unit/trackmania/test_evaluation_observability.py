from __future__ import annotations

import pytest

from trackmaniarl.experiments.evaluation import EvaluationResult, aggregate_results
from trackmaniarl.trackmania.evaluation_observability import EvaluationObservability


def _result() -> EvaluationResult:
    return EvaluationResult(False, None, False, 0.0, 0.0, 0.0)


def test_issued_commands_distinguish_no_gas_from_full_brake_and_missing_evidence() -> None:
    observability = EvaluationObservability()
    observability.record({"control/issued_command": [0, 0, -1 / 3]}, 1)
    observability.record({"control/issued_command": [0, 1, 0]}, 1)
    result = observability.annotated_result(_result(), 2)
    assert result.issued_gas_mean == 0
    assert result.issued_control_measurements_valid
    assert len(result.issued_control_counts) == 2
    observability.record({"control/issued_command": [1, -1, 0]}, 1)
    assert sum(result.issued_control_counts.values()) == 2
    missing = EvaluationObservability()
    missing.record({"control_gas": 0, "input_gas": 0}, 1)
    unknown = missing.annotated_result(_result(), 1)
    assert unknown.issued_gas_mean is None
    assert not unknown.issued_control_measurements_valid


@pytest.mark.parametrize("control", [[True, 0, 0], [2, 0, 0], [0, 0, float("nan")], [0, 0]])
def test_invalid_issued_controls_are_not_counted(control: list) -> None:
    observability = EvaluationObservability()
    observability.record({"control/issued_command": control}, 1)
    result = observability.annotated_result(_result(), 1)
    assert result.issued_control_measurement_count == 0
    assert result.issued_gas_mean is None


def test_step_race_time_measurements_are_counted_when_every_step_is_valid() -> None:
    observability = EvaluationObservability()
    observability.record({"step_race_time_ms": 40.0}, 1.0)
    observability.record({"step_race_time_ms": 60.0}, 1.0)

    result = observability.annotated_result(_result(), step_count=2)

    assert result.step_race_time_measurement_count == 2
    assert result.step_race_time_measurements_valid is True
    assert result.step_race_time_ms_p99 == 60.0
    assert result.step_race_time_ms_max == 60.0


@pytest.mark.parametrize("invalid", [None, 0.0, -1.0, float("nan"), float("inf"), "50"])
def test_step_race_time_measurements_reject_every_invalid_individual_step(
    invalid: object,
) -> None:
    observability = EvaluationObservability()
    observability.record({"step_race_time_ms": 50.0}, 1.0)
    observability.record({"step_race_time_ms": invalid}, 1.0)

    result = observability.annotated_result(_result(), step_count=2)

    assert result.step_race_time_measurement_count == 1
    assert result.step_race_time_measurements_valid is False
    assert result.step_race_time_ms_p99 == 50.0
    assert result.step_race_time_ms_max == 50.0


def test_finish_flag_at_same_race_clock_is_a_measured_zero_interval() -> None:
    observability = EvaluationObservability()
    observability.record({"step_race_time_ms": 50.0}, 1.0)
    observability.record(
        {"step_race_time_ms": 0.0, "termination_reason": "finished", "telemetry_health": "ok"}, 1.0
    )
    result = observability.annotated_result(_result(), step_count=2)
    assert result.step_race_time_measurement_count == 2
    assert result.step_race_time_measurements_valid is True
    assert result.step_race_time_ms_max == 50.0


@pytest.mark.parametrize("value", [-1.0, False, None, float("nan")])
def test_finish_does_not_excuse_invalid_or_regressing_clock(value: object) -> None:
    observability = EvaluationObservability()
    observability.record({"step_race_time_ms": 50.0}, 1.0)
    observability.record(
        {"step_race_time_ms": value, "termination_reason": "finished", "telemetry_health": "ok"},
        1.0,
    )
    result = observability.annotated_result(_result(), step_count=2)
    assert result.step_race_time_measurements_valid is False


def test_camera_missing_measurements_are_not_counted_as_zero_and_tails_are_per_episode() -> None:
    observability = EvaluationObservability()
    for index in range(20):
        info = {"vision/capture_ms": 2.0, "vision/pairing_delay_ms": 3.0}
        if index == 19:
            info = {"vision/capture_ms": float("nan"), "vision/pairing_delay_ms": -1.0}
        observability.record(info, float(index + 1))
    camera = observability.annotated_result(_result(), step_count=20)
    assert camera.vision_capture_measurement_count == 19
    assert camera.vision_capture_ms == 2.0
    assert camera.vision_pairing_measurement_count == 19
    assert camera.vision_pairing_delay_ms == 3.0
    assert camera.action_latency_ms_p95 == 19.0
    metrics = aggregate_results([camera, _result()])
    assert metrics["eval/vision_capture_measurement_count"] == 19.0
    assert metrics["eval/vision_capture_ms"] == 2.0
    assert metrics["eval/vision_pairing_delay_ms"] == 3.0
    assert metrics["eval/action_latency_ms_p95_max"] == 19.0
