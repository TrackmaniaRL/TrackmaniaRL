from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any

import pytest

from trackmaniarl.commands.evaluation import _apply_benchmark_gate
from trackmaniarl.commands.parser import build_parser
from trackmaniarl.core.spec import EvaluationMapSpec, EvaluationSuiteSpec


def _suite(
    target_mean_s: float | None, max_step_race_time_ms: float | None = None
) -> EvaluationSuiteSpec:
    evaluation_map = EvaluationMapSpec(
        id="test-map",
        map_path=Path("test.Map.Gbx"),
        geometry_path=Path("test-geometry.npz"),
        expected_map_uid="test-map-uid",
    )
    return EvaluationSuiteSpec(
        name="test-suite",
        version="1",
        maps=(evaluation_map,),
        trials_per_map=10,
        target_median_s=40.0,
        target_mean_s=target_mean_s,
        max_step_race_time_ms=max_step_race_time_ms,
        min_finish_rate=1.0,
    )


def _finished_trials(times: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "finished": True,
            "finish_time_s": finish_time_s,
            "telemetry_error": None,
            "controller_error": None,
            "steps": 1,
            "step_race_time_ms_max": 50.0,
            "step_race_time_measurement_count": 1,
            "step_race_time_measurements_valid": True,
            "telemetry_skipped_frames_total": 0,
            "telemetry_skipped_frames_measurement_count": 1,
            "telemetry_skipped_frames_measurements_valid": True,
        }
        for finish_time_s in times
    ]


def _metrics(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    midpoint = len(ordered) // 2
    median_s = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "eval/finish_time_s": fmean(times),
        "eval/median_finish_time_s": median_s,
    }


@pytest.mark.parametrize(
    "times",
    [
        [36.6] * 9 + [50.0],
        [37.0] * 10,
    ],
)
def test_optional_mean_gate_is_strict(times: list[float]) -> None:
    with pytest.raises(RuntimeError, match=r"mean completed time <37\.0s"):
        _apply_benchmark_gate(_finished_trials(times), _metrics(times), _suite(37.0))


def test_gate_without_optional_mean_target_does_not_require_mean_metric(
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = {"eval/median_finish_time_s": 36.6}
    _apply_benchmark_gate(_finished_trials([36.6] * 10), metrics, _suite(None))
    assert "median 36.600s" in capsys.readouterr().out


def test_optional_mean_gate_rejects_a_missing_metric_cleanly() -> None:
    metrics = {"eval/median_finish_time_s": 36.6}
    with pytest.raises(RuntimeError, match=r"mean completed time <37\.0s"):
        _apply_benchmark_gate(_finished_trials([36.6] * 10), metrics, _suite(37.0))


def test_optional_runtime_gate_accepts_its_inclusive_boundary() -> None:
    trials = _finished_trials([36.6] * 10)
    trials[-1]["step_race_time_ms_max"] = 100.0
    _apply_benchmark_gate(trials, _metrics([36.6] * 10), _suite(None, 100.0))


def test_optional_runtime_gate_rejects_a_slow_step() -> None:
    trials = _finished_trials([36.6] * 10)
    trials[3]["step_race_time_ms_max"] = 100.01
    metrics = {**_metrics([36.6] * 10), "eval/step_race_time_ms_max": 50.0}
    with pytest.raises(RuntimeError, match=r"maximum race-clock step <=100\.0ms"):
        _apply_benchmark_gate(trials, metrics, _suite(None, 100.0))


def test_optional_runtime_gate_uses_trial_measurements_when_aggregate_metric_is_missing() -> None:
    _apply_benchmark_gate(_finished_trials([36.6] * 10), _metrics([36.6] * 10), _suite(None, 100.0))


def test_optional_runtime_gate_rejects_a_zero_placeholder() -> None:
    trials = _finished_trials([36.6] * 10)
    trials[3]["step_race_time_ms_max"] = 0.0
    with pytest.raises(RuntimeError, match=r"maximum race-clock step <=100\.0ms"):
        _apply_benchmark_gate(trials, _metrics([36.6] * 10), _suite(None, 100.0))


@pytest.mark.parametrize(
    "field",
    ["step_race_time_measurement_count", "step_race_time_measurements_valid"],
)
def test_optional_runtime_gate_rejects_an_individual_trial_without_complete_measurements(
    field: str,
) -> None:
    trials = _finished_trials([36.6] * 10)
    trials[3][field] = 0 if field.endswith("count") else False

    with pytest.raises(RuntimeError, match=r"maximum race-clock step <=100\.0ms"):
        _apply_benchmark_gate(trials, _metrics([36.6] * 10), _suite(None, 100.0))


def test_optional_runtime_gate_rejects_a_trial_without_measurement_metadata() -> None:
    trials = _finished_trials([36.6] * 10)
    trials[3].pop("step_race_time_measurements_valid")

    with pytest.raises(RuntimeError, match=r"maximum race-clock step <=100\.0ms"):
        _apply_benchmark_gate(trials, _metrics([36.6] * 10), _suite(None, 100.0))


def test_benchmark_parser_accepts_strict_telemetry_skip_rejection() -> None:
    args = build_parser().parse_args(
        ["benchmark", "config.yaml", "checkpoint.pt", "--reject-telemetry-skips"]
    )

    assert args.reject_telemetry_skips is True


def test_strict_telemetry_skip_benchmark_rejects_a_finished_trial_with_skips() -> None:
    trials = _finished_trials([36.6] * 10)
    trials[3]["telemetry_skipped_frames_total"] = 1
    evaluation = _suite(None).model_copy(update={"reject_telemetry_skips": True})

    with pytest.raises(RuntimeError, match="zero skipped telemetry frames"):
        _apply_benchmark_gate(
            trials,
            _metrics([36.6] * 10),
            evaluation,
        )


def test_strict_telemetry_gate_accepts_complete_zero_measurements() -> None:
    evaluation = _suite(None).model_copy(update={"reject_telemetry_skips": True})
    _apply_benchmark_gate(_finished_trials([36.6] * 10), _metrics([36.6] * 10), evaluation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("telemetry_skipped_frames_total", None),
        ("telemetry_skipped_frames_total", -1),
        ("telemetry_skipped_frames_total", float("nan")),
        ("telemetry_skipped_frames_total", False),
        ("telemetry_skipped_frames_measurement_count", 0),
        ("telemetry_skipped_frames_measurements_valid", False),
    ],
)
def test_strict_telemetry_gate_rejects_missing_or_invalid_measurements(
    field: str, value: object
) -> None:
    trials = _finished_trials([36.6] * 10)
    trials[3][field] = value
    evaluation = _suite(None).model_copy(update={"reject_telemetry_skips": True})
    with pytest.raises(RuntimeError, match="complete measurements"):
        _apply_benchmark_gate(trials, _metrics([36.6] * 10), evaluation)


def test_strict_telemetry_gate_rejects_legacy_trial_without_coverage() -> None:
    trials = _finished_trials([36.6] * 10)
    trials[0].pop("telemetry_skipped_frames_measurement_count")
    evaluation = _suite(None).model_copy(update={"reject_telemetry_skips": True})
    with pytest.raises(RuntimeError, match="complete measurements"):
        _apply_benchmark_gate(trials, _metrics([36.6] * 10), evaluation)
