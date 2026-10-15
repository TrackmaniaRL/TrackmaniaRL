"""Versioned TrackMania evaluation-suite models and standard score calculation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import mean, median

STANDARD_METRICS = (
    "eval/finish_rate",
    "eval/finish_time_s",
    "eval/median_finish_time_s",
    "eval/best_finish_time_s",
    "eval/sub40_rate",
    "eval/sub38_rate",
    "eval/sub36_rate",
    "eval/failure_progress_mean_pct",
    "eval/crash_rate",
    "eval/reward",
    "eval/action_latency_ms",
    "eval/action_latency_ms_p95_max",
    "eval/vision_capture_ms",
    "eval/vision_capture_ms_p95_max",
    "eval/vision_capture_measurement_count",
    "eval/vision_pairing_delay_ms",
    "eval/vision_pairing_delay_ms_p95_max",
    "eval/vision_pairing_measurement_count",
    "eval/controller_apply_ms",
    "eval/telemetry_wait_ms",
    "eval/control_brake_tap_fraction",
    "eval/step_race_time_ms_p99",
    "eval/step_race_time_ms_max",
    "eval/step_race_time_measurement_count",
    "eval/step_race_time_expected_measurement_count",
    "eval/step_race_time_measurements_valid",
    "eval/telemetry_skipped_frames_total",
    "eval/telemetry_skipped_frames_mean",
    "eval/telemetry_skipped_frames_max",
    "eval/telemetry_steps_with_skipped_frames_fraction",
    "eval/throughput_fps",
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One episode's normalized outcome, independent of the tracker backend."""

    finished: bool
    finish_time_s: float | None
    crashed: bool
    reward: float
    action_latency_ms: float
    throughput_fps: float
    progress_pct: float = 0.0
    map_id: str = ""
    map_uid: str = ""
    trial_index: int = 0
    telemetry_error: str | None = None
    controller_error: str | None = None
    progress_bins: Mapping[str, Mapping[str, float | None]] | None = None
    steps: int = 0
    controller_apply_ms: float = 0.0
    telemetry_wait_ms: float = 0.0
    control_brake_tap_fraction: float = 0.0
    issued_control_counts: Mapping[str, int] | None = None
    issued_control_measurement_count: int = 0
    issued_control_measurements_valid: bool = False
    issued_gas_mean: float | None = None
    step_race_time_ms_p99: float = 0.0
    step_race_time_ms_max: float = 0.0
    step_race_time_measurement_count: int = 0
    step_race_time_measurements_valid: bool = False
    telemetry_skipped_frames_total: int = 0
    telemetry_skipped_frames_mean: float = 0.0
    telemetry_skipped_frames_max: int = 0
    telemetry_steps_with_skipped_frames_fraction: float = 0.0
    termination_reason: str = "unknown"
    finish_time_source: str | None = None
    action_latency_ms_p95: float = 0.0
    vision_capture_ms: float = 0.0
    vision_capture_ms_p95: float = 0.0
    vision_capture_measurement_count: int = 0
    vision_pairing_delay_ms: float = 0.0
    vision_pairing_delay_ms_p95: float = 0.0
    vision_pairing_measurement_count: int = 0


@dataclass(frozen=True, slots=True)
class _EvaluationSummary:
    values: list[EvaluationResult]
    finished_times: list[float]
    failed_progress: list[float]
    total_steps: int
    skipped_frames: int


def _summarize(values: list[EvaluationResult]) -> _EvaluationSummary:
    finished_times = [item.finish_time_s for item in values if item.finished and item.finish_time_s]
    return _EvaluationSummary(
        values,
        finished_times,
        [item.progress_pct for item in values if not item.finished],
        sum(item.steps for item in values),
        sum(item.telemetry_skipped_frames_total for item in values),
    )


def _threshold_rate(values: list[EvaluationResult], threshold: float) -> float:
    return mean(
        float(item.finished and item.finish_time_s is not None and item.finish_time_s < threshold)
        for item in values
    )


def _finish_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    times = summary.finished_times
    return {
        "eval/finish_rate": mean(float(item.finished) for item in summary.values),
        "eval/finish_time_s": mean(times) if times else 0.0,
        "eval/median_finish_time_s": median(times) if times else 0.0,
        "eval/best_finish_time_s": min(times) if times else 0.0,
        "eval/sub40_rate": _threshold_rate(summary.values, 40.0),
        "eval/sub38_rate": _threshold_rate(summary.values, 38.0),
        "eval/sub36_rate": _threshold_rate(summary.values, 36.0),
    }


def _step_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    metrics = _control_loop_metrics(summary)
    metrics.update(_skipped_frame_metrics(summary))
    metrics.update(_vision_timing_metrics(summary.values))
    return metrics


def _control_loop_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    values = summary.values
    steps = summary.total_steps
    return {
        "eval/action_latency_ms": _step_weighted_mean(values, "action_latency_ms", steps),
        "eval/action_latency_ms_p95_max": max(item.action_latency_ms_p95 for item in values),
        "eval/controller_apply_ms": _step_weighted_mean(values, "controller_apply_ms", steps),
        "eval/telemetry_wait_ms": _step_weighted_mean(values, "telemetry_wait_ms", steps),
        "eval/control_brake_tap_fraction": _step_weighted_mean(
            values, "control_brake_tap_fraction", steps
        ),
        **_step_race_time_metrics(values, steps),
    }


def _vision_timing_metrics(values: list[EvaluationResult]) -> dict[str, float]:
    """Report measured means and worst episode tails; absent cameras have zero counts."""
    metrics = {}
    for prefix, count_field in (
        ("vision_capture", "vision_capture_measurement_count"),
        ("vision_pairing_delay", "vision_pairing_measurement_count"),
    ):
        count = sum(getattr(item, count_field) for item in values)
        measured_sum = sum(
            getattr(item, f"{prefix}_ms") * getattr(item, count_field) for item in values
        )
        metrics[f"eval/{prefix}_ms"] = measured_sum / count if count else 0.0
        metrics[f"eval/{prefix}_ms_p95_max"] = max(
            getattr(item, f"{prefix}_ms_p95") for item in values
        )
        metrics[f"eval/{count_field}"] = float(count)
    return metrics


def _step_race_time_metrics(values: list[EvaluationResult], steps: int) -> dict[str, float]:
    return {
        "eval/step_race_time_ms_p99": max(item.step_race_time_ms_p99 for item in values),
        "eval/step_race_time_ms_max": max(item.step_race_time_ms_max for item in values),
        "eval/step_race_time_measurement_count": float(
            sum(item.step_race_time_measurement_count for item in values)
        ),
        "eval/step_race_time_expected_measurement_count": float(steps),
        "eval/step_race_time_measurements_valid": float(_step_race_time_measurements_valid(values)),
    }


def _step_race_time_measurements_valid(values: list[EvaluationResult]) -> bool:
    return all(
        item.steps > 0
        and item.step_race_time_measurements_valid
        and item.step_race_time_measurement_count == item.steps
        for item in values
    )


def _skipped_frame_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    values = summary.values
    steps = summary.total_steps
    skipped = summary.skipped_frames
    return {
        "eval/telemetry_skipped_frames_total": float(skipped),
        "eval/telemetry_skipped_frames_mean": skipped / steps if steps else 0.0,
        "eval/telemetry_skipped_frames_max": float(
            max(item.telemetry_skipped_frames_max for item in values)
        ),
        "eval/telemetry_steps_with_skipped_frames_fraction": _step_weighted_mean(
            values, "telemetry_steps_with_skipped_frames_fraction", steps
        ),
    }


def _outcome_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    values = summary.values
    progress = mean(summary.failed_progress) if summary.failed_progress else 100.0
    return {
        "eval/failure_progress_mean_pct": progress,
        "eval/crash_rate": mean(float(item.crashed) for item in values),
        "eval/reward": mean(item.reward for item in values),
    }


def aggregate_results(results: Iterable[EvaluationResult]) -> Mapping[str, float]:
    values = list(results)
    if not values:
        raise ValueError("An evaluation suite must contain at least one episode result")
    summary = _summarize(values)
    metrics = _finish_metrics(summary)
    metrics.update(_outcome_metrics(summary))
    metrics.update(_step_metrics(summary))
    metrics["eval/throughput_fps"] = mean(item.throughput_fps for item in values)
    return metrics


def _step_weighted_mean(values: list[EvaluationResult], field: str, total_steps: int) -> float:
    if not total_steps:
        return 0.0
    return sum(float(getattr(item, field)) * item.steps for item in values) / total_steps
