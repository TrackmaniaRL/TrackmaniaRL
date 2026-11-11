"""Checkpoint benchmark command and shared benchmark reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil, isfinite, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from trackmaniarl.commands.helpers import _training_learner_state
from trackmaniarl.core.runtime import ResolvedRun, prepare_run, record_run_attempt, resolve_run
from trackmaniarl.core.spec import EvaluationSuiteSpec, RunSpec
from trackmaniarl.trackmania.recording import RecordingOptions, record_window


@dataclass(frozen=True, slots=True)
class _BenchmarkArtifact:
    trials: list[dict[str, Any]]
    checkpoint: str | None
    evaluation: EvaluationSuiteSpec
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class _BenchmarkGate:
    trials: int
    completed: int
    finish_times_valid: bool
    required: int
    median_s: float
    mean_s: float | None
    maximum_step_ms: float | None
    runtime_trials_valid: bool
    skipped_telemetry_trials: int
    evaluation: EvaluationSuiteSpec


def _benchmark(args: argparse.Namespace) -> None:
    stop_file = getattr(args, "stop_file", None)
    if stop_file is not None and stop_file.exists():
        raise ValueError(f"Stop file already exists; remove it before starting: {stop_file}")
    spec, evaluation = _benchmark_spec(args)
    run = resolve_run(spec, base_dir=Path(args.config).parent)
    recording = (
        RecordingOptions(args.record, args.window_title, args.ffmpeg)
        if getattr(args, "record", None) is not None
        else None
    )
    try:
        if stop_file is not None:
            _configure_evaluation_stop(run, stop_file.resolve())
        with record_window(recording):
            trials, metrics, checkpoint = _evaluate_checkpoint(run, args.checkpoint)
    finally:
        run.logger.close()
    artifact = _BenchmarkArtifact(trials, checkpoint, evaluation, args.checkpoint)
    _validate_benchmark_artifact(artifact)
    _print_benchmark_report(trials, metrics)
    _apply_benchmark_gate(trials, metrics, evaluation)


def _configure_evaluation_stop(run: ResolvedRun, stop_file: Path) -> None:
    set_stop_requested = getattr(run.evaluator, "set_stop_requested", None)
    if not callable(set_stop_requested):
        raise ValueError("This evaluator does not support --stop-file cancellation")
    set_stop_requested(stop_file.exists)


def _benchmark_spec(args: argparse.Namespace) -> tuple[RunSpec, EvaluationSuiteSpec]:
    spec = RunSpec.from_yaml(args.config)
    evaluation = _require_evaluation(spec, "benchmark")
    updates = _evaluation_overrides(args)
    if updates:
        evaluation = EvaluationSuiteSpec.model_validate({**evaluation.model_dump(), **updates})
        spec = spec.model_copy(update={"evaluation": evaluation})
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    spec = spec.model_copy(update={"run_id": f"{spec.run_id}-benchmark-{stamp}"})
    return spec, evaluation


def _evaluation_overrides(args: argparse.Namespace) -> dict[str, float | int | bool]:
    candidates = (
        ("trials_per_map", getattr(args, "trials", None)),
        ("target_median_s", getattr(args, "target_median", None)),
        ("target_mean_s", getattr(args, "target_mean", None)),
        ("max_step_race_time_ms", getattr(args, "max_step_race_time_ms", None)),
        ("min_finish_rate", getattr(args, "min_finish_rate", None)),
    )
    updates = {key: value for key, value in candidates if value is not None}
    if getattr(args, "reject_telemetry_skips", False):
        updates["reject_telemetry_skips"] = True
    return updates


def _require_evaluation(spec: RunSpec, command: str) -> EvaluationSuiteSpec:
    evaluation = spec.evaluation
    if evaluation is None or not evaluation.maps:
        raise ValueError(f"{command} requires an evaluation suite with at least one map")
    return evaluation


def _evaluate_checkpoint(
    run: ResolvedRun, checkpoint_path: Path
) -> tuple[list[dict[str, Any]], dict[str, float], str | None]:
    if run.evaluator is None:
        raise ValueError("benchmark requires components.evaluator")
    prepare_run(run)
    _load_checkpoint(run, checkpoint_path)
    record_run_attempt(run)
    metrics = dict(run.evaluator.evaluate(run.learner.policy()))
    artifact = _load_evaluation_artifact(run.run_dir)
    return _artifact_trials(artifact), metrics, _artifact_checkpoint(artifact)


def _load_checkpoint(run: ResolvedRun, checkpoint_path: Path) -> None:
    set_checkpoint = getattr(run.evaluator, "set_checkpoint", None)
    if callable(set_checkpoint):
        set_checkpoint(checkpoint_path)
    run.learner.setup(
        {
            "seed": run.spec.seed,
            "run_dir": run.run_dir,
            "model_factory": run.model_factory,
            "restoring_checkpoint": True,
        }
    )
    checkpoint = run.checkpoint_codec.load(checkpoint_path)
    learner_state = _training_learner_state(checkpoint)
    load_policy_state = getattr(run.learner, "load_policy_state_dict", None)
    if callable(load_policy_state):
        load_policy_state(learner_state)
    else:
        run.learner.load_state_dict(learner_state)


def _load_evaluation_artifact(run_dir: Path) -> dict[str, Any]:
    raw = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("evaluation.json must contain an object")
    return cast(dict[str, Any], raw)


def _artifact_trials(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    trials = artifact["trials"]
    if not isinstance(trials, list) or not all(isinstance(trial, dict) for trial in trials):
        raise TypeError("evaluation.json trials must be a list of objects")
    return cast(list[dict[str, Any]], trials)


def _artifact_checkpoint(artifact: dict[str, Any]) -> str | None:
    checkpoint = artifact["checkpoint"]
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise TypeError("evaluation.json checkpoint must be a string or null")
    return checkpoint


def _validate_benchmark_artifact(artifact: _BenchmarkArtifact) -> None:
    if artifact.checkpoint != str(artifact.checkpoint_path):
        raise RuntimeError("benchmark artifact checkpoint does not match the evaluated checkpoint")
    trials = artifact.trials
    evaluation = artifact.evaluation
    expected_trials = evaluation.trials_per_map * len(evaluation.maps)
    expected_maps = {item.id for item in evaluation.maps}
    observed_maps = {str(trial["map_id"]) for trial in trials}
    if len(trials) != expected_trials or observed_maps != expected_maps:
        raise RuntimeError(
            f"benchmark artifact must contain exactly {expected_trials} trials covering "
            f"{sorted(expected_maps)}"
        )
    expected = {
        (item.id, index) for item in evaluation.maps for index in range(evaluation.trials_per_map)
    }
    observed = {(trial["map_id"], trial["trial_index"]) for trial in trials}
    if expected != observed:
        raise RuntimeError("benchmark artifact contains duplicate or missing trial identities")
    map_uids = {item.id: item.expected_map_uid for item in evaluation.maps}
    if any(trial["map_uid"] != map_uids[trial["map_id"]] for trial in trials):
        raise RuntimeError("benchmark artifact contains a mismatched map UID")


def _apply_benchmark_gate(
    trials: list[dict[str, Any]],
    metrics: dict[str, float],
    evaluation: EvaluationSuiteSpec,
) -> None:
    gate = _benchmark_gate(trials, metrics, evaluation)
    if not _benchmark_gate_passed(gate) or _has_runtime_errors(trials):
        raise RuntimeError(_benchmark_gate_failure(gate))
    print(_benchmark_gate_success(gate))


def _benchmark_gate(
    trials: list[dict[str, Any]],
    metrics: dict[str, float],
    evaluation: EvaluationSuiteSpec,
) -> _BenchmarkGate:
    trial_count = len(trials)
    maximum_step_ms, runtime_trials_valid = _runtime_trial_statistics(trials)
    completed = _completed_trials(trials)
    return _BenchmarkGate(
        trials=trial_count,
        completed=len(completed),
        finish_times_valid=bool(completed)
        and all(
            _positive_finite_number(trial.get("finish_time_s")) is not None for trial in completed
        ),
        required=ceil(evaluation.min_finish_rate * trial_count),
        median_s=float(metrics["eval/median_finish_time_s"]),
        mean_s=_gated_metric(metrics, "eval/finish_time_s", evaluation.target_mean_s),
        maximum_step_ms=maximum_step_ms,
        runtime_trials_valid=runtime_trials_valid,
        skipped_telemetry_trials=len(_telemetry_skipped_trials(trials)),
        evaluation=evaluation,
    )


def _gated_metric(metrics: dict[str, float], key: str, target: float | None) -> float | None:
    value = metrics.get(key)
    return None if target is None or value is None else float(value)


def _runtime_trial_statistics(trials: list[dict[str, Any]]) -> tuple[float | None, bool]:
    measurements: list[float] = []
    valid = True
    for trial in trials:
        measurement = _positive_finite_number(trial.get("step_race_time_ms_max"))
        if measurement is None:
            valid = False
        else:
            measurements.append(measurement)
        valid = _runtime_trial_measurements_valid(trial) and valid
    return (max(measurements) if measurements else None), valid


def _runtime_trial_measurements_valid(trial: dict[str, Any]) -> bool:
    steps = trial.get("steps")
    measurement_count = trial.get("step_race_time_measurement_count")
    return (
        _positive_integer(steps)
        and _nonnegative_integer(measurement_count)
        and measurement_count == steps
        and trial.get("step_race_time_measurements_valid") is True
    )


def _positive_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    number = float(value)
    return number if isfinite(number) and number > 0.0 else None


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _benchmark_gate_passed(gate: _BenchmarkGate) -> bool:
    evaluation = gate.evaluation
    mean_ok = evaluation.target_mean_s is None or (
        gate.finish_times_valid
        and _positive_finite_number(gate.mean_s) is not None
        and gate.mean_s is not None
        and gate.mean_s < evaluation.target_mean_s
    )
    runtime_ok = evaluation.max_step_race_time_ms is None or (
        gate.runtime_trials_valid
        and gate.maximum_step_ms is not None
        and 0.0 < gate.maximum_step_ms <= evaluation.max_step_race_time_ms
    )
    telemetry_ok = not evaluation.reject_telemetry_skips or gate.skipped_telemetry_trials == 0
    median_ok = evaluation.target_median_s is None or (
        gate.finish_times_valid
        and _positive_finite_number(gate.median_s) is not None
        and gate.median_s < evaluation.target_median_s
    )
    finishes_ok = (
        gate.trials > 0
        and gate.completed >= gate.required
        and (gate.completed == 0 or gate.finish_times_valid)
    )
    return finishes_ok and median_ok and mean_ok and runtime_ok and telemetry_ok


def _benchmark_gate_failure(gate: _BenchmarkGate) -> str:
    evaluation = gate.evaluation
    return (
        "benchmark failed: require "
        f">={gate.required}/{gate.trials} finishes, "
        f"{_median_requirement(evaluation)}"
        f"{_mean_requirement(evaluation)}"
        f"{_runtime_requirement(evaluation)}"
        f"{_telemetry_skip_requirement(gate)}"
        "and no telemetry/controller errors"
    )


def _benchmark_gate_success(gate: _BenchmarkGate) -> str:
    summary = (
        f"Benchmark passed: {gate.completed}/{gate.trials} finishes, median {gate.median_s:.3f}s"
    )
    if gate.mean_s is not None:
        summary += f", mean {gate.mean_s:.3f}s"
    if gate.maximum_step_ms is not None:
        summary += f", maximum race-clock step {gate.maximum_step_ms:.3f}ms"
    return summary


def _mean_requirement(evaluation: EvaluationSuiteSpec) -> str:
    target = evaluation.target_mean_s
    return "" if target is None else f"mean completed time <{target}s, "


def _runtime_requirement(evaluation: EvaluationSuiteSpec) -> str:
    maximum = evaluation.max_step_race_time_ms
    return "" if maximum is None else f"maximum race-clock step <={maximum}ms, "


def _telemetry_skip_requirement(gate: _BenchmarkGate) -> str:
    if not gate.evaluation.reject_telemetry_skips:
        return ""
    return "zero skipped telemetry frames with complete measurements in every trial, "


def _median_requirement(evaluation: EvaluationSuiteSpec) -> str:
    target = evaluation.target_median_s
    return "" if target is None else f"median completed time <{target}s, "


def _completed_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial["finished"]]


def _telemetry_skipped_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if _skipped_telemetry_frames(trial) > 0]


def _skipped_telemetry_frames(trial: dict[str, Any]) -> int:
    steps = trial.get("steps")
    count = trial.get("telemetry_skipped_frames_measurement_count")
    value = trial.get("telemetry_skipped_frames_total")
    if not (
        _positive_integer(steps)
        and _nonnegative_integer(count)
        and count == steps
        and trial.get("telemetry_skipped_frames_measurements_valid") is True
        and not isinstance(value, bool)
        and isinstance(value, int | float)
        and isfinite(value)
        and value >= 0
        and value % 1 == 0
    ):
        return 1
    return int(value)


def _has_runtime_errors(trials: list[dict[str, Any]]) -> bool:
    return any(
        trial["telemetry_error"] is not None or trial["controller_error"] is not None
        for trial in trials
    )


def _print_benchmark_report(trials: list[dict[str, Any]], metrics: dict[str, float]) -> None:
    completed = _completed_trials(trials)
    print("Benchmark trials:")
    for trial in trials:
        _print_benchmark_trial(trial)
    print(
        f"Benchmark summary: finishes={len(completed)}/{len(trials)}, "
        f"mean_completed={float(metrics['eval/finish_time_s']):.3f}s, "
        f"median_completed={float(metrics['eval/median_finish_time_s']):.3f}s"
    )
    _print_benchmark_confidence(completed, len(trials))


def _print_benchmark_trial(trial: dict[str, Any]) -> None:
    finish_time = trial["finish_time_s"]
    time_text = "-" if finish_time is None else f"{float(finish_time):.3f}s"
    print(
        f"  trial={trial['trial_index']} map={trial['map_id']} "
        f"finished={trial['finished']} time={time_text} "
        f"progress={float(trial['progress_pct']):.1f}% "
        f"telemetry_error={trial['telemetry_error'] or '-'} "
        f"controller_error={trial['controller_error'] or '-'}"
    )


def _print_benchmark_confidence(completed: list[dict[str, Any]], trial_count: int) -> None:
    finish_low, finish_high = _wilson_interval(len(completed), trial_count)
    finish_times = [float(trial["finish_time_s"]) for trial in completed]
    interval = _bootstrap_median_interval(finish_times)
    median_text = "n/a" if interval is None else f"[{interval[0]:.3f}s, {interval[1]:.3f}s]"
    print(
        f"Benchmark 95% CI: finish_rate Wilson=[{finish_low:.4f}, {finish_high:.4f}], "
        f"median_completed bootstrap={median_text}"
    )


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials or not 0.0 < confidence < 1.0:
        raise ValueError("Wilson interval requires valid successes, trials, and confidence")
    probability = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * sqrt(probability * (1.0 - probability) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _bootstrap_median_interval(
    values: list[float], confidence: float = 0.95, samples: int = 10_000
) -> tuple[float, float] | None:
    if not values:
        return None
    if samples < 1 or not 0.0 < confidence < 1.0 or not np.isfinite(values).all():
        raise ValueError("bootstrap interval requires finite values, samples, and confidence")
    observed = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(0)
    indices = generator.integers(0, len(observed), size=(samples, len(observed)))
    medians = np.median(observed[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(medians, (tail, 1.0 - tail))
    return float(low), float(high)
