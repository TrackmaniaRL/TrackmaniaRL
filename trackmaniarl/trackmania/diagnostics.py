"""Progress-binned policy diagnostics for TrackMania runs.

``action_entropy`` is a compatibility alias for ``action_histogram_entropy_normalized``:
entropy of the visited action histogram divided by log(action_count), not the
policy's conditional entropy in nats. Missing Q measurements remain null.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.expert_diagnostics import (
    ExpertActionDiagnostics,
    ExpertDiagnosticRecord,
    aggregate_expert_actions,
    aggregate_expert_bins,
)

__all__ = [
    "ExpertActionDiagnostics",
    "ExpertDiagnosticRecord",
    "ProgressBinDiagnostics",
    "ProgressDiagnosticRecord",
    "aggregate_expert_actions",
    "aggregate_expert_bins",
    "aggregate_progress_bins",
]

_TIMING_METRICS = (
    "entry_time_s",
    "reference_time_s",
    "time_debt_s",
    "projected_velocity_mps_mean",
    "step_race_ms_mean",
    "step_race_ms_p50",
    "step_race_ms_p95",
    "step_race_ms_max",
    "decision_interval_abs_error_ms_mean",
    "decision_interval_abs_error_ms_p95",
    "decision_interval_abs_error_ms_max",
    "action_switch_rate",
    "steer_switch_rate",
)


@dataclass(frozen=True, slots=True)
class ProgressDiagnosticRecord:
    progress_pct: float
    action: Any
    policy: Any
    info: Mapping[str, Any] | None = None


class ProgressBinDiagnostics:
    def __init__(self, action_count: int, bin_count: int = 10) -> None:
        if action_count < 2 or bin_count < 1:
            raise ValueError("action_count must be at least two and bin_count must be positive")
        self.action_count = action_count
        self.bin_count = bin_count
        self._initialize_action_metrics()
        self._initialize_timing_metrics()
        self._initialize_switch_metrics()

    def _initialize_action_metrics(self) -> None:
        self._actions = [[0] * self.action_count for _ in range(self.bin_count)]
        self._q_margin_totals = [0.0] * self.bin_count
        self._q_margin_minimums = [float("inf")] * self.bin_count
        self._q_margin_samples = [0] * self.bin_count
        self._q_max_totals = [0.0] * self.bin_count
        self._q_max_samples = [0] * self.bin_count

    def _initialize_timing_metrics(self) -> None:
        self._entry_time_s: list[float | None] = [None] * self.bin_count
        self._reference_time_s: list[float | None] = [None] * self.bin_count
        self._time_debt_s: list[float | None] = [None] * self.bin_count
        self._projected_velocity_totals = [0.0] * self.bin_count
        self._step_race_ms: list[list[float]] = [[] for _ in range(self.bin_count)]
        self._decision_interval_errors: list[list[float]] = [[] for _ in range(self.bin_count)]

    def _initialize_switch_metrics(self) -> None:
        self._action_switches = [0] * self.bin_count
        self._steer_switches = [0] * self.bin_count
        self._previous_action: int | None = None
        self._previous_steer: float | None = None

    def record(self, record: ProgressDiagnosticRecord) -> None:
        index = self._index(record.progress_pct)
        action_index = _diagnostic_action_index(record.action, self.action_count)
        if action_index is not None:
            self._actions[index][action_index] += 1
            if self._previous_action is not None and action_index != self._previous_action:
                self._action_switches[index] += 1
            self._previous_action = action_index
        self._record_q(index, record.policy)
        if record.info is not None:
            self._record_telemetry(index, record.info)

    def _record_q(self, index: int, policy: Any) -> None:
        margin = getattr(policy, "last_q_margin", None)
        if margin is not None:
            value = float(margin)
            self._q_margin_totals[index] += value
            self._q_margin_minimums[index] = min(self._q_margin_minimums[index], value)
            self._q_margin_samples[index] += 1
        maximum = getattr(policy, "last_q_max", None)
        if maximum is not None:
            self._q_max_totals[index] += float(maximum)
            self._q_max_samples[index] += 1

    def _record_telemetry(self, index: int, info: Mapping[str, Any]) -> None:
        if self._entry_time_s[index] is None:
            self._entry_time_s[index] = float(info.get("race_time_ms", 0.0)) / 1_000.0
            self._reference_time_s[index] = float(info.get("reference_time_s", 0.0))
            self._time_debt_s[index] = float(info.get("time_debt_s", 0.0))
        self._projected_velocity_totals[index] += float(info.get("projected_velocity_mps", 0.0))
        self._step_race_ms[index].append(float(info.get("step_race_time_ms", 0.0)))
        self._decision_interval_errors[index].append(
            abs(float(info.get("decision_interval_error_ms", 0.0)))
        )
        steer = float(info.get("control_steer", 0.0))
        if self._previous_steer is not None and steer != self._previous_steer:
            self._steer_switches[index] += 1
        self._previous_steer = steer

    def summary(self) -> dict[str, dict[str, float | None]]:
        return {self._name(index): self._bin_summary(index) for index in range(self.bin_count)}

    def flat_summary(self) -> dict[str, float | None]:
        return {
            f"progress_bin/{name}/{metric}": value
            for name, metrics in self.summary().items()
            for metric, value in metrics.items()
        }

    def _index(self, progress_pct: float) -> int:
        bounded = min(max(progress_pct, 0.0), 100.0)
        return min(self.bin_count - 1, int(bounded * self.bin_count / 100.0))

    def _name(self, index: int) -> str:
        start = index * 100 // self.bin_count
        end = (index + 1) * 100 // self.bin_count
        return f"{start:02d}_{end:03d}"

    def _bin_summary(self, index: int) -> dict[str, float | None]:
        counts = self._actions[index]
        samples = sum(counts)
        nonzero = [count for count in counts if count]
        summary = self._action_summary(index, samples, nonzero)
        summary.update(self._timing_summary(index))
        return summary

    def _action_summary(
        self, index: int, samples: int, nonzero: list[int]
    ) -> dict[str, float | None]:
        margin_samples = self._q_margin_samples[index]
        maximum_samples = self._q_max_samples[index]
        return {
            "action_count": float(samples),
            "action_entropy": self._entropy(samples, nonzero),
            "action_histogram_entropy_normalized": self._entropy(samples, nonzero),
            "action_coverage": len(nonzero) / self.action_count,
            "q_margin_mean": self._q_margin_totals[index] / margin_samples
            if margin_samples
            else None,
            "q_margin_min": self._q_margin_minimums[index] if margin_samples else None,
            "q_max_mean": self._q_max_totals[index] / maximum_samples if maximum_samples else None,
            "q_margin_sample_count": float(margin_samples),
            "q_max_sample_count": float(maximum_samples),
        }

    def _entropy(self, samples: int, nonzero: list[int]) -> float:
        if not samples:
            return 0.0
        entropy = -sum((count / samples) * log(count / samples) for count in nonzero)
        return entropy / log(self.action_count)

    def _timing_summary(self, index: int) -> dict[str, float]:
        durations = self._step_race_ms[index]
        if not durations:
            return {}
        count = len(durations)
        errors = self._decision_interval_errors[index]
        return {
            "entry_time_s": float(self._entry_time_s[index] or 0.0),
            "reference_time_s": float(self._reference_time_s[index] or 0.0),
            "time_debt_s": float(self._time_debt_s[index] or 0.0),
            "projected_velocity_mps_mean": self._projected_velocity_totals[index] / count,
            **_step_distribution(durations),
            **_error_distribution(errors),
            "action_switch_rate": self._action_switches[index] / count,
            "steer_switch_rate": self._steer_switches[index] / count,
        }


def _step_distribution(values: list[float]) -> dict[str, float]:
    return {
        "step_race_ms_mean": float(np.mean(values)),
        "step_race_ms_p50": float(np.quantile(values, 0.5)),
        "step_race_ms_p95": float(np.quantile(values, 0.95)),
        "step_race_ms_max": float(np.max(values)),
    }


def _error_distribution(values: list[float]) -> dict[str, float]:
    return {
        "decision_interval_abs_error_ms_mean": float(np.mean(values)),
        "decision_interval_abs_error_ms_p95": float(np.quantile(values, 0.95)),
        "decision_interval_abs_error_ms_max": float(np.max(values)),
    }


def _diagnostic_action_index(action: Any, action_count: int) -> int | None:
    if isinstance(action, (int, np.integer)):
        index = int(action)
        return index if 0 <= index < action_count else None
    if not isinstance(action, np.ndarray) or action.shape != (3,):
        return None
    canonical_count, action_table = build_brake_tap_action_table()
    if action_count != canonical_count:
        return None
    return continuous_control_to_discrete_index(action, action_table)


def aggregate_progress_bins(
    summaries: Iterable[Mapping[str, Mapping[str, float | None]]],
) -> dict[str, Any]:
    grouped = _group_progress_summaries(summaries)
    result: dict[str, Any] = {}
    for name, values in grouped.items():
        result.update(_aggregate_progress_bin(name, values))
    return result


def _group_progress_summaries(
    summaries: Iterable[Mapping[str, Mapping[str, float | None]]],
) -> dict[str, list[Mapping[str, float | None]]]:
    grouped: dict[str, list[Mapping[str, float | None]]] = {}
    for summary in summaries:
        for name, metrics in summary.items():
            grouped.setdefault(name, []).append(metrics)
    return grouped


def _aggregate_progress_bin(
    name: str, values: list[Mapping[str, float | None]]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    total = sum(float(item["action_count"] or 0.0) for item in values)
    prefix = f"progress_bin/{name}"
    result[f"{prefix}/action_count"] = total
    result.update(_weighted_progress_metrics(prefix, values))
    minima = [
        value
        for item in values
        if (value := item.get("q_margin_min")) is not None
        and float(item.get("q_margin_sample_count", item["action_count"]) or 0.0) > 0.0
    ]
    result[f"{prefix}/q_margin_min"] = min(minima) if minima else None
    result.update(_observed_progress_metrics(prefix, values))
    return result


def _weighted_progress_metrics(
    prefix: str, values: list[Mapping[str, float | None]]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("action_entropy", "action_coverage", "q_margin_mean", "q_max_mean"):
        count_metric = metric.removesuffix("_mean") + "_sample_count"
        observed = [
            (measurement, float(item.get(count_metric, item["action_count"]) or 0.0))
            for item in values
            if (measurement := item.get(metric)) is not None
        ]
        count = sum(samples for _, samples in observed)
        result[f"{prefix}/{metric}"] = (
            sum(value * samples for value, samples in observed) / count
            if count
            else (None if metric.startswith("q_") else 0.0)
        )
        if metric.startswith("q_"):
            result[f"{prefix}/{count_metric}"] = count
    result[f"{prefix}/action_histogram_entropy_normalized"] = result[f"{prefix}/action_entropy"]
    return result


def _observed_progress_metrics(
    prefix: str, values: list[Mapping[str, float | None]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in _TIMING_METRICS:
        observed = [value for item in values if (value := item.get(metric)) is not None]
        if observed:
            result[f"{prefix}/{metric}"] = float(np.mean(observed))
    return result
