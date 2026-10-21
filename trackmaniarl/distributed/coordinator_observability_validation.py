"""Validation rules for actor timing and control-loop observability summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

_OBSERVABILITY_FIELDS = {
    "timing/policy_inference_ms_mean",
    "timing/policy_inference_ms_max",
    "timing/step_race_ms_p99",
    "timing/step_race_ms_max",
    "timing/step_race_measurement_count",
    "control/brake_tap_fraction",
    "controller_apply_ms_mean",
    "controller_apply_ms_max",
    "telemetry_wait_ms_mean",
    "telemetry_wait_ms_max",
    "telemetry_skipped_frames_total",
    "telemetry_skipped_frames_mean",
    "telemetry_skipped_frames_max",
}


def _validate_observability_summary(value: Mapping[str, Any], name: str, steps: int) -> None:
    missing = (
        _OBSERVABILITY_FIELDS
        | {
            "timing/step_race_measurements_valid",
            "telemetry_steps_with_skipped_frames_fraction",
        }
    ) - value.keys()
    if missing:
        raise ValueError(f"{name} observability summary is missing {sorted(missing)}")
    observed = {
        key: _validate_nonnegative_number(value[key], f"{name} {key}")
        for key in _OBSERVABILITY_FIELDS
    }
    fraction = _skipped_frame_fraction(value, name)
    brake_tap_fraction = _brake_tap_fraction(value, name)
    if steps == 0 and (any(observed.values()) or fraction > 0.0 or brake_tap_fraction > 0.0):
        raise ValueError(f"{name} timing and frame metrics require at least one step")
    _validate_skipped_frame_counts(observed, name)
    _validate_step_race_time_tail(observed, name)
    _validate_step_race_time_metadata(value, name, steps)
    _validate_optional_timing(value, name, steps)


def _validate_optional_timing(value: Mapping[str, Any], name: str, steps: int) -> None:
    for phase in ("environment", "preprocessing", "diagnostics", "bookkeeping"):
        prefix = f"timing/actor_{phase}_ms_"
        if prefix + "mean" not in value and prefix + "max" not in value:
            continue
        mean = _validate_nonnegative_number(value.get(prefix + "mean"), name + " " + prefix)
        maximum = _validate_nonnegative_number(value.get(prefix + "max"), name + " " + prefix)
        if mean > maximum or (steps == 0 and maximum > 0):
            raise ValueError(f"{name} actor phase timing is inconsistent")
    for key in (
        "timing/policy_inference_over_100ms_fraction",
        "timing/step_race_over_100ms_fraction",
    ):
        if key in value:
            fraction = _validate_nonnegative_number(value[key], name + " " + key)
            if fraction > 1 or (steps == 0 and fraction > 0):
                raise ValueError(f"{name} timing fraction must be between zero and one")
    key = "timing/step_race_invalid_measurements"
    if key in value:
        count = _validate_nonnegative_number(value[key], name + " " + key)
        if not count.is_integer() or count > steps:
            raise ValueError(f"{name} invalid timing count must be an integer within step count")


def _skipped_frame_fraction(value: Mapping[str, Any], name: str) -> float:
    key = "telemetry_steps_with_skipped_frames_fraction"
    fraction = _validate_nonnegative_number(value[key], f"{name} {key}")
    if fraction > 1.0:
        raise ValueError(f"{name} {key} must be at most one")
    return fraction


def _brake_tap_fraction(value: Mapping[str, Any], name: str) -> float:
    key = "control/brake_tap_fraction"
    fraction = _validate_nonnegative_number(value[key], f"{name} {key}")
    if fraction > 1.0:
        raise ValueError(f"{name} {key} must be at most one")
    return fraction


def _validate_skipped_frame_counts(observed: Mapping[str, float], name: str) -> None:
    total = observed["telemetry_skipped_frames_total"]
    maximum = observed["telemetry_skipped_frames_max"]
    if not total.is_integer():
        raise ValueError(f"{name} skipped frame total must be an integer")
    if not maximum.is_integer():
        raise ValueError(f"{name} skipped frame maximum must be an integer")
    if maximum > total:
        raise ValueError(f"{name} skipped frame maximum cannot exceed its total")


def _validate_step_race_time_tail(observed: Mapping[str, float], name: str) -> None:
    p99 = observed["timing/step_race_ms_p99"]
    maximum = observed["timing/step_race_ms_max"]
    if p99 > maximum:
        raise ValueError(f"{name} step race-time p99 cannot exceed its maximum")


def _validate_step_race_time_metadata(value: Mapping[str, Any], name: str, steps: int) -> None:
    count = _step_race_time_measurement_count(value, name, steps)
    if _step_race_time_measurements_are_valid(value, name):
        _validate_step_race_time_coverage(count, name, steps)


def _step_race_time_measurement_count(value: Mapping[str, Any], name: str, steps: int) -> float:
    count = _validate_nonnegative_number(
        value["timing/step_race_measurement_count"],
        f"{name} step race-time measurement count",
    )
    if not count.is_integer():
        raise ValueError(f"{name} step race-time measurement count must be an integer")
    if count > steps:
        raise ValueError(f"{name} step race-time measurement count cannot exceed steps")
    return count


def _step_race_time_measurements_are_valid(value: Mapping[str, Any], name: str) -> bool:
    key = "timing/step_race_measurements_valid"
    _validate_binary_flag(value[key], f"{name} {key}")
    return float(value[key]) == 1.0


def _validate_step_race_time_coverage(count: float, name: str, steps: int) -> None:
    if count != steps or steps == 0:
        raise ValueError(f"{name} valid step race-time measurements must cover every step")


def _validate_nonnegative_number(value: Any, name: str) -> float:
    _validate_finite_number(value, name)
    scalar = float(value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _validate_finite_number(value: Any, name: str) -> None:
    numeric = isinstance(value, (int, float, np.number, torch.Tensor))
    if isinstance(value, bool) or not numeric:
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, torch.Tensor) and value.numel() != 1:
        raise TypeError(f"{name} must be scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")


def _validate_binary_flag(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        return
    _validate_finite_number(value, name)
    if float(value) not in {0.0, 1.0}:
        raise ValueError(f"{name} must be boolean or numeric zero/one")
