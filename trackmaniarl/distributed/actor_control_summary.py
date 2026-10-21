"""Default control-loop observability output for distributed actors."""

from __future__ import annotations

_EMPTY_CONTROL_SUMMARY: dict[str, float] = {
    "control_gas_fraction": 0.0,
    "control_brake_fraction": 0.0,
    "control_brake_tap_fraction": 0.0,
    "control_steer_abs_mean": 0.0,
    "step_race_time_ms_mean": 0.0,
    "step_race_time_ms_p99": 0.0,
    "step_race_time_ms_max": 0.0,
    "step_race_time_measurement_count": 0.0,
    "step_race_time_measurements_valid": 0.0,
    "step_race_time_invalid_measurements": 0.0,
    "step_race_time_over_100ms_fraction": 0.0,
    "controller_apply_ms_mean": 0.0,
    "controller_apply_ms_max": 0.0,
    "telemetry_wait_ms_mean": 0.0,
    "telemetry_wait_ms_max": 0.0,
    "telemetry_skipped_frames_total": 0.0,
    "telemetry_skipped_frames_mean": 0.0,
    "telemetry_skipped_frames_max": 0.0,
    "telemetry_steps_with_skipped_frames_fraction": 0.0,
}
