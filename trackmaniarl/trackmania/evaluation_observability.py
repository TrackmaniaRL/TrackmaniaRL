"""Per-episode control-loop measurements for TrackMania evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from json import dumps
from math import ceil, isfinite
from numbers import Real
from typing import Any

from trackmaniarl.experiments.evaluation import EvaluationResult


@dataclass(slots=True)
class EvaluationObservability:
    action_latency_ms: float = 0.0
    controller_apply_ms: float = 0.0
    telemetry_wait_ms: float = 0.0
    control_brake_taps: int = 0
    issued_control_counts: dict[str, int] = field(default_factory=dict)
    issued_control_measurement_count: int = 0
    issued_gas_sum: float = 0.0
    step_race_times_ms: list[float] = field(default_factory=list)
    step_race_time_invalid_measurements: int = 0
    telemetry_skipped_frames_total: int = 0
    telemetry_skipped_frames_max: int = 0
    telemetry_steps_with_skipped_frames: int = 0
    action_latencies_ms: list[float] = field(default_factory=list)
    vision_capture_times_ms: list[float] = field(default_factory=list)
    vision_pairing_delays_ms: list[float] = field(default_factory=list)

    def record(self, info: Mapping[str, Any], action_duration_ms: float) -> None:
        self.action_latency_ms += action_duration_ms
        self.action_latencies_ms.append(action_duration_ms)
        self._record_vision_timing(info)
        self.controller_apply_ms += float(info.get("controller_apply_ms", 0.0))
        self.telemetry_wait_ms += float(info.get("telemetry_wait_ms", 0.0))
        self.control_brake_taps += int(bool(info.get("control_brake_tap", False)))
        self._record_issued_control(info)
        self._record_step_race_time(info)
        self._record_skipped_frames(info)

    def _record_issued_control(self, info: Mapping[str, Any]) -> None:
        # Command issued by the backend, not delayed input telemetry. This does
        # not imply that the game acknowledged or physically executed it.
        control = info.get("control/issued_command")
        if not isinstance(control, (list, tuple)) or len(control) != 3:
            return
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in control):
            return
        gas, brake, steer = (float(v) for v in control)
        if not (
            isfinite(gas)
            and 0 <= gas <= 1
            and isfinite(brake)
            and -1 <= brake <= 1
            and isfinite(steer)
            and -1 <= steer <= 1
        ):
            return
        key = dumps([gas, brake, steer], separators=(",", ":"))
        self.issued_control_counts[key] = self.issued_control_counts.get(key, 0) + 1
        self.issued_control_measurement_count += 1
        self.issued_gas_sum += gas

    def _record_vision_timing(self, info: Mapping[str, Any]) -> None:
        for key, destination in (
            ("vision/capture_ms", self.vision_capture_times_ms),
            ("vision/pairing_delay_ms", self.vision_pairing_delays_ms),
        ):
            value = info.get(key)
            if isinstance(value, Real) and not isinstance(value, bool):
                measurement = float(value)
                if isfinite(measurement) and measurement >= 0:
                    destination.append(measurement)

    def _record_step_race_time(self, info: Mapping[str, Any]) -> None:
        if _same_clock_finish(info):
            self.step_race_times_ms.append(0.0)
            return
        value = _positive_finite_measurement(info.get("step_race_time_ms"))
        if value is None:
            self.step_race_time_invalid_measurements += 1
            return
        self.step_race_times_ms.append(value)

    def _record_skipped_frames(self, info: Mapping[str, Any]) -> None:
        skipped = int(info.get("telemetry_skipped_frames", 0))
        self.telemetry_skipped_frames_total += skipped
        self.telemetry_skipped_frames_max = max(self.telemetry_skipped_frames_max, skipped)
        self.telemetry_steps_with_skipped_frames += int(skipped > 0)

    def mean_action_latency_ms(self, step_count: int) -> float:
        return self.action_latency_ms / max(step_count, 1)

    def annotated_result(self, result: EvaluationResult, step_count: int) -> EvaluationResult:
        steps = max(step_count, 1)
        measured = self._control_result(result, steps)
        return self._vision_result(self._telemetry_result(measured, steps))

    def _vision_result(self, result: EvaluationResult) -> EvaluationResult:
        capture, pairing = self.vision_capture_times_ms, self.vision_pairing_delays_ms
        return replace(
            result,
            action_latency_ms_p95=_percentile(self.action_latencies_ms, 0.95),
            vision_capture_ms=sum(capture) / len(capture) if capture else 0.0,
            vision_capture_ms_p95=_percentile(capture, 0.95),
            vision_capture_measurement_count=len(capture),
            vision_pairing_delay_ms=sum(pairing) / len(pairing) if pairing else 0.0,
            vision_pairing_delay_ms_p95=_percentile(pairing, 0.95),
            vision_pairing_measurement_count=len(pairing),
        )

    def _control_result(self, result: EvaluationResult, steps: int) -> EvaluationResult:
        race_p99, race_max = _race_time_tail(self.step_race_times_ms)
        return replace(
            result,
            controller_apply_ms=self.controller_apply_ms / steps,
            telemetry_wait_ms=self.telemetry_wait_ms / steps,
            control_brake_tap_fraction=self.control_brake_taps / steps,
            issued_control_counts=dict(self.issued_control_counts),
            issued_control_measurement_count=self.issued_control_measurement_count,
            issued_control_measurements_valid=self.issued_control_measurement_count == steps,
            issued_gas_mean=(
                self.issued_gas_sum / self.issued_control_measurement_count
                if self.issued_control_measurement_count
                else None
            ),
            step_race_time_ms_p99=race_p99,
            step_race_time_ms_max=race_max,
            step_race_time_measurement_count=len(self.step_race_times_ms),
            step_race_time_measurements_valid=self._step_race_time_measurements_valid(steps),
        )

    def _step_race_time_measurements_valid(self, steps: int) -> bool:
        return (
            steps > 0
            and len(self.step_race_times_ms) == steps
            and self.step_race_time_invalid_measurements == 0
        )

    def _telemetry_result(self, result: EvaluationResult, steps: int) -> EvaluationResult:
        return replace(
            result,
            telemetry_skipped_frames_total=self.telemetry_skipped_frames_total,
            telemetry_skipped_frames_mean=self.telemetry_skipped_frames_total / steps,
            telemetry_skipped_frames_max=self.telemetry_skipped_frames_max,
            telemetry_steps_with_skipped_frames_fraction=(
                self.telemetry_steps_with_skipped_frames / steps
            ),
        )


def _race_time_tail(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    p99_index = ceil(0.99 * len(ordered)) - 1
    return ordered[p99_index], ordered[-1]


def _percentile(values: list[float], quantile: float) -> float:
    """Use the nearest-rank empirical percentile within one episode."""
    return sorted(values)[ceil(quantile * len(values)) - 1] if values else 0.0


def _positive_finite_measurement(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    measurement = float(value)
    return measurement if isfinite(measurement) and measurement > 0.0 else None


def _same_clock_finish(info: Mapping[str, Any]) -> bool:
    value = info.get("step_race_time_ms")
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and value == 0.0
        and info.get("termination_reason") == "finished"
        and info.get("telemetry_health") == "ok"
    )
