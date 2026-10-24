"""Diagnostics from an already qualified native result; no additional game I/O."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from trackmaniarl.distributed.metric_diagnostics import (
    METRIC_DIAGNOSTICS_SCHEMA,
    validate_metric_diagnostics,
)

_UNMEASURED = (
    "progress_pct",
    "progress_m",
    "duration_s",
    "race_time_s",
    "finish_time_s",
    "reward/time",
    "reward/pace",
    "reward/pbrs",
    "reward/progress",
    "reward/projected_velocity",
    "reward/projected_speed",
    "reward/steering_delta",
    "reward/collision",
    "reward/terminal",
    "reward/time_attack_terminal",
    "collision/count",
    "collision/detected_count",
    "velocity/ratio_mean",
    "velocity/ratio_max",
    "control/gas_fraction",
    "control/brake_fraction",
    "control/brake_tap_fraction",
    "control/steer_abs_mean",
    "controller_apply_ms_mean",
    "controller_apply_ms_max",
    "telemetry_wait_ms_mean",
    "telemetry_wait_ms_max",
    "telemetry_skipped_frames_total",
)


def native_metric_diagnostics(
    result: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Unknown SI/race/collision semantics remain unavailable, including numeric0."""
    state = result["observation"]
    projection = state["projection"]
    availability = dict.fromkeys(_UNMEASURED, False)
    source = record["profile"]
    reference_source = "authored-reference:" + record["referenceFingerprint"]

    def measured(value: Any, unit: str, origin: str) -> dict[str, Any]:
        return {"value": value, "unit": unit, "source": origin}

    measurements = {
        "native_clock": measured(state["nativeT"], "native_time_unit", "qualified-native-clock"),
        "native_delta": measured(
            result["deltaNative"], "native_time_unit", "qualified-native-clock"
        ),
    }
    # Diagnostic absence must not add a new learning-data requirement. Reduced
    # archived fixtures and compatible facades can omit optional projection data.
    for name, key in (("reference_progress", "progress"), ("reference_deviation", "deviation")):
        value = projection.get(key)
        available = type(value) in (int, float) and math.isfinite(value)
        availability[name] = available
        if available:
            measurements[name] = measured(value, "native_length_unit", reference_source)
    raw_hex = state.get("native76Hex")
    raw = b""
    if isinstance(raw_hex, str):
        with suppress(ValueError):
            raw = bytes.fromhex(raw_hex)
    velocity = struct.unpack_from("<3f", raw, 28) if len(raw) == 76 else ()
    availability["physical_speed"] = len(velocity) == 3 and all(math.isfinite(v) for v in velocity)
    if availability["physical_speed"]:
        measurements["physical_speed"] = measured(
            math.sqrt(sum(v * v for v in velocity)), "native_velocity_unit", "qualified-native76"
        )
    category = "time_limit" if record["truncated"] else "ongoing"
    availability["termination/time_limit"] = record["truncated"] or record["terminated"]
    if record["terminated"]:
        category = "finished"
        availability["race_time_s"] = availability["finish_time_s"] = True
        measurements["computed_finish_ms"] = measured(
            result["outcome"]["computedResult"]["Time"],
            "qualified_finish_ms",
            "full-original-validator-TMRESULT",
        )
    return validate_metric_diagnostics(
        {
            "schema": METRIC_DIAGNOSTICS_SCHEMA,
            "scope": "step",
            "termination_category": category,
            "source": source,
            "availability": availability,
            "measurements": measurements,
        }
    )
