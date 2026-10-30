from copy import deepcopy
from typing import Any

import pytest

from trackmaniarl.distributed.metric_diagnostics import (
    METRIC_DIAGNOSTICS_KEY,
    METRIC_DIAGNOSTICS_SCHEMA,
    MetricDiagnosticsTracker,
    metric_is_available,
    validate_metric_diagnostics,
)


def diagnostic() -> dict[str, Any]:
    return {
        "schema": METRIC_DIAGNOSTICS_SCHEMA,
        "scope": "step",
        "source": "qualified-native-clock",
        "termination_category": "ongoing",
        "availability": {"physical_speed": True, "race_time_s": False},
        "measurements": {
            "physical_speed": {"value": 3.0, "unit": "native_velocity_unit", "source": "native76"}
        },
    }


def test_validated_diagnostics_own_nested_input_and_keep_explicit_unknowns() -> None:
    original = diagnostic()
    expected = deepcopy(original)
    snapshot = validate_metric_diagnostics(original)
    original["availability"]["physical_speed"] = False
    original["measurements"]["physical_speed"]["value"] = 999.0
    assert snapshot == expected
    assert metric_is_available({METRIC_DIAGNOSTICS_KEY: snapshot}, "physical_speed")
    assert not metric_is_available({METRIC_DIAGNOSTICS_KEY: snapshot}, "race_time_s")
    assert metric_is_available({}, "race_time_s")


def test_missing_final_diagnostics_do_not_carry_a_previous_measurement() -> None:
    tracker = MetricDiagnosticsTracker()
    tracker.record({METRIC_DIAGNOSTICS_KEY: diagnostic()})
    tracker.record({})
    result = tracker.summary()
    assert result is not None
    assert result["measurements"] == {}
    assert result["availability"] == {"physical_speed": False, "race_time_s": False}
    assert result["termination_category"] == "unknown"
    assert result["coverage"]["steps"] == 2
    assert result["coverage"]["measurements"] == {"physical_speed": 1}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_and_boolean_measurements_cannot_qualify(value: Any) -> None:
    record = diagnostic()
    record["measurements"]["physical_speed"]["value"] = value
    with pytest.raises(ValueError, match="finite measured scalar"):
        validate_metric_diagnostics(record)


def test_episode_measurement_unit_cannot_change_without_a_new_source_contract() -> None:
    tracker = MetricDiagnosticsTracker()
    tracker.record({METRIC_DIAGNOSTICS_KEY: diagnostic()})
    changed = diagnostic()
    changed["measurements"]["physical_speed"]["unit"] = "metres_per_second"
    with pytest.raises(ValueError, match="source/unit changed"):
        tracker.record({METRIC_DIAGNOSTICS_KEY: changed})
