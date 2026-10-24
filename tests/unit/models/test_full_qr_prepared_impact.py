"""Prepared source routing/normalization, not motor notification qualification."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.prepared_contact_capture import (
    PreparedContactCapture,
)
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact import project_prepared_impact

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/prepared_contacts_v1/context.json"


def contacts() -> list[PreparedContactCapture]:
    records = json.loads(FIXTURE.read_text())["expected"]
    for row in records:
        for field in ("normal", "point", "contact_velocity", "normalization_thresholds"):
            row[field] = tuple(row[field])
    return [PreparedContactCapture(**row) for row in records]


def test_actual_strong_side_and_body_contacts_project_without_invented_flags() -> None:
    actual = contacts()
    projected = [project_prepared_impact(x) for x in actual]
    assert all(p is not None for p in projected)
    for capture, magnitude in zip(actual, projected, strict=True):
        assert magnitude.event_time == capture.event_time > magnitude.native_time
        assert not hasattr(magnitude, "flags")
        assert not hasattr(magnitude, "collision_detected")
    peak = max(projected, key=lambda x: x.raw_strength)
    assert peak.channel == 0
    assert peak.raw_strength == 77.80525970458984
    assert peak.normalized_strength == 1
    body = projected[next(i for i, x in enumerate(actual) if x.wheel_index == -1)]
    assert body.channel == 0
    assert body.normalized_strength == 1
    assert any(x.channel in (1, 2) and x.normalized_strength == 0 for x in projected)


def test_wheel_thresholds_are_not_replaced_by_main_thresholds() -> None:
    source = contacts()[0]
    wheel = replace(
        source,
        normal=(0.0, 1.0, 0.0),
        contact_velocity=(0.0, 20.0, 0.0),
        wheel_index=0,
        wheel_category=1,
    )
    body = replace(wheel, wheel_index=-1, wheel_category=None)
    first = project_prepared_impact(wheel)
    main = project_prepared_impact(body)
    assert first.channel == 1
    assert main.channel == 0
    assert first.normalized_strength > main.normalized_strength
    assert project_prepared_impact(replace(wheel, wheel_category=0)).channel == 2


def test_missing_required_wheel_category_is_not_assumed_zero() -> None:
    source = replace(contacts()[0], normal=(0.0, 1.0, 0.0), wheel_category=None)
    with pytest.raises(ValueError, match="wheel category source unavailable"):
        project_prepared_impact(source)


def test_tiny_contact_speed_is_no_scalar_update_not_interval_coverage() -> None:
    source = replace(contacts()[0], contact_velocity=(0.0, 0.0, 0.0))
    assert project_prepared_impact(source) is None


def test_unavailable_thresholds_fail_instead_of_using_a_default() -> None:
    source = replace(contacts()[0], normalization_thresholds=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="source thresholds"):
        project_prepared_impact(source)
