"""Tests of diagnostic claims, missing observations, and source preservation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .diagnose_ghost_divergence import (
    SCHEMA,
    Frame,
    Json,
    Series,
    Settings,
    array,
    compare,
    load_series,
    mapping,
)
from .prepare_archived_inputs import write_new

ZERO = (0.0, 0.0, 0.0)


def test_sparse_states_report_an_interval_and_never_interpolate() -> None:
    ghost = Series("map", {
        0: Frame(0, ZERO, ZERO, ZERO),
        50: Frame(50, ZERO, ZERO, ZERO),
    }, {})
    native = Series("map", {
        time: Frame(time, (0.002 if time else 0.0, 0.0, 0.0), ZERO, ZERO)
        for time in range(0, 51, 10)
    }, {})
    report = compare((ghost, native), Settings())
    first = mapping(mapping(report["first_differences"])["position"])
    assert first["first_observed_difference_ms"] == 50
    assert first["possible_first_tick_interval_ms"] == [10, 50]
    assert mapping(report["compared_samples"])["position"] == 2
    assert report["complete_ten_ms_state_grid"] is False
    at_40 = next(mapping(row) for row in array(first["context"]) if mapping(row)["time_ms"] == 40)
    assert at_40["ghost"] is None


def test_strict_threshold_and_neutral_reset_are_distinct_from_driving_inputs() -> None:
    ghost = Series("map", {t: Frame(t, ZERO, ZERO, (0.0, 1.0, 0.0)) for t in (0, 10)}, {})
    native = Series("map", {
        0: Frame(0, (0.001, 0.0, 0.0), ZERO, ZERO, phase="neutral_reset"),
        10: Frame(10, ZERO, ZERO, (0.0, 1.0, 0.0)),
    }, {})
    report = compare((ghost, native), Settings(input_start_ms=10))
    differences = mapping(report["first_differences"])
    assert "position" not in differences
    assert "input" not in differences
    assert mapping(differences["input_including_reset"])["first_observed_difference_ms"] == 0
    assert mapping(report["compared_samples"])["input"] == 1


def test_no_overlap_is_not_a_success_and_clocks_are_not_fitted() -> None:
    ghost = Series("map", {0: Frame(0, ZERO, ZERO, ZERO)}, {})
    native = Series("map", {10: Frame(10, ZERO, ZERO, ZERO)}, {})
    report = compare((ghost, native), Settings())
    assert report["status"] == "INSUFFICIENT_SHARED_OBSERVATIONS"
    assert report["complete_ten_ms_state_grid"] is False
    with pytest.raises(ValueError, match="UIDs differ"):
        compare((ghost, Series("other", ghost.frames, {})), Settings())


@pytest.mark.parametrize("rows", [
    [{"time_ms": 0}, {"time_ms": 0}],
    [{"time_ms": 5}],
    [{"time_ms": 0, "velocity_m_s": [float("nan"), 0, 0]}],
    [{"time_ms": 0, "control_sgb": [0, 0.5, 0]}],
])
def test_malformed_observations_are_rejected(tmp_path: Path, rows: list[Json]) -> None:
    path = tmp_path / "series.json"
    path.write_text(json.dumps({"schema": SCHEMA, "map_uid": "map",
                               "coordinate_frame": "world_xyz_y_up", "frames": rows}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match=r"required|domain"):
        load_series(path)
    assert path.read_bytes() == before


def test_writing_a_report_cannot_overwrite_evidence(tmp_path: Path) -> None:
    path = tmp_path / "existing.json"
    before = b'{"evidence":true}'
    path.write_bytes(before)
    with pytest.raises(FileExistsError):
        write_new(path, {"replacement": True})
    assert path.read_bytes() == before
