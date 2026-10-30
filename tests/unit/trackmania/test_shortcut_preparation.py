"""Preparation must expose information loss instead of certifying a replay."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_shortcut_reference import (
    cadence_report,
    control_plan,
    exact_action_ids,
    read_export,
    reference_poses,
)


def test_50ms_hold_reports_a_real_10ms_change_without_qualifying_it() -> None:
    times = np.arange(10, dtype=np.int64) * 10
    controls = np.zeros((10, 3), dtype=np.float32)
    controls[:, 1] = 1
    controls[6:, 0] = 1
    report = cadence_report(times, controls, 50)
    assert report["off_grid_changes"] == 1
    assert report["changed_source_ticks"] == 4
    assert not report["exact_source_controls_preserved"]
    assert not report["native_replay_qualified"]
    assert cadence_report(times, controls, 10)["changed_source_ticks"] == 0
    exact_action_ids(controls)
    controls[6, 0] = 0.2
    with pytest.raises(ValueError, match="outside the exact"):
        exact_action_ids(controls)


def test_provenance_map_and_control_continuity_fail_closed(tmp_path: Path) -> None:
    document = {
        "metadata": {"map_uid_literals": ["test-map"]},
        "input_archives": [
            {
                "packet_period_ms": 10,
                "packets": [
                    {
                        "race_time_ms": t,
                        "steer_normalized": 0,
                        "accel": 1,
                        "brake": 0,
                        "respawn_input": False,
                    }
                    for t in (0, 10, 20)
                ],
            }
        ],
    }
    raw = json.dumps(document).encode()
    archive = tmp_path / "reference.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("reference.json", raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert read_export(archive, "reference.json", digest) == document
    with pytest.raises(ValueError, match="SHA-256"):
        read_export(archive, "reference.json", "0" * 64)
    with pytest.raises(ValueError, match="UID mismatch"):
        control_plan(document, "different-map")
    control_plan(document, "test-map")
    document["input_archives"][0]["packets"][1]["race_time_ms"] = 11
    with pytest.raises(ValueError, match="consecutive"):
        control_plan(document, "test-map")


def test_partial_pose_data_is_not_filled_with_zeroes() -> None:
    sample = {
        "record_time_ms": 0,
        "position_m": [0, 0, 0],
        "linear_velocity_m_s": [0, 0, 0],
        "orientation_xyzw": [0, 0, 0, 1],
    }
    document = {"samples": [sample, {**sample, "record_time_ms": 50}]}
    result = reference_poses(document)
    np.testing.assert_array_equal(result["record_time_ms"], [0, 50])
    document["samples"][1]["linear_velocity_m_s"] = None
    with pytest.raises((ValueError, TypeError)):
        reference_poses(document)
