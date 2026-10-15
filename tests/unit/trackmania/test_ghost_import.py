from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.trackmania.demonstration_data import save_demonstration
from trackmaniarl.trackmania.demonstration_processing import validate_recording_quality
from trackmaniarl.trackmania.environment import OpenPlanetEnvironmentFactory
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.ghost_import import (
    GhostAssetRequest,
    build_ghost_assets,
    corridor_boundaries,
    ghost_demonstration,
    ghost_frames,
    load_ghost_lap,
)
from trackmaniarl.trackmania.lidar_feature_setup import LidarFeatureConfig
from trackmaniarl.trackmania.pace import PaceDemonstrationRequest, ReferencePaceProfile
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import TransitionInput

RADIUS_M = 60.0
SPEED_MPS = 20.0
FINISH_MS = 4_730.0


def _left_turn_ghost(path: Path) -> Path:
    """A lap that starts facing +Z and turns a quarter circle to the driver's left (+X)."""

    samples = []
    for time_ms in np.arange(0.0, FINISH_MS, 50.0):
        angle = SPEED_MPS * time_ms / 1_000.0 / RADIUS_M
        samples.append(
            {
                "time_ms": float(time_ms),
                "position": [RADIUS_M * (1 - np.cos(angle)), 12.0, RADIUS_M * np.sin(angle)],
                "velocity": [SPEED_MPS * np.sin(angle), 0.0, SPEED_MPS * np.cos(angle)],
                "rotation": [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)],
                "speed": SPEED_MPS,
                "extras": {"steer": -0.4, "gas": 1, "brake": 0, "wheel_slip": [0, 0, 1, 1]},
            }
        )
    document = {
        "map_uid": "ghost-test-map",
        "samples": samples,
        "checkpoints": [
            {"time_ms": 2_000.0, "checkpoint_index": 0, "lap": 0, "finish": False},
            {"time_ms": FINISH_MS, "checkpoint_index": 1, "lap": 0, "finish": False},
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_frames_follow_the_plugin_layout_and_end_on_the_finish(tmp_path: Path) -> None:
    lap = load_ghost_lap(_left_turn_ghost(tmp_path / "ghost.json"))

    frames = ghost_frames(lap)

    assert frames.shape == (len(lap.times_ms) + 1, 33)
    assert frames[-1, 3] == FINISH_MS
    assert frames[:, 2].tolist() == [0.0] * (len(frames) - 1) + [1.0]
    assert frames[0, 0] == 0.0
    assert frames[-2, 0] == 1.0
    heading = frames[:, 10:13]
    velocity_direction = frames[:, 7:10] / np.linalg.norm(frames[:, 7:10], axis=1, keepdims=True)
    assert np.allclose(heading, velocity_direction, atol=1e-5)
    assert np.allclose(frames[:, 13:16], [0.0, 1.0, 0.0], atol=1e-6)
    assert frames[5, 30:33].tolist() == pytest.approx([-0.4, 1.0, 0.0])
    assert frames[5, 27] == 2.0


def test_corridor_puts_the_inner_wall_on_the_turning_side(tmp_path: Path) -> None:
    lap = load_ghost_lap(_left_turn_ghost(tmp_path / "ghost.json"))
    centre = np.array([RADIUS_M, 12.0, 0.0])

    left, right = corridor_boundaries(lap, half_width_m=8.0)

    left_radius = np.linalg.norm((left - centre)[:, [0, 2]], axis=1)
    right_radius = np.linalg.norm((right - centre)[:, [0, 2]], axis=1)
    assert np.allclose(left_radius[2:-2], RADIUS_M - 8.0, atol=0.2)
    assert np.allclose(right_radius[2:-2], RADIUS_M + 8.0, atol=0.2)


def test_inner_wall_stays_inside_a_turn_tighter_than_the_corridor(tmp_path: Path) -> None:
    lap = load_ghost_lap(_left_turn_ghost(tmp_path / "ghost.json"))

    left, _ = corridor_boundaries(lap, half_width_m=RADIUS_M * 2)

    inner = np.linalg.norm((left - [RADIUS_M, 12.0, 0.0])[:, [0, 2]], axis=1)
    assert np.all(inner[2:-2] >= 0.15 * RADIUS_M)


def test_ghost_assets_load_as_geometry_pace_and_a_finishing_reward(tmp_path: Path) -> None:
    ghost = _left_turn_ghost(tmp_path / "ghost.json")
    map_path = tmp_path / "map.Map.Gbx"
    map_path.write_bytes(b"map")

    assets = build_ghost_assets(GhostAssetRequest(ghost, map_path, tmp_path / "assets"))

    geometry = BoundaryGeometry(assets.geometry_path, expected_map_uid="ghost-test-map")
    pace = ReferencePaceProfile.from_demonstration(
        PaceDemonstrationRequest(assets.pace_path, geometry, geometry.reward_center)
    )
    assert assets.finish_time_s == pytest.approx(FINISH_MS / 1_000.0)
    assert pace.time_at_index(len(geometry.reward_center) - 1) == pytest.approx(4.73)
    frames = ghost_frames(load_ghost_lap(ghost))
    reward = TrajectoryReward(geometry.reward_center, RewardConfig(minimum_finish_steps=1))
    reward.reset(frames[0, 4:7], velocity=frames[0, 7:10], race_time_ms=0.0)
    for frame in frames[1:]:
        result = reward.step(
            TransitionInput(frame[4:7], bool(frame[2]), frame[7:10], float(frame[3]), False, 0.0)
        )
    assert result.reason == "finished"


def _with_input_events(path: Path) -> Path:
    """Add tick-exact input changes that fall between the 50 ms samples."""

    document = json.loads(path.read_text(encoding="utf-8"))
    document["inputs"] = [
        {"time_ms": -900, "steer": 0.0, "throttle": 1, "brake": 0},
        {"time_ms": 1_230, "steer": -0.4, "throttle": 1, "brake": 0},
        {"time_ms": 2_510, "steer": -0.4, "throttle": 1, "brake": 1},
        {"time_ms": 2_540, "steer": -0.4, "throttle": 1, "brake": 0},
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_demonstration_rebuilds_ticks_with_exact_input_changes(tmp_path: Path) -> None:
    lap = load_ghost_lap(_with_input_events(_left_turn_ghost(tmp_path / "ghost.json")))

    demonstration = ghost_demonstration(lap, "0" * 64)

    validate_recording_quality(demonstration)
    times = demonstration.frames[:, 3]
    steer = demonstration.controls[:, 2]
    brake = demonstration.controls[:, 1]
    assert steer[times[:-1] < 1_230].tolist() == pytest.approx(
        [0.0] * int(np.sum(times[:-1] < 1_230))
    )
    assert steer[(times[:-1] >= 1_230)] == pytest.approx(-0.4)
    assert np.flatnonzero(brake).tolist() == [251, 252, 253]
    positions = demonstration.frames[:-1, 4:7]
    radius = np.linalg.norm((positions - [RADIUS_M, 12.0, 0.0])[:, [0, 2]], axis=1)
    assert np.allclose(radius, RADIUS_M, atol=0.01)


def test_line_only_ghost_gives_assets_but_no_demonstration(tmp_path: Path) -> None:
    path = _left_turn_ghost(tmp_path / "ghost.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    for sample in document["samples"]:
        sample["extras"] = {"wheel_slip": [0, 0, 0, 0]}
    path.write_text(json.dumps(document), encoding="utf-8")
    lap = load_ghost_lap(path)

    assert len(corridor_boundaries(lap, 8.0)[0]) > 0
    with pytest.raises(ValueError, match="no inputs"):
        ghost_demonstration(lap, "0" * 64)


def test_ghost_demonstration_loads_as_finishing_transitions(tmp_path: Path) -> None:
    ghost = _with_input_events(_left_turn_ghost(tmp_path / "ghost.json"))
    map_path = tmp_path / "map.Map.Gbx"
    map_path.write_bytes(b"map")
    assets = build_ghost_assets(GhostAssetRequest(ghost, map_path, tmp_path / "assets"))
    geometry = BoundaryGeometry(assets.geometry_path, expected_map_uid="ghost-test-map")
    path = save_demonstration(
        tmp_path / "demo", ghost_demonstration(load_ghost_lap(ghost), geometry.sha256)
    )
    factory = OpenPlanetEnvironmentFactory(
        {
            "geometry_path": str(assets.geometry_path),
            "expected_map_uid": "ghost-test-map",
            "decision_interval_ms": 50.0,
            "action_repeat_frames": 1,
            "demonstration_control_aggregation": True,
            "minimum_finish_steps": 1,
        }
    )
    pipeline = LidarFeaturePipeline(
        LidarFeatureConfig(
            assets.geometry_path,
            expected_map_uid="ghost-test-map",
            include_control_inputs=False,
        )
    )

    transitions = factory.load_demonstration(path, pipeline)

    assert len(transitions) == int(np.ceil(FINISH_MS / 50.0))
    assert transitions[-1].terminated
