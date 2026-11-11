from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.unit.trackmania._lidar_fixtures import _asset
from trackmaniarl.core.spec import EvaluationMapSpec
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.geometry import build_geometry_asset
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest
from trackmaniarl.trackmania.multimodal import LidarVisionFeaturePipeline
from trackmaniarl.trackmania.reward_points import save_reward_points


@pytest.mark.parametrize("with_vision", [False, True])
def test_lidar_reward_map_reuses_geometry_only_with_proven_identity(
    tmp_path: Path, *, with_vision: bool
) -> None:
    geometry_path = _asset(tmp_path)
    map_path = tmp_path / "trackmaniarl-test.Map.Gbx"
    config = {"geometry_path": str(geometry_path)}
    if with_vision:
        pipeline = LidarVisionFeaturePipeline(config, {"width": 8, "height": 8, "frame_stack": 2})
        lidar = pipeline.lidar
        pipeline.transform_observation(pipeline.synthetic_observation())
        assert pipeline.vision._frames
    else:
        pipeline = LidarFeaturePipeline(config)
        lidar = pipeline
    geometry = lidar.geometry

    def spec(name: str, uid: str, source: Path | None) -> EvaluationMapSpec:
        points_path = save_reward_points(
            tmp_path / f"{name}.npz",
            np.asarray([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
            map_uid=uid,
            map_path=source,
        )
        return EvaluationMapSpec(
            id=name,
            map_path=map_path,
            reward_points_path=points_path,
            expected_map_uid=uid,
        )

    valid = spec("valid", "trackmaniarl-test", map_path)
    wrong_uid = spec("wrong-uid", "different-map", map_path)
    other_map = tmp_path / "other.Map.Gbx"
    other_map.write_bytes(b"different-map")
    wrong_checksum = spec("wrong-checksum", "trackmaniarl-test", other_map)
    uid_only = spec("uid-only", "trackmaniarl-test", None)
    missing_map = valid.model_copy(update={"map_path": None})

    for invalid, message in [
        (wrong_uid, "map UID"),
        (wrong_checksum, "same map checksum"),
        (uid_only, "same map checksum"),
        (missing_map, "requires a map file"),
    ]:
        with pytest.raises(ValueError, match=message):
            pipeline.set_evaluation_map(invalid)
        assert lidar.geometry is geometry
        if with_vision:
            assert pipeline.vision._frames

    pipeline.set_evaluation_map(valid)
    assert lidar.geometry is geometry
    if with_vision:
        assert not pipeline.vision._frames

    map_path.write_bytes(b"modified-after-asset-creation")
    with pytest.raises(ValueError, match="checksum"):
        pipeline.set_evaluation_map(valid)
    assert lidar.geometry is geometry


def test_reward_points_follow_current_geometry_across_map_switches(tmp_path: Path) -> None:
    geometry_a = _asset(tmp_path)
    map_a = tmp_path / "trackmaniarl-test.Map.Gbx"
    map_b = tmp_path / "other.Map.Gbx"
    map_b.write_bytes(b"another-map")
    geometry_b = build_geometry_asset(
        GeometryBuildRequest(
            tmp_path / "other.npz",
            tmp_path / "left.npy",
            tmp_path / "right.npy",
            "other-map",
            map_b,
        )
    )
    points = np.asarray([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    rewards_a = save_reward_points(
        tmp_path / "a-rewards.npz", points, map_uid="trackmaniarl-test", map_path=map_a
    )
    rewards_b = save_reward_points(
        tmp_path / "b-rewards.npz", points, map_uid="other-map", map_path=map_b
    )
    spec_a = EvaluationMapSpec(
        id="a", map_path=map_a, reward_points_path=rewards_a, expected_map_uid="trackmaniarl-test"
    )
    spec_b = EvaluationMapSpec(
        id="b", map_path=map_b, reward_points_path=rewards_b, expected_map_uid="other-map"
    )
    spec_geometry_b = EvaluationMapSpec(
        id="b-geometry", map_path=map_b, geometry_path=geometry_b, expected_map_uid="other-map"
    )
    pipeline = LidarFeaturePipeline({"geometry_path": str(geometry_a)})
    pipeline.set_evaluation_map(spec_a)
    pipeline.set_evaluation_map(spec_geometry_b)
    with pytest.raises(ValueError, match="map UID"):
        pipeline.set_evaluation_map(spec_a)
    assert pipeline.geometry.path == geometry_b
    pipeline.set_evaluation_map(spec_b)
    assert pipeline.geometry.path == geometry_b
