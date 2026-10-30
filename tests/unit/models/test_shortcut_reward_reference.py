"""A task route must not change map bindings or manufacture a Finish."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.trackmania.native_headless.full_qr.reward_reference import load_reward_reference
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import TransitionInput


def test_reference_requires_hash_and_exact_map_identity(tmp_path: Path) -> None:
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        schema="tmrl.recorded-reward-reference/v1",
        map_uid="map",
        map_sha256="a" * 64,
        position_m=np.asarray([[0, 0, 0], [1, 0, 0]]),
    )
    binding = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    points = load_reward_reference(binding, map_uid="map", map_sha256="a" * 64)
    assert not points.flags.writeable
    for uid, digest in (("other", "a" * 64), ("map", "b" * 64)):
        with pytest.raises(ValueError, match="map binding"):
            load_reward_reference(binding, map_uid=uid, map_sha256=digest)
    with pytest.raises(ValueError, match="SHA-256"):
        load_reward_reference({**binding, "sha256": "0" * 64}, map_uid="map", map_sha256="a" * 64)


def test_reference_endpoint_without_actual_finish_does_not_finish() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    reward = TrajectoryReward(points, RewardConfig(minimum_finish_steps=1))
    reward.reset(points[0], race_time_ms=0)
    result = reward.step(
        TransitionInput(
            position=points[-1],
            finish_ui_active=False,
            velocity=np.asarray([40, 0, 0]),
            race_time_ms=50,
            collision=False,
            steering=0,
        )
    )
    assert result.reason != "finished"


def test_corrupt_reference_geometry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        schema="tmrl.recorded-reward-reference/v1",
        map_uid="map",
        map_sha256="a" * 64,
        position_m=np.zeros((2, 3)),
    )
    binding = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="duplicate"):
        load_reward_reference(binding, map_uid="map", map_sha256="a" * 64)
