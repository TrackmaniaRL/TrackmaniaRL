"""A recorded reward path, with map identity, without any road boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.geometry import BoundaryGeometry, file_sha256
from trackmaniarl.trackmania.reward_config import build_reward_trajectory


class RewardPoints:
    def __init__(self, path: str | Path, *, expected_map_uid: str | None = None) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            if str(data["asset_version"].item()) != "reward-points-1":
                raise ValueError("unsupported reward points asset version")
            self.reward_center = build_reward_trajectory(data["points"]).points
            self.map_uid = str(data["map_uid"].item())
            self.map_sha256 = str(data["map_sha256"].item())
            binding = str(data["map_binding"].item()) if "map_binding" in data else "uid+sha256"
        if not self.map_uid or (not self.map_sha256 and binding != "uid"):
            raise ValueError("reward points require map UID and an explicit identity binding")
        if expected_map_uid is not None and expected_map_uid != self.map_uid:
            raise ValueError("reward points map UID does not match the configured map")
        self.sha256 = file_sha256(self.path)

    def validate_map(self, path: str | Path | None) -> None:
        if not self.map_sha256:
            return  # Official campaign maps can be bound to the game's verified UID alone.
        if path is None:
            raise ValueError(
                "these reward points require the original map file for checksum validation"
            )
        if file_sha256(path) != self.map_sha256:
            raise ValueError("reward points checksum does not match the map file")


def save_reward_points(  # noqa: PLR0913 - map identity and sampling are explicit asset metadata
    output: Path,
    points: np.ndarray,
    *,
    map_uid: str,
    map_path: Path | None = None,
    spacing_m: float = 2.0,
) -> Path:
    if not np.isfinite(spacing_m) or spacing_m <= 0 or not map_uid.strip():
        raise ValueError("reward point spacing and map UID must be valid")
    points = build_reward_trajectory(points).points
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    targets = np.linspace(0.0, distances[-1], max(2, int(np.ceil(distances[-1] / spacing_m)) + 1))
    sampled = np.stack([np.interp(targets, distances, points[:, i]) for i in range(3)], axis=1)
    build_reward_trajectory(sampled)
    checksum = file_sha256(map_path) if map_path is not None else ""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            asset_version="reward-points-1",
            points=sampled.astype(np.float32),
            map_uid=map_uid,
            map_sha256=checksum,
            map_binding="uid+sha256" if map_path is not None else "uid",
        )
    return output


def load_reward_asset(
    geometry_path: str | Path | None,
    reward_points_path: str | Path | None,
    expected_map_uid: str | None,
) -> BoundaryGeometry | RewardPoints:
    if (geometry_path is None) == (reward_points_path is None):
        raise ValueError("configure exactly one of geometry_path and reward_points_path")
    if reward_points_path is not None:
        return RewardPoints(reward_points_path, expected_map_uid=expected_map_uid)
    assert geometry_path is not None
    return BoundaryGeometry(geometry_path, expected_map_uid=expected_map_uid)
