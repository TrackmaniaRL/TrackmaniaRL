"""Hash-bound task trajectories, kept separate from physical road boundaries."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trackmaniarl.trackmania.reward_config import build_reward_trajectory


def load_reward_reference(
    binding: Mapping[str, Any], *, map_uid: str, map_sha256: str
) -> np.ndarray:
    """Load recorded positions for progress; this grants no Finish or state restore."""
    path = Path(binding["path"])
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("reward reference requires a canonical absolute path")
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("reward reference exceeds 16 MiB")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("reward reference SHA-256 mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if (
            str(archive["schema"].item()) != "tmrl.recorded-reward-reference/v1"
            or str(archive["map_uid"].item()) != map_uid
            or str(archive["map_sha256"].item()) != map_sha256
        ):
            raise ValueError("reward reference schema/map binding differs")
        positions = np.asarray(archive["position_m"], dtype=np.float32).copy()
    trajectory = build_reward_trajectory(positions)
    trajectory.points.setflags(write=False)
    return trajectory.points
