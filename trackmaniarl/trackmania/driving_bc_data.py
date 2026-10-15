"""Episode-separated recurrent BC data, with no materialized sliding-window copies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.trackmania.driving_vision import DrivingVisionFeaturePipeline


@dataclass(frozen=True)
class DrivingEpisode:
    path: Path
    images: torch.Tensor
    vehicle: torch.Tensor
    actions: torch.Tensor


def load_episodes(
    directory: Path, pipeline: DrivingVisionFeaturePipeline, contract: dict[str, Any]
) -> list[DrivingEpisode]:
    episodes = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["metadata"].item())) != contract:
                raise ValueError(f"{path}: recording contract differs from configured map/capture")
            frames, telemetry = data["frames"], data["telemetry"]
            actions, times = data["actions"], data["timestamps_ms"]
            _validate_episode(data)
            identity = hashlib.sha256(
                frames.tobytes() + actions.astype(np.int64).tobytes()
            ).hexdigest()
            if identity in seen:
                raise ValueError(f"{path}: duplicate episode would contaminate validation")
            seen.add(identity)
            pipeline.reset_episode()
            observations = [
                pipeline.transform_observation({"images": frame, "telemetry": state})
                for frame, state in zip(frames, telemetry, strict=True)
            ]
            images = torch.stack([item["images"] for item in observations])
            vehicle = torch.stack([item["vehicle"] for item in observations])
            episodes.append(
                DrivingEpisode(path, images, vehicle, torch.from_numpy(actions.astype(np.int64)))
            )
            del times
    pipeline.reset_episode()
    if len(episodes) < 2:
        raise ValueError("record at least two distinct complete laps for an episode-level split")
    return episodes


def _validate_episode(data: Any) -> None:
    frames, telemetry = data["frames"], data["telemetry"]
    actions, times = data["actions"], data["timestamps_ms"]
    count = len(actions)
    if frames.dtype != np.uint8 or frames.shape != (count, 90, 160, 3):
        raise ValueError("demo frames must be uint8 RGB (steps, 90, 160, 3)")
    if telemetry.shape != (count, 33) or not np.isfinite(telemetry).all():
        raise ValueError("demo telemetry must contain 33 finite fields per frame")
    if actions.shape != (count,) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("demo action IDs must be a one-dimensional integer array")
    if count < 2 or (actions < 0).any() or (actions >= 78).any():
        raise ValueError("demo has too few frames or invalid action IDs")
    if times.shape != (count,) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("demo timestamps must be finite and strictly increasing")
    if (times < 0).any() or not np.allclose(times, telemetry[:, 3], atol=1):
        raise ValueError("demo telemetry and image timestamps must agree")
    finish = float(data["finish_time_s"])
    if not np.isfinite(finish) or times[-1] >= finish * 1000:
        raise ValueError("demo must finish after its last observation")


def sequence_batch(
    episodes: list[DrivingEpisode], endpoints: list[tuple[int, int]], length: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    images, vehicles, labels, previous = [], [], [], []
    for episode_id, end in endpoints:
        episode = episodes[episode_id]
        indices = torch.arange(end - length + 1, end + 1).clamp_min(0)
        images.append(episode.images[indices])
        vehicles.append(episode.vehicle[indices])
        labels.append(episode.actions[end])
        previous.append(episode.actions[end - 1] if end else torch.tensor(78))
    return {
        "images": torch.stack(images),
        "vehicle": torch.stack(vehicles),
        # Training diagnostics only; neither camera model feeds this label into its encoder.
        "expert_previous_action": torch.stack(previous),
    }, torch.stack(labels)


def sample_endpoints(
    episodes: list[DrivingEpisode], count: int, generator: np.random.Generator
) -> list[tuple[int, int]]:
    return [
        (index, int(generator.integers(len(episodes[index].actions))))
        for index in generator.integers(len(episodes), size=count)
    ]
