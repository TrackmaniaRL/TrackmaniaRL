"""Import recorded camera/control pairs without inventing a terminal camera frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trackmaniarl.core.contracts import FeaturePipeline
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.driving_bc_data import _validate_episode
from trackmaniarl.trackmania.environment_config import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import file_sha256
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_points import RewardPoints
from trackmaniarl.trackmania.reward_types import TransitionInput


@dataclass(frozen=True)
class DrivingReplayContext:
    config: TrackmaniaEnvironmentConfig
    capture: dict[str, Any]
    pipeline: FeaturePipeline


def driving_demonstration_transitions(
    path: str | Path, context: DrivingReplayContext
) -> list[Transition]:
    config = context.config
    pipeline_config = getattr(context.pipeline, "config", None)
    if getattr(pipeline_config, "control_history", "measured") == "issued":
        raise ValueError(
            "v1 human demonstrations contain measured controls, not issued commands; "
            "use measured control history in a separate demonstration experiment"
        )
    if config.reward_points_path is None or config.compact_action_ids is not None:
        raise ValueError("camera demonstrations require reward points and the full action space")
    points = RewardPoints(config.reward_points_path, expected_map_uid=config.expected_map_uid)
    expected = {
        "format": "trackmaniarl-driving-demo-v1",
        "map_uid": points.map_uid,
        "reward_sha256": points.sha256,
        "interval_ms": config.decision_interval_ms,
        "capture": context.capture,
        "image_size": [160, 90],
        "alignment": "image_before_control_window",
    }
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["metadata"].item())) != expected:
            raise ValueError("camera demonstration map/capture contract differs from configuration")
        _validate_episode(data)
        frames, telemetry, actions = data["frames"], data["telemetry"], data["actions"]
        finish_time = float(data["finish_time_s"])
    pipeline = context.pipeline
    reset = getattr(pipeline, "reset_episode", None)
    if callable(reset):
        reset()
    reward = TrajectoryReward(points.reward_center, config.reward_config(None))
    reward.reset(
        telemetry[0, list(config.position_indices)],
        velocity=telemetry[0, list(config.velocity_indices)],
        race_time_ms=float(telemetry[0, 3]),
    )
    observations = [
        pipeline.transform_observation({"images": image, "telemetry": state})
        for image, state in zip(frames, telemetry, strict=True)
    ]
    _, table = select_brake_tap_actions(None)
    steering = np.asarray(table)[actions, 2]
    switches = np.flatnonzero(np.r_[False, steering[1:] != steering[:-1]])
    episode = f"camera-demo-{file_sha256(path)[:16]}"
    transitions = []
    for index, state in enumerate(telemetry[1:]):
        action = int(actions[index])
        scored = reward.step(
            TransitionInput(
                state[list(config.position_indices)],
                bool(state[2]),
                state[list(config.velocity_indices)],
                float(state[3]),
                False,
                float(table[action][2]),
            )
        )
        if scored.terminated:
            raise ValueError(f"camera demonstration reward terminated early: {scored.reason}")
        # v1 recordings omit the finish image: retain only measured pairs and bootstrap
        # at their truncated end, without fabricating a finish bonus or observation.
        transitions.append(
            Transition(
                observations[index],
                action,
                scored.reward,
                observations[index + 1],
                terminated=False,
                truncated=index == len(telemetry) - 2,
                info={
                    "source": "demo",
                    "is_demo": True,
                    "sampling/projected_lap_time_s": finish_time,
                    "demonstration_progress_fraction": reward.progress_pct / 100,
                    "demonstration_steering_switch": bool(index in switches),
                    "demonstration_steering_switch_distance": (
                        int(np.abs(switches - index).min()) if len(switches) else 1_000_000
                    ),
                },
                episode_id=episode,
                step=index,
            )
        )
    if callable(reset):
        reset()
    return transitions
