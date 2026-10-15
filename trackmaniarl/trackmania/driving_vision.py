"""Compact camera and vehicle-state observations, independent of track geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import functional as F

from trackmaniarl.builtins.features import GymnasiumObservationCollator
from trackmaniarl.core.data import Transition

VEHICLE_FEATURE_DIM = 23


class DrivingVisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(default=160, ge=32)
    height: int = Field(default=90, ge=32)
    grayscale: bool = True
    mask_bottom_fraction: float = Field(default=0.1, ge=0.0, le=0.25)
    velocity_to_mps_scale: float = Field(default=1.0, gt=0)
    control_history: Literal["measured", "issued"] = "measured"


class DrivingVisionFeaturePipeline:
    """Store one uint8 frame and causal vehicle state per replay observation.

    No position, elapsed race time, checkpoint progress, or geometry enters the
    policy. Legacy measured controls and opt-in last issued commands have distinct
    semantics; the latter must be supplied by the environment after execution.
    Models reconstruct history from sequence replay without storing image stacks.
    """

    def __init__(self, config: DrivingVisionConfig | Mapping[str, Any] | None = None) -> None:
        self.config = DrivingVisionConfig.model_validate(config or {})
        channels = 1 if self.config.grayscale else 3
        self.observation_space = spaces.Dict(
            {
                "images": spaces.Box(
                    0, 255, (channels, self.config.height, self.config.width), dtype=np.uint8
                ),
                "vehicle": spaces.Box(-1.0, 1.0, (VEHICLE_FEATURE_DIM,), dtype=np.float32),
            }
        )
        self._collator = GymnasiumObservationCollator(self.observation_space)
        self.reset_episode()

    def reset_episode(self) -> None:
        self._previous_time: float | None = None
        self._previous_forward: np.ndarray | None = None

    def set_evaluation_map(self, map_spec: Any) -> None:
        del map_spec
        self.reset_episode()

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        required = {"telemetry", "images"}
        if self.config.control_history == "issued":
            required.add("previous_command")
        if not isinstance(observation, Mapping) or set(observation) != required:
            raise ValueError(
                f"driving vision {self.config.control_history} mode requires {required}"
            )
        values = np.asarray(observation["telemetry"], dtype=np.float32)
        if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError("driving vision requires 33 finite telemetry fields")
        if self.config.control_history == "issued":
            command = np.asarray(observation["previous_command"], dtype=np.float32)
            if (
                command.shape != (3,)
                or not np.isfinite(command).all()
                or np.any(command < [0, -1, -1])
                or np.any(command > 1)
                or (command[1] < 0 and command[1] != -1)
            ):
                raise ValueError(
                    "previous_command must be [gas, brake, steer]; -1 denotes brake tap"
                )
            values = values.copy()
            values[30:33] = command[[2, 0, 1]]
        return {"images": self._image(observation["images"]), "vehicle": self._vehicle(values)}

    def _image(self, image: Any) -> torch.Tensor:
        frame = torch.as_tensor(np.ascontiguousarray(image))
        if frame.dtype != torch.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("camera image must be HWC RGB uint8")
        frame = frame.permute(2, 0, 1).float()
        # Match the recorder's RGB resize/rounding before grayscale conversion.
        # Converting full desktop frames to grayscale first costs ~40 ms on this host.
        if frame.shape[-2:] != (self.config.height, self.config.width):
            frame = (
                F.interpolate(
                    frame.unsqueeze(0),
                    size=(self.config.height, self.config.width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0]
                .round()
                .clamp_(0, 255)
            )
        if self.config.grayscale:
            frame = (frame * frame.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(
                dim=0, keepdim=True
            )
        resized = frame
        masked_rows = int(self.config.height * self.config.mask_bottom_fraction)
        if masked_rows:
            resized[:, -masked_rows:] = 0  # Mask the configured strip; other HUD may remain.
        return resized.round().clamp_(0, 255).to(torch.uint8)

    def _vehicle(self, values: np.ndarray) -> torch.Tensor:
        forward = _unit(values[10:13], np.array([0.0, 0.0, 1.0]))
        up = _unit(values[13:16], np.array([0.0, 1.0, 0.0]))
        right = _unit(np.cross(up, forward), np.array([1.0, 0.0, 0.0]))
        up = _unit(np.cross(forward, right), up)
        basis = np.stack((right, up, forward))
        dt = 0.0 if self._previous_time is None else (float(values[3]) - self._previous_time) / 1000
        turn = np.zeros(3, dtype=np.float32)
        if 0.001 <= dt <= 1.0 and self._previous_forward is not None:
            turn = basis @ ((forward - self._previous_forward) / dt) / 10.0
        self._previous_time = float(values[3])
        self._previous_forward = forward.copy()
        velocity = basis @ values[7:10] * self.config.velocity_to_mps_scale / 150.0
        gravity = basis @ np.array([0.0, -1.0, 0.0])
        features = np.concatenate(
            (
                velocity,
                gravity,
                turn,
                [
                    values[16] * self.config.velocity_to_mps_scale / 150.0,
                    values[17] / 10000.0,
                    values[18] / 6.0,
                ],
                values[19:23],
                [values[27] / 4.0, values[28] / 3000.0, values[29]],
                values[30:33],
                [max(0.0, dt) / 0.25],
            )
        )
        return torch.from_numpy(np.clip(features, -1.0, 1.0).astype(np.float32))

    def synthetic_observation(self) -> dict[str, np.ndarray]:
        values = np.zeros(33, dtype=np.float32)
        values[12], values[14] = 1.0, 1.0
        observation: dict[str, np.ndarray] = {
            "telemetry": values,
            "images": np.zeros((90, 160, 3), dtype=np.uint8),
        }
        if self.config.control_history == "issued":
            observation["previous_command"] = np.zeros(3, dtype=np.float32)
        return observation

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return dict(self._collator.collate_transitions(transitions))


def _unit(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return np.asarray(value / norm if norm > 1e-6 else fallback, dtype=np.float32)
