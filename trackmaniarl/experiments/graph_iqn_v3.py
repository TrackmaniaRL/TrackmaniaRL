"""Residual context features for the V2 boundary-graph IQN policy.

V3 keeps every trained V2 encoder tensor compatible and adds a zero-initialized
context branch.  Loading a V2 policy therefore reproduces its output exactly at
initialization, while training can learn corrections from road clearance,
vehicle contact state, track position and speed-adaptive curvature previews.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, pi
from typing import Any, cast

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.experiments.graph_iqn_v2 import (
    TRACK_SCALE_M,
    BoundaryGraphFeaturePipelineV2,
    TrackGnnSimbaEncoderV2,
)

PREVIEW_HORIZONS_S = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0)
PROGRESS_FREQUENCIES = (1.0, 2.0, 4.0)

CONTEXT_V3_LAYOUT: tuple[str, ...] = (
    "signed_lateral_offset",
    "left_clearance",
    "right_clearance",
    "half_width",
    "minimum_clearance",
    "sideslip",
    "rear_left_slip",
    "rear_right_slip",
    "rpm",
    "adherence",
    "flying_duration",
    *(f"progress_sin_{frequency:g}" for frequency in PROGRESS_FREQUENCIES),
    *(f"progress_cos_{frequency:g}" for frequency in PROGRESS_FREQUENCIES),
    *(f"preview_curvature_{horizon:g}s" for horizon in PREVIEW_HORIZONS_S),
    *(f"preview_lateral_demand_{horizon:g}s" for horizon in PREVIEW_HORIZONS_S),
)
CONTEXT_V3_DIM = len(CONTEXT_V3_LAYOUT)


class BoundaryGraphFeaturePipelineV3(BoundaryGraphFeaturePipelineV2):
    """V2 observation plus compact road, grip and preview context."""

    schema_version = "3"

    @staticmethod
    def _build_observation_space() -> spaces.Dict:
        base = BoundaryGraphFeaturePipelineV2._build_observation_space()
        return spaces.Dict(
            {
                "physics": base["physics"],
                "track": base["track"],
                "context": spaces.Box(-4.0, 4.0, (CONTEXT_V3_DIM,), dtype=np.float32),
            }
        )

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if isinstance(observation, Mapping):
            return self._validate_prepared(observation)
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        prepared = super().transform_observation(values)
        context = self._context_v3(values, self._progress_index)
        return {
            "context": torch.from_numpy(context),
            "physics": prepared["physics"],
            "track": prepared["track"],
        }

    def _validate_prepared(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if set(observation) != {"physics", "track", "context"}:
            raise ValueError("prepared V3 observation requires physics, track and context")
        result = {
            "context": torch.as_tensor(observation["context"], dtype=torch.float32),
            "physics": torch.as_tensor(observation["physics"], dtype=torch.float32),
            "track": torch.as_tensor(observation["track"], dtype=torch.float32),
        }
        if (
            result["physics"].shape != (60,)
            or result["track"].shape != (3, 88)
            or result["context"].shape != (CONTEXT_V3_DIM,)
        ):
            raise ValueError("prepared V3 observation has invalid shape")
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError("prepared V3 observation contains non-finite values")
        return result

    def _context_v3(self, values: np.ndarray, nearest: int) -> np.ndarray:
        road = self._road_context(values[4:7], nearest)
        vehicle = self._vehicle_context(values)
        progress = nearest / max(len(self._reward_center) - 1, 1)
        phase = 2.0 * pi * progress * np.asarray(PROGRESS_FREQUENCIES, dtype=np.float32)
        previews = self._preview_context(float(values[16]), nearest)
        context = np.concatenate((road, vehicle, np.sin(phase), np.cos(phase), previews))
        return np.clip(np.asarray(context, dtype=np.float32), -4.0, 4.0)

    def _road_context(self, position: np.ndarray, nearest: int) -> np.ndarray:
        left = self._left[nearest][[0, 2]]
        right = self._right[nearest][[0, 2]]
        corridor = right - left
        width = max(float(np.linalg.norm(corridor)), 1.0e-3)
        fraction = float((position[[0, 2]] - left) @ corridor) / (width * width)
        left_clearance = 2.0 * fraction
        right_clearance = 2.0 * (1.0 - fraction)
        return np.asarray(
            (
                2.0 * fraction - 1.0,
                left_clearance,
                right_clearance,
                width / 20.0,
                min(left_clearance, right_clearance),
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _vehicle_context(values: np.ndarray) -> np.ndarray:
        heading = values[[10, 12]]
        heading /= max(float(np.linalg.norm(heading)), 1.0e-6)
        velocity = values[[7, 9]]
        forward_velocity = float(velocity @ heading)
        lateral_velocity = float(velocity[0] * heading[1] - velocity[1] * heading[0])
        sideslip = np.arctan2(lateral_velocity, max(abs(forward_velocity), 1.0)) / (0.5 * pi)
        return np.asarray(
            (
                sideslip,
                np.clip(values[21], 0.0, 1.0),
                np.clip(values[22], 0.0, 1.0),
                np.clip(values[17] / 10_000.0, 0.0, 2.0),
                np.clip(values[29], 0.0, 1.0),
                np.clip(values[28] / 1_000.0, 0.0, 4.0),
            ),
            dtype=np.float32,
        )

    def _preview_context(self, speed_mps: float, nearest: int) -> np.ndarray:
        preview_speed = max(speed_mps, 10.0)
        targets = self._reward_distance[nearest] + preview_speed * np.asarray(
            PREVIEW_HORIZONS_S, dtype=np.float32
        )
        indices = np.searchsorted(self._distance, targets).clip(0, len(self._distance) - 1)
        scaled_curvature = self._curvature(np.asarray(indices, dtype=np.int64))
        curvature_per_m = scaled_curvature / 10.0
        lateral_demand = np.square(speed_mps) * np.abs(curvature_per_m) / 50.0
        return np.concatenate((scaled_curvature, lateral_demand)).astype(np.float32)


class TrackGnnSimbaEncoderV3(TrackGnnSimbaEncoderV2):
    """V2 encoder with an initially inert residual context correction."""

    fingerprint_parameters: tuple[str, ...] = ("context_residual_scale",)

    def __init__(self, context_residual_scale: float = 1.0) -> None:
        super().__init__()
        if not isfinite(context_residual_scale) or not 0.0 < context_residual_scale <= 1.0:
            raise ValueError("context_residual_scale must be finite and in (0, 1]")
        self.context_residual_scale = float(context_residual_scale)
        self.context_adapter = nn.Sequential(
            nn.Linear(CONTEXT_V3_DIM, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 192),
        )
        final = cast(nn.Linear, self.context_adapter[-1])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        track = observation["track"].float()
        physics = observation["physics"].float()
        context = observation["context"].float()
        if track.ndim != 3 or track.shape[1:] != (3, 88):
            raise ValueError("track observation must have shape (batch, 3, 88)")
        if physics.shape != (track.shape[0], 60):
            raise ValueError("physics observation must have shape (batch, 60)")
        if context.shape != (track.shape[0], CONTEXT_V3_DIM):
            raise ValueError(f"context observation must have shape (batch, {CONTEXT_V3_DIM})")
        physics_latent = self.physics_proj(physics) + self._context_correction(context)
        joint = torch.cat((self.track_conv(track / TRACK_SCALE_M), physics_latent), dim=-1)
        return cast(torch.Tensor, self.backbone(joint))

    def _context_correction(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_residual_scale * torch.tanh(self.context_adapter(context))


class AdapterOnlyDiscreteValueLearner(DiscreteValueLearner):
    """Warm-start learner that updates only the V3 context adapter.

    The inherited V2 encoder, temporal core and IQN head remain fixed.  This is
    the first-stage safety gate for a new residual feature branch: no gradient
    can degrade the established source policy except through the adapter.
    """

    def _prepare_model(self) -> None:
        super()._prepare_model()
        assert isinstance(self.model.encoder, TrackGnnSimbaEncoderV3)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.encoder.context_adapter.parameters():
            parameter.requires_grad_(True)
