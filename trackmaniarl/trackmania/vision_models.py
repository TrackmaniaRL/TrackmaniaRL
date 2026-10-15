"""Reusable image encoder and first-party Trackmania actor-critic models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.models.actors import (
    GaussianActorConfig,
    PpoGaussianActor,
)
from trackmaniarl.models.critics import (
    ContinuousValueCritic,
)
from trackmaniarl.models.encoders.convolutional import ConvolutionalSensorEncoder
from trackmaniarl.models.sensor_actor_critic import SensorActorCriticModelFactory


class VisionSensorEncoder(ConvolutionalSensorEncoder):
    """Encode normalized CHW stacks, preserving any leading batch/time axes."""

    def __init__(self, channels: int = 4, output_dim: int = 256, hidden_dim: int = 64) -> None:
        super().__init__(channels, output_dim, hidden_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim < 4 or frames.shape[-3] != self.channels:
            raise ValueError("vision input must have shape (..., channels, height, width)")
        leading = frames.shape[:-3]
        features = super().forward(frames.reshape(-1, *frames.shape[-3:]))
        return features.reshape(*leading, self.output_dim)


class VisionPpoModel(nn.Module):
    def __init__(self, channels: int = 4, hidden_dim: int = 256) -> None:
        super().__init__()
        self.actor = PpoGaussianActor(
            VisionSensorEncoder(channels, hidden_dim),
            GaussianActorConfig(
                hidden_dim, 3, action_low=(0.0, 0.0, -1.0), action_high=(1.0, 1.0, 1.0)
            ),
        )
        self.value = ContinuousValueCritic(VisionSensorEncoder(channels, hidden_dim), hidden_dim)


class VisionPpoModelFactory:
    model_contract = ModelContract.CONTINUOUS_ACTOR_VALUE

    def __init__(self, channels: int = 4, hidden_dim: int = 256) -> None:
        if channels < 1 or hidden_dim < 1:
            raise ValueError("vision model dimensions must be positive")
        self.channels, self.hidden_dim = channels, hidden_dim

    def build(self) -> VisionPpoModel:
        return VisionPpoModel(self.channels, self.hidden_dim)


class VisionActorCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    channels: int = Field(default=4, ge=1)
    hidden_dim: int = Field(default=256, ge=1)
    critic_count: int = Field(default=10, ge=2)
    quantile_count: int = Field(default=25, ge=2)
    action_count: int = Field(default=78, ge=2)


class VisionActorCriticModelFactory:
    """Build independent CNN actor/critic branches for off-policy algorithms."""

    def __init__(
        self,
        algorithm: str,
        config: VisionActorCriticConfig | Mapping[str, Any] | None = None,
    ) -> None:
        algorithm = {"discrete-sac": "sd-sac", "dsac": "sd-sac"}.get(algorithm, algorithm)
        contracts = {
            "sac": ModelContract.CONTINUOUS_ACTOR_CRITIC,
            "redq": ModelContract.ENSEMBLE_ACTOR_CRITIC,
            "tqc": ModelContract.CONTINUOUS_QUANTILE_ACTOR_CRITIC,
            "sd-sac": ModelContract.DISCRETE_ACTOR_CRITIC,
        }
        if algorithm not in contracts:
            raise ValueError(f"unknown vision actor-critic algorithm: {algorithm}")
        shape = VisionActorCriticConfig.model_validate({} if config is None else config)
        self.model_contract = contracts[algorithm]
        self.algorithm = algorithm
        self.channels, self.hidden_dim = shape.channels, shape.hidden_dim
        self.critic_count, self.quantile_count = shape.critic_count, shape.quantile_count
        self.action_count = shape.action_count

    def build(self) -> nn.Module:
        return SensorActorCriticModelFactory(
            self.algorithm,
            {
                "class_path": "trackmaniarl.trackmania.vision_models:VisionSensorEncoder",
                "kwargs": {"channels": self.channels, "output_dim": self.hidden_dim},
            },
            {
                "feature_dim": self.hidden_dim,
                "action_count": self.action_count,
                "critic_count": self.critic_count,
                "quantile_count": self.quantile_count,
            },
        ).build()
