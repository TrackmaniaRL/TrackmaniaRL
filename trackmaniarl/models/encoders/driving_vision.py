"""Small spatial residual CNN fused with track-independent vehicle telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from trackmaniarl.trackmania.driving_vision import VEHICLE_FEATURE_DIM


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, value + self.layers(value))


class DrivingVisionEncoder(nn.Module):
    """One frame per recurrent step; normalize uint8 only on the model device."""

    def __init__(self, channels: int = 1, output_dim: int = 256) -> None:
        super().__init__()
        if channels not in (1, 3) or output_dim < 1:
            raise ValueError("channels must be 1 or 3 and output_dim must be positive")
        self.channels, self.output_dim = channels, output_dim
        stages: list[nn.Module] = []
        current = channels
        for width in (32, 64, 64):
            stages.extend(
                (nn.Conv2d(current, width, 3, stride=2, padding=1), _ResidualBlock(width))
            )
            current = width
        self.convolution = nn.Sequential(*stages, nn.SiLU(), nn.AvgPool2d(2))
        self.image_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 10, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.vehicle = nn.Sequential(
            nn.Linear(VEHICLE_FEATURE_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(320, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        images, vehicle = observation["images"], observation["vehicle"]
        if images.ndim != 4 or images.shape[1] != self.channels:
            raise ValueError("images must have shape (batch, configured channels, height, width)")
        if images.shape[-2:] != (90, 160):
            raise ValueError("DrivingVisionEncoder requires 160x90 frames")
        if images.dtype != torch.uint8:
            raise ValueError("DrivingVisionEncoder expects uint8 images in [0, 255]")
        if vehicle.shape != (images.shape[0], VEHICLE_FEATURE_DIM):
            raise ValueError("vehicle state must match image batch and vehicle feature count")
        visual = self.image_projection(self.convolution(images.float() / 255.0))
        state = self.vehicle(vehicle.float())
        return cast(torch.Tensor, self.fusion(torch.cat((visual, state), dim=-1)))
