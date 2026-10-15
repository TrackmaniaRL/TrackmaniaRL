"""Small rectangular-image encoder with temporally consistent random shifts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from trackmaniarl.trackmania.driving_vision import VEHICLE_FEATURE_DIM


def random_shift(images: torch.Tensor, padding: int) -> torch.Tensor:
    """One integer translation per sample, shared by every color/history channel."""
    if not padding:
        return images
    batch, _, height, width = images.shape
    padded = F.pad(images, (padding,) * 4, mode="replicate")
    offsets = torch.randint(2 * padding + 1, (batch, 2), device=images.device)
    rows = torch.arange(height, device=images.device)[None, :, None] + offsets[:, :1, None]
    columns = torch.arange(width, device=images.device)[None, None, :] + offsets[:, None, 1:]
    samples = torch.arange(batch, device=images.device)[:, None, None]
    return padded.permute(0, 2, 3, 1)[samples, rows, columns].permute(0, 3, 1, 2)


class DrivingStackEncoder(nn.Module):
    """Four stride-two convolutions; an experimental baseline, not exact DrQ-v2."""

    def __init__(self, channels: int = 3, frame_stack: int = 4, *, shift_padding: int = 4) -> None:
        super().__init__()
        if channels not in (1, 3) or frame_stack < 1 or shift_padding < 0:
            raise ValueError("invalid channels, frame_stack or shift_padding")
        self.channels, self.frame_stack = channels, frame_stack
        self.shift_padding, self.output_dim = shift_padding, 256
        layers: list[nn.Module] = []
        incoming = channels * frame_stack
        for _ in range(4):
            layers.extend((nn.Conv2d(incoming, 32, 3, stride=2, padding=1), nn.ReLU()))
            incoming = 32
        self.convolution = nn.Sequential(*layers)
        self.image_projection = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 6 * 10, 256), nn.LayerNorm(256), nn.ReLU()
        )
        self.vehicle = nn.Sequential(
            nn.Linear(VEHICLE_FEATURE_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(nn.Linear(320, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        images, vehicle = observation["images"], observation["vehicle"]
        expected = (self.channels * self.frame_stack, 90, 160)
        if images.ndim != 4 or images.shape[1:] != expected or images.dtype != torch.uint8:
            raise ValueError(f"stack images must be uint8 (batch, {expected})")
        if vehicle.shape != (len(images), VEHICLE_FEATURE_DIM):
            raise ValueError("vehicle features do not match image batch")
        values = images.float() / 255.0 - 0.5
        if self.training:
            values = random_shift(values, self.shift_padding)
        visual = self.image_projection(self.convolution(values))
        return cast(torch.Tensor, self.fusion(torch.cat((visual, self.vehicle(vehicle)), dim=-1)))
