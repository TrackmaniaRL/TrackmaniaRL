"""Generic MLP sensor encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import cast

import torch
from torch import nn


class MlpSensorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim) < 1:
            raise ValueError("encoder dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 2 or frames.shape[-1] != self.input_dim:
            raise ValueError("MLP frames must have shape (frames, input_dim)")
        return cast(torch.Tensor, self.network(frames))


class StructuredMlpSensorEncoder(nn.Module):
    """MLP over explicitly shaped named sensors, independent of any game profile.

    Sensor order follows sorted names, so equivalent configuration mappings have
    identical semantics. Batch and sequence axes are preserved.
    """

    def __init__(
        self,
        input_shapes: Mapping[str, Sequence[int]],
        output_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if not input_shapes or any(not isinstance(name, str) for name in input_shapes):
            raise ValueError("explicit named sensor shapes required")
        self.input_shapes = {name: tuple(input_shapes[name]) for name in sorted(input_shapes)}
        if any(
            not shape or any(type(size) is not int or size < 1 for size in shape)
            for shape in self.input_shapes.values()
        ):
            raise ValueError("sensor dimensions must be positive integers")
        self.input_dim = sum(prod(shape) for shape in self.input_shapes.values())
        self.output_dim = output_dim
        self.encoder = MlpSensorEncoder(self.input_dim, output_dim, hidden_dim)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(observation) != set(self.input_shapes):
            raise ValueError("observation must contain exactly the configured sensors")
        sensors = []
        leading = None
        for name, shape in self.input_shapes.items():
            value = observation[name]
            if value.ndim < len(shape) or tuple(value.shape[-len(shape) :]) != shape:
                raise ValueError(f"sensor shape differs: {name}")
            axes = value.shape[: -len(shape)]
            if leading is not None and axes != leading:
                raise ValueError("sensor batch and sequence axes must agree")
            leading = axes
            sensors.append(value.reshape(*axes, -1))
        merged = torch.cat(sensors, dim=-1)
        assert leading is not None
        encoded = self.encoder(merged.reshape(-1, self.input_dim))
        return cast(torch.Tensor, encoded.reshape(*leading, self.output_dim))
