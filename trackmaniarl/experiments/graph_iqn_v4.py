"""Order-preserving residual track adapter for the V3 TrackMania policy.

V4 retains the complete V3 policy and adds a small directional Conv1D branch
over the 44 ordered lookahead stations.  The branch uses dilations to combine
near and far geometry without the reversal-invariant global mean reduction of
the baseline graph.  Its output projection is zero-initialized, so a V3 warm
start is exactly policy preserving before the first optimizer update.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import cast

import torch
from torch import nn

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.experiments.graph_iqn_v2 import TRACK_SCALE_M
from trackmaniarl.experiments.graph_iqn_v3 import (
    CONTEXT_V3_DIM,
    TrackGnnSimbaEncoderV3,
)


class _DilatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, value + self.block(value))


class OrderedTrackResidualAdapter(nn.Module):
    """Encode ordered boundary stations without discarding their direction."""

    output_dim = 192
    dilations: Sequence[int] = (1, 2, 4, 8)
    fingerprint_parameters = ("point_count", "dilations")

    def __init__(self, *, point_count: int = 44, hidden_dim: int = 64) -> None:
        super().__init__()
        if point_count < 2 or hidden_dim < 8 or hidden_dim % 8:
            raise ValueError("ordered adapter dimensions are invalid")
        self.point_count = point_count
        self.input_projection = nn.Conv1d(7, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            _DilatedResidualBlock(hidden_dim, dilation) for dilation in self.dilations
        )
        self.output_projection = nn.Linear(hidden_dim * 3, self.output_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        position = torch.linspace(0.0, 1.0, point_count, dtype=torch.float32)
        self.register_buffer("normalized_position", position, persistent=False)

    def forward(self, track: torch.Tensor) -> torch.Tensor:
        if track.ndim != 3 or track.shape[1:] != (3, self.point_count * 2):
            raise ValueError("ordered adapter expects paired XZ coordinates")
        nodes = (
            track.reshape(track.shape[0], 3, self.point_count, 2)
            .permute(0, 1, 3, 2)
            .reshape(track.shape[0], 6, self.point_count)
        )
        position = cast(torch.Tensor, self.normalized_position).to(dtype=track.dtype)
        position = position.view(1, 1, -1).expand(track.shape[0], -1, -1)
        hidden = self.input_projection(torch.cat((nodes, position), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        pooled = torch.cat((hidden[:, :, 0], hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=-1)
        return cast(torch.Tensor, self.output_projection(pooled))


class TrackGnnSimbaEncoderV4(TrackGnnSimbaEncoderV3):
    """V3 encoder plus an initially inert ordered-geometry correction."""

    fingerprint_parameters: tuple[str, ...] = (
        *TrackGnnSimbaEncoderV3.fingerprint_parameters,
        "spatial_residual_scale",
    )

    def __init__(
        self,
        context_residual_scale: float = 1.0,
        spatial_residual_scale: float = 1.0,
    ) -> None:
        super().__init__(context_residual_scale=context_residual_scale)
        if not isfinite(spatial_residual_scale) or not 0.0 < spatial_residual_scale <= 1.0:
            raise ValueError("spatial_residual_scale must be finite and in (0, 1]")
        self.spatial_residual_scale = float(spatial_residual_scale)
        self.spatial_adapter = OrderedTrackResidualAdapter()

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
        scaled_track = track / TRACK_SCALE_M
        track_latent = self.track_conv(scaled_track) + self._spatial_correction(scaled_track)
        physics_latent = self.physics_proj(physics) + self._context_correction(context)
        return cast(torch.Tensor, self.backbone(torch.cat((track_latent, physics_latent), dim=-1)))

    def _spatial_correction(self, track: torch.Tensor) -> torch.Tensor:
        return self.spatial_residual_scale * torch.tanh(self.spatial_adapter(track))


class SpatialAdapterOnlyDiscreteValueLearner(DiscreteValueLearner):
    """Freeze the V3 policy and optimize only the new ordered track branch."""

    def _prepare_model(self) -> None:
        super()._prepare_model()
        assert isinstance(self.model.encoder, TrackGnnSimbaEncoderV4)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.encoder.spatial_adapter.parameters():
            parameter.requires_grad_(True)
