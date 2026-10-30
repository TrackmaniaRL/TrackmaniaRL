"""Fixed-location quantile head used by QR-DQN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self, cast

import torch
from torch import nn

from trackmaniarl.models.contracts import ValueRepresentation, ValueSupport


@dataclass(frozen=True, slots=True)
class FixedQuantileHeadConfig:
    feature_dim: int
    action_count: int
    quantile_count: int = 32
    dueling: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        return cls(**dict(values))


class FixedQuantileHead(nn.Module):
    representation = ValueRepresentation.FIXED_QUANTILE

    def __init__(self, config: FixedQuantileHeadConfig | Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(config, FixedQuantileHeadConfig):
            config = FixedQuantileHeadConfig.from_mapping(config)
        if min(config.feature_dim, config.action_count, config.quantile_count) < 1:
            raise ValueError("head dimensions must be positive")
        self.feature_dim = config.feature_dim
        self.action_count = config.action_count
        self.quantile_count = config.quantile_count
        # Execution-only opt-in; parameters, quantile locations and checkpoint
        # names are unchanged. Sparse selection remains the default.
        self.dense_cuda_selected = False
        self.advantage = nn.Linear(config.feature_dim, config.action_count * config.quantile_count)
        self.value = (
            nn.Linear(config.feature_dim, config.quantile_count) if config.dueling else None
        )

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor:
        self._validate_support(support)
        leading = features.shape[:-1]
        advantages = self.advantage(features).reshape(
            *leading, self.quantile_count, self.action_count
        )
        if self.value is None:
            return cast(torch.Tensor, advantages)
        value = self.value(features).unsqueeze(-1)
        return cast(torch.Tensor, value + advantages - advantages.mean(dim=-1, keepdim=True))

    def evaluate_actions(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_support(support)
        if (
            self.dense_cuda_selected
            and features.is_cuda
            and features.dtype == self.advantage.weight.dtype == torch.float32
            and actions.dtype == torch.int64
            and features.numel() // self.feature_dim >= 64
        ):
            # Large CUDA batches favor GEMM and its dense backward over a
            # batch-sized gather of weights followed by deterministic scatter.
            values = self.evaluate_all(features, support)
            indices = actions[..., None, None].expand(*actions.shape, self.quantile_count, 1)
            return values.gather(-1, indices).squeeze(-1)
        quantiles = torch.arange(self.quantile_count, device=actions.device)
        rows = actions.unsqueeze(-1) + self.action_count * quantiles
        selected_weight = self.advantage.weight[rows]
        selected_bias = self.advantage.bias[rows]
        selected = (features.unsqueeze(-2) * selected_weight).sum(dim=-1) + selected_bias
        if self.value is None:
            return selected
        mean_advantage = self._mean_advantage(features)
        return cast(torch.Tensor, self.value(features) + selected - mean_advantage)

    def _mean_advantage(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.advantage.weight.reshape(
            self.quantile_count, self.action_count, self.feature_dim
        )
        biases = self.advantage.bias.reshape(self.quantile_count, self.action_count)
        mean_advantage = torch.einsum("...d,nd->...n", features, weights.mean(dim=1))
        return mean_advantage + biases.mean(dim=1)

    def _validate_support(self, support: ValueSupport) -> None:
        if support.points.shape[-1] != self.quantile_count:
            raise ValueError("fixed head and strategy quantile counts must match")
