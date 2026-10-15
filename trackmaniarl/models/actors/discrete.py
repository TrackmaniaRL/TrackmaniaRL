"""Categorical policy heads for discrete TrackMania actions."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from trackmaniarl.core.contracts import PolicyMode


class CategoricalActor(nn.Module):
    """Categorical policy exposing logits, sampled actions and deterministic argmax actions."""

    def __init__(self, encoder: nn.Module, feature_dim: int, action_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.logits = nn.Linear(feature_dim, action_count)

    def forward(
        self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(self.encoder(observation))
        distribution: Any = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if mode is PolicyMode.EVALUATION else distribution.sample()
        return actions, distribution.log_prob(actions)

    def probabilities(self, observation: Any) -> torch.Tensor:
        return cast(torch.Tensor, self.logits(self.encoder(observation)).softmax(dim=-1))

    def log_probabilities(self, observation: Any) -> torch.Tensor:
        """Keep gradients for unlikely actions even when their probabilities underflow."""
        return cast(
            torch.Tensor, self.logits(self.encoder(observation)).float().log_softmax(dim=-1)
        )
