"""A short, complete PPO run retaining the long campaign's learning-rate horizon."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Unpack

from torch import nn

from trackmaniarl.algorithms.ppo_config import PPOOptions
from trackmaniarl.algorithms.proximal_policy_optimization import ProximalPolicyOptimization


class CampaignPrefixPPO(ProximalPolicyOptimization):
    """Change only setup's LR horizon; keep collection, losses and optimizer unchanged."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        schedule_transitions: int,
        **options: Unpack[PPOOptions],
    ) -> None:
        if type(schedule_transitions) is not int or schedule_transitions <= 0:
            raise ValueError("PPO schedule horizon must be a positive integer")
        self.schedule_transitions = schedule_transitions
        super().__init__(model, **options)

    def setup(self, context: Mapping[str, Any]) -> None:
        budget = context.get("total_transitions")
        if type(budget) is not int or not 0 < budget <= self.schedule_transitions:
            raise ValueError("Pilot budget must be positive and within the campaign horizon")
        super().setup({**context, "total_transitions": self.schedule_transitions})
