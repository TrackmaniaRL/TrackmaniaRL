"""Supervised initialization of a value policy, without offline TD bootstrapping.

The resulting module remains an ordinary value policy. Labels and class counts
belong to this training helper and must never be exported with inference weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import ValuePhase


def balanced_action_weights(actions: torch.Tensor, action_count: int) -> torch.Tensor:
    """Give each demonstrated action equal total weight; leave unseen actions zero."""
    if actions.ndim != 1 or actions.dtype != torch.long or not actions.numel():
        raise ValueError("a nonempty vector of integer action labels is required")
    if action_count < 2 or bool(((actions < 0) | (actions >= action_count)).any()):
        raise ValueError("action labels must belong to the model action space")
    counts = torch.bincount(actions, minlength=action_count)
    present = counts > 0
    weights = torch.zeros(action_count, device=actions.device, dtype=torch.float32)
    weights[present] = actions.numel() / (present.sum() * counts[present])
    return weights


class BehaviorCloningPretrainer:
    """Fit the same neutral quantile-mean decisions used by greedy evaluation.

    No target network, rewards, future observations, timestamps, replay playback,
    or teacher action selection participate in the loss or policy forward pass.
    """

    gradient_clip_norm = 10.0

    def __init__(
        self,
        model: CompositeValueModel,
        actions: torch.Tensor,
        *,
        learning_rate: float = 0.0003,
    ) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("a positive finite learning rate is required")
        self.model = model
        self.class_weights = balanced_action_weights(actions, model.action_count)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.updates = 0

    def logits(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.model.encode_frames(observations)
        support = self.model.support(encoded, ValuePhase.EVALUATE)
        quantiles = self.model.head.evaluate_all(encoded, support)
        if not bool(torch.isfinite(quantiles).all()):
            raise ValueError("nonfinite policy quantiles")
        return cast(torch.Tensor, quantiles.mean(dim=-2))

    def update(
        self, observations: Mapping[str, torch.Tensor], actions: torch.Tensor
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.logits(observations)
        loss = F.cross_entropy(logits, actions, weight=self.class_weights)
        torch.autograd.backward(loss)
        norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.updates += 1
        return {"balanced_cross_entropy": float(loss.detach()), "gradient_norm": float(norm)}
