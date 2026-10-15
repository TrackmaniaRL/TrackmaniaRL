"""Frozen nearest-reference action support for explicit evaluation ablations.

Feature selection, normalization and reference provenance belong to the caller.
The filter has no recurrent state and never learns from evaluation observations.
It restricts existing finite Q values; it cannot make an excluded action valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from trackmaniarl.core.contracts import PolicyMode


@dataclass(frozen=True, slots=True)
class ReferenceSupportConfig:
    """Distances use the caller's fixed feature scaling and squared Euclidean norm."""

    action_count: int
    neighbors: int = 15
    max_squared_distance: float = 9.0
    majority_only: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.action_count, bool) or not isinstance(self.action_count, int):
            raise ValueError("action_count must be a positive integer")
        if isinstance(self.neighbors, bool) or not isinstance(self.neighbors, int):
            raise ValueError("neighbors must be a positive integer")
        if self.action_count < 1 or self.neighbors < 1:
            raise ValueError("action_count and neighbors must be positive")
        if not isfinite(self.max_squared_distance) or self.max_squared_distance < 0:
            raise ValueError("max_squared_distance must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ReferenceSupportRequest:
    """Per-decision mode and matched on/off ablation switch."""

    mode: PolicyMode = PolicyMode.EVALUATION
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ReferenceSupportDecision:
    """One decision, preserving the proposal and post-filter command separately.

    ``q_gap`` is the source Q advantage lost by intervention, not a physical
    steering distance or a confidence estimate. ``values`` retains input shape.
    """

    values: torch.Tensor
    proposed_action: int
    selected_action: int
    q_gap: float
    supported_action_count: int
    nearest_squared_distance: float | None
    reason: str

    def log_record(self) -> dict[str, str | int | float | bool | None]:
        """Return a JSON-compatible record for every decision, including bypasses."""
        return {
            "proposed_action": self.proposed_action,
            "selected_action": self.selected_action,
            "intervened": self.proposed_action != self.selected_action,
            "q_gap": self.q_gap,
            "supported_action_count": self.supported_action_count,
            "nearest_squared_distance": self.nearest_squared_distance,
            "reason": self.reason,
        }


class ReferenceActionSupport:
    """Restrict evaluation choices using a copied, immutable reference bank.

    Inputs are reference features ``[N, D]`` and integer action IDs ``[N]``.
    Queries are ``[D]`` and Q values are ``[A]`` or ``[1, A]`` on the same
    device. No temporal state needs an episode reset. The caller must freeze and
    identify the bank, scaling and activation rule before evaluation; none is
    inferred from the map. This component deliberately rejects training mode.
    """

    def __init__(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        *,
        config: ReferenceSupportConfig,
    ) -> None:
        self.config = config
        self._validate_bank(features, actions)
        self._features = features.detach().clone()
        self._actions = actions.detach().to(dtype=torch.long).clone()

    def _validate_bank(self, features: torch.Tensor, actions: torch.Tensor) -> None:
        if features.ndim != 2 or min(features.shape) < 1:
            raise ValueError("reference features must have nonempty shape [N, D]")
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("reference features must be finite floating-point values")
        if actions.shape != (features.shape[0],) or actions.device != features.device:
            raise ValueError("reference actions must have shape [N] and match feature device")
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("reference actions must have integer dtype")
        if bool(((actions < 0) | (actions >= self.config.action_count)).any()):
            raise ValueError("reference action outside the configured action space")

    def _neighbors(self, query: torch.Tensor) -> tuple[torch.Tensor | None, float]:
        if query.shape != self._features.shape[1:] or query.device != self._features.device:
            raise ValueError("query must have shape [D] and match reference device")
        if query.dtype != self._features.dtype or not bool(torch.isfinite(query).all()):
            raise ValueError("query must be finite and match reference dtype")
        distances = (self._features - query).square().sum(dim=-1)
        nearest = distances.topk(min(self.config.neighbors, len(distances)), largest=False)
        distance = float(nearest.values[0])
        if not isfinite(distance):
            raise ValueError("reference distance overflowed; rescale the features")
        if distance > self.config.max_squared_distance:
            return None, distance
        votes = torch.bincount(self._actions[nearest.indices], minlength=self.config.action_count)
        return votes, distance

    @torch.no_grad()
    def votes(self, query: torch.Tensor) -> torch.Tensor | None:
        """Return nearest-reference vote counts, or None for a distant query."""
        return self._neighbors(query)[0]

    @torch.no_grad()
    def apply(
        self,
        q_values: torch.Tensor,
        query: torch.Tensor,
        *,
        request: ReferenceSupportRequest | None = None,
    ) -> ReferenceSupportDecision:
        """Select the highest-Q supported action, with an explicit unchanged fallback.

        ``enabled=False`` is the matched off condition. Distant states and an
        empty intersection with finite actions preserve the original values.
        NaN, positive infinity and an entirely excluded action space are errors.
        """
        request = request or ReferenceSupportRequest()
        if request.mode is not PolicyMode.EVALUATION:
            raise ValueError("reference support is evaluation-only; training is not supported")
        self._validate_values(q_values)
        proposed = int(q_values.argmax())
        values, distance, reason = q_values, None, "disabled"
        if request.enabled:
            votes, distance = self._neighbors(query)
            reason = "distant"
            if votes is not None:
                support = votes == votes.max() if self.config.majority_only else votes > 0
                masked = q_values.masked_fill(~support, -torch.inf)
                values = masked if bool(torch.isfinite(masked).any()) else q_values
                reason = "supported" if values is masked else "empty_intersection"
        selected = int(values.argmax())
        flat = q_values.reshape(-1)
        return ReferenceSupportDecision(
            values=values,
            proposed_action=proposed,
            selected_action=selected,
            q_gap=float(flat[proposed] - flat[selected]),
            supported_action_count=int(torch.isfinite(values).sum()),
            nearest_squared_distance=distance,
            reason=reason,
        )

    def _validate_values(self, values: torch.Tensor) -> None:
        if values.shape not in ((self.config.action_count,), (1, self.config.action_count)):
            raise ValueError("Q values must have shape [A] or [1, A]")
        if values.device != self._features.device or not values.is_floating_point():
            raise ValueError("Q values must be floating-point and match reference device")
        if bool((torch.isnan(values) | torch.isposinf(values)).any()):
            raise ValueError("Q values cannot contain NaN or positive infinity")
        if not bool(torch.isfinite(values).any()):
            raise ValueError("Q values require at least one finite action")
