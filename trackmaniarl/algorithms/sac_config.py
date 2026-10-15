from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import torch

from trackmaniarl.algorithms.execution import TorchExecutionConfig


def _validate_float32_temperature(name: str, value: float, *, learned: bool) -> None:
    coefficient = torch.tensor(value, dtype=torch.float32)
    if not bool(torch.isfinite(coefficient) & (coefficient > 0)):
        raise ValueError(f"SD-SAC {name} must remain finite and positive in float32")
    if learned:
        # Initialization logs a float32 coefficient; bounds cast math.log(value).
        logs = torch.stack((coefficient.log(), torch.tensor(math.log(value), dtype=torch.float32)))
        restored = logs.exp()
        if not bool((torch.isfinite(restored) & (restored > 0)).all()):
            raise ValueError(f"SD-SAC {name} must remain finite and positive after float32 log/exp")


class SACOptions(TypedDict, total=False):
    model_factory: Any | None
    target_tau: float
    learning_rate: float
    entropy_coefficient: float
    target_entropy: float | None
    learn_entropy_coefficient: bool
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


@dataclass(frozen=True, slots=True)
class SACConfig:
    model_factory: Any | None = None
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.2
    target_entropy: float | None = None
    learn_entropy_coefficient: bool = True
    execution: TorchExecutionConfig | Mapping[str, Any] | None = None
    seed: int = 0

    def validate(self) -> None:
        values = (self.target_tau, self.learning_rate, self.entropy_coefficient)
        if min(values) <= 0.0 or self.target_tau > 1.0:
            raise ValueError("target_tau, learning_rate, and entropy_coefficient must be positive")


class REDQOptions(TypedDict, total=False):
    model_factory: Any | None
    target_tau: float
    learning_rate: float
    entropy_coefficient: float
    target_subset_size: int
    policy_update_interval: int
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


@dataclass(frozen=True, slots=True)
class REDQConfig:
    model_factory: Any | None = None
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.2
    target_subset_size: int = 2
    policy_update_interval: int = 20
    execution: TorchExecutionConfig | Mapping[str, Any] | None = None
    seed: int = 0

    def validate(self) -> None:
        values = (self.target_tau, self.learning_rate, self.entropy_coefficient)
        if min(values) <= 0.0 or self.target_tau > 1.0:
            raise ValueError("target_tau, learning_rate, and entropy_coefficient must be positive")
        if self.target_subset_size < 1 or self.policy_update_interval < 1:
            raise ValueError("target_subset_size and policy_update_interval must be positive")


class TQCOptions(SACOptions, total=False):
    top_quantiles_to_drop_per_critic: int


@dataclass(frozen=True, slots=True)
class TQCConfig(SACConfig):
    top_quantiles_to_drop_per_critic: int = 2

    def validate(self) -> None:
        SACConfig.validate(self)
        if self.top_quantiles_to_drop_per_critic < 0:
            raise ValueError("top_quantiles_to_drop_per_critic must be non-negative")


class SDSACOptions(SACOptions, total=False):
    actor_learning_rate: float | None
    entropy_learning_rate: float | None
    entropy_coefficient_min: float | None
    entropy_coefficient_max: float | None
    actor_objective: Literal["sac", "soft_q_forward_kl"]
    q_clip_epsilon: float
    terminal_value_loss_coefficient: float
    entropy_penalty_coefficient: float
    entropy_penalty_reference: Literal["target_policy", "behavior"]


@dataclass(frozen=True, slots=True)
class SDSACConfig(SACConfig):
    actor_learning_rate: float | None = None
    entropy_learning_rate: float | None = None
    entropy_coefficient_min: float | None = None
    entropy_coefficient_max: float | None = None
    actor_objective: Literal["sac", "soft_q_forward_kl"] = "sac"
    q_clip_epsilon: float = 0.5
    terminal_value_loss_coefficient: float = 0.0
    entropy_penalty_coefficient: float = 0.5
    entropy_penalty_reference: Literal["target_policy", "behavior"] = "target_policy"

    def validate(self) -> None:
        SACConfig.validate(self)
        positive = {
            "learning_rate": self.learning_rate,
            "target_tau": self.target_tau,
            "entropy_coefficient": self.entropy_coefficient,
            "actor_learning_rate": self.actor_learning_rate,
            "entropy_learning_rate": self.entropy_learning_rate,
            "entropy_coefficient_min": self.entropy_coefficient_min,
            "entropy_coefficient_max": self.entropy_coefficient_max,
        }
        for name, value in positive.items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"SD-SAC {name} must be finite and positive")
        for name in ("entropy_coefficient", "entropy_coefficient_min", "entropy_coefficient_max"):
            value = positive[name]
            if value is not None:
                _validate_float32_temperature(name, value, learned=self.learn_entropy_coefficient)
        if self.target_entropy is not None and (
            not math.isfinite(self.target_entropy) or self.target_entropy < 0
        ):
            raise ValueError("SD-SAC target_entropy must be finite and non-negative")
        lower, upper = self.entropy_coefficient_min, self.entropy_coefficient_max
        if lower is not None and self.entropy_coefficient < lower:
            raise ValueError("entropy_coefficient must be at least entropy_coefficient_min")
        if upper is not None and self.entropy_coefficient > upper:
            raise ValueError("entropy_coefficient must not exceed entropy_coefficient_max")
        if self.actor_objective not in {"sac", "soft_q_forward_kl"}:
            raise ValueError("actor_objective must be 'sac' or 'soft_q_forward_kl'")
        if not math.isfinite(self.q_clip_epsilon) or self.q_clip_epsilon <= 0:
            raise ValueError("SD-SAC q_clip_epsilon must be finite and positive")
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.entropy_penalty_coefficient,
                self.terminal_value_loss_coefficient,
            )
        ):
            raise ValueError(
                "SD-SAC entropy and terminal penalty coefficients must be finite and non-negative"
            )
        if self.entropy_penalty_reference not in {"target_policy", "behavior"}:
            raise ValueError("entropy_penalty_reference must be 'target_policy' or 'behavior'")


# Existing imports remain readable; new code uses the paper's SD-SAC name.
DiscreteSACOptions = SDSACOptions
DiscreteSACConfig = SDSACConfig
