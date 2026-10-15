"""Actor-critic models with independently constructed observation encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.spec import ComponentSpec
from trackmaniarl.models.actors import (
    CategoricalActor,
    GaussianActor,
    GaussianActorConfig,
    PpoGaussianActor,
)
from trackmaniarl.models.critics import (
    ContinuousQCritic,
    ContinuousValueCritic,
    QuantileCritic,
    QuantileCriticConfig,
)
from trackmaniarl.models.factory import _component


class SensorActorCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_dim: int = Field(default=256, ge=1)
    action_count: int = Field(default=78, ge=2)
    action_low: tuple[float, ...] = (0.0, 0.0, -1.0)
    action_high: tuple[float, ...] = (1.0, 1.0, 1.0)
    critic_count: int = Field(default=10, ge=2)
    quantile_count: int = Field(default=25, ge=2)


class SensorActorCriticModelFactory:
    """Select an algorithm and an encoder without coupling either to a sensor."""

    def __init__(
        self,
        algorithm: str,
        encoder: ComponentSpec | Mapping[str, Any],
        config: SensorActorCriticConfig | Mapping[str, Any] | None = None,
    ) -> None:
        algorithm = {"discrete-sac": "sd-sac", "dsac": "sd-sac"}.get(algorithm, algorithm)
        contracts = {
            "sac": ModelContract.CONTINUOUS_ACTOR_CRITIC,
            "redq": ModelContract.ENSEMBLE_ACTOR_CRITIC,
            "tqc": ModelContract.CONTINUOUS_QUANTILE_ACTOR_CRITIC,
            "sd-sac": ModelContract.DISCRETE_ACTOR_CRITIC,
            "ppo": ModelContract.CONTINUOUS_ACTOR_VALUE,
        }
        if algorithm not in contracts:
            raise ValueError(f"unknown actor-critic algorithm: {algorithm}")
        self.model_contract = contracts[algorithm]
        self.algorithm = algorithm
        self.encoder = ComponentSpec.model_validate(encoder)
        self.config = SensorActorCriticConfig.model_validate({} if config is None else config)

    def _encoder(self) -> nn.Module:
        encoder = _component(self.encoder)
        if getattr(encoder, "output_dim", None) != self.config.feature_dim:
            raise ValueError("encoder output_dim must match actor-critic feature_dim")
        return encoder

    def build(self) -> nn.Module:
        model = nn.Module()
        shape = self.config
        width = shape.feature_dim
        if self.algorithm == "sd-sac":
            model.actor = CategoricalActor(self._encoder(), width, shape.action_count)
            model.q1 = nn.Sequential(self._encoder(), nn.Linear(width, shape.action_count))
            model.q2 = nn.Sequential(self._encoder(), nn.Linear(width, shape.action_count))
            return model
        action_dim = len(shape.action_low)
        actor = PpoGaussianActor if self.algorithm == "ppo" else GaussianActor
        model.actor = actor(
            self._encoder(),
            GaussianActorConfig(width, action_dim, shape.action_low, shape.action_high),
        )
        if self.algorithm == "ppo":
            model.value = ContinuousValueCritic(self._encoder(), width)
        elif self.algorithm == "sac":
            model.q1 = ContinuousQCritic(self._encoder(), width, action_dim)
            model.q2 = ContinuousQCritic(self._encoder(), width, action_dim)
        elif self.algorithm == "redq":
            model.critics = nn.ModuleList(
                ContinuousQCritic(self._encoder(), width, action_dim)
                for _ in range(shape.critic_count)
            )
        else:
            model.critics = nn.ModuleList(
                QuantileCritic(
                    self._encoder(), QuantileCriticConfig(width, action_dim, shape.quantile_count)
                )
                for _ in range(shape.critic_count)
            )
        return model
