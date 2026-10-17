"""Experimental SD-SAC-inspired learner; excluded from the standard training campaign."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Unpack, cast

import torch
from torch import nn

from trackmaniarl.algorithms._torch import (
    TorchLearnerBase,
    evaluated_actor_state,
    polyak_update,
    weighted_mean,
)
from trackmaniarl.algorithms.sac_config import SDSACConfig, SDSACOptions
from trackmaniarl.algorithms.sac_support import (
    EntropyConfig,
    EntropyRestoreTarget,
    SACBatch,
    alpha_value,
    discrete_batch,
    entropy_state,
    restore_entropy_state,
)
from trackmaniarl.algorithms.sd_sac_objectives import (
    EntropyMatchedTarget,
    actor_diagnostics,
    categorical_statistics,
    entropy_matched_diagnostics,
    entropy_matched_target,
    soft_q_log_probabilities,
    terminal_value_loss,
)
from trackmaniarl.core.contracts import ModelContract, PolicyMode
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.core.pytree import sanitize_finite, tree_collate, tree_to_device
from trackmaniarl.models.backbones import HypersphericalLinear, project_hyperspherical_weights


def _batch_vector(value: Any, name: str, size: int) -> torch.Tensor:
    """Accept scalar transition fields as (B,) or an explicit (B, 1) column."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"SD-SAC {name} must be a tensor")
    if value.shape not in ((size,), (size, 1)):
        raise ValueError(f"SD-SAC {name} must have shape ({size},) or ({size}, 1)")
    return value.reshape(size)


def _action_values(value: Any, shape: tuple[int, int], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"SD-SAC {name} must have shape {shape}")
    return value


def _policy_shape(policy: tuple[torch.Tensor, torch.Tensor], size: int) -> tuple[int, int]:
    probabilities, logs = policy
    if probabilities.ndim != 2 or probabilities.shape[0] != size or probabilities.shape[1] < 2:
        raise ValueError("SD-SAC actor must produce [batch, actions] with at least two actions")
    shape = (size, int(probabilities.shape[1]))
    _action_values(logs, shape, "actor log probabilities")
    return shape


class _DiscretePolicy:
    def __init__(self, actor: nn.Module, device: torch.device) -> None:
        self.actor = deepcopy(actor).to(device).eval()
        self.device = device

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> Any:
        action, _ = self.act_with_info(observation, mode)
        return action

    def act_with_info(
        self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE
    ) -> tuple[int, Mapping[str, Any]]:
        observation = tree_to_device(tree_collate([sanitize_finite(observation)]), self.device)
        with torch.no_grad():
            actor = cast(Any, self.actor)
            probabilities, log_probabilities = categorical_statistics(actor, observation)
            distribution: Any = torch.distributions.Categorical(probs=probabilities)
            action = (
                probabilities.argmax(dim=-1)
                if mode is PolicyMode.EVALUATION
                else distribution.sample()
            )
            entropy = -(probabilities * log_probabilities).sum(-1).detach().cpu().reshape(-1)
        values = action.detach().cpu().reshape(-1)
        if values.numel() != 1:
            raise ValueError("Discrete policy must produce exactly one action per observation")
        return int(values.item()), {"_trackmaniarl_behavior_entropy": float(entropy.item())}

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.actor.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state)


@dataclass(frozen=True, slots=True)
class _DiscreteCriticStep:
    loss: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    targets: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor] = field(default_factory=dict)
    action_count: int = 0


@dataclass(frozen=True, slots=True)
class _DiscreteActorStep:
    loss: torch.Tensor
    entropy: torch.Tensor
    action_count: int
    diagnostics: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _DiscreteUpdate:
    critic: _DiscreteCriticStep
    actor: _DiscreteActorStep
    alpha_loss: torch.Tensor
    alpha: torch.Tensor
    alpha_used: torch.Tensor


class StableDiscreteSoftActorCritic(TorchLearnerBase):
    """Experimental double-average SAC with behavior or legacy entropy anchoring.

    Retained for isolated research and checkpoint compatibility. Trackmania pilots
    have not qualified this implementation for the multi-computer campaign.
    """

    accepted_model_contracts = frozenset({ModelContract.DISCRETE_ACTOR_CRITIC})
    supports_sequence_training = False
    supports_n_step = False

    def __init__(
        self,
        model: nn.Module | None = None,
        **options: Unpack[SDSACOptions],
    ) -> None:
        config = SDSACConfig(**options)
        super().__init__(
            model,
            model_factory=config.model_factory,
            execution=config.execution,
            seed=config.seed,
        )
        config.validate()
        self._configure(config)

    def _configure(self, config: SDSACConfig) -> None:
        self.learning_rate = config.learning_rate
        self.actor_learning_rate = (
            config.learning_rate
            if config.actor_learning_rate is None
            else config.actor_learning_rate
        )
        self.entropy_learning_rate = (
            config.learning_rate
            if config.entropy_learning_rate is None
            else config.entropy_learning_rate
        )
        self.entropy_coefficient_min = config.entropy_coefficient_min
        self.entropy_coefficient_max = config.entropy_coefficient_max
        self.actor_objective = config.actor_objective
        self.target_tau = config.target_tau
        self.initial_entropy_coefficient = config.entropy_coefficient
        self.entropy_mode = "learned" if config.learn_entropy_coefficient else "fixed"
        self.target_entropy = config.target_entropy
        self.q_clip_epsilon = config.q_clip_epsilon
        self.terminal_value_loss_coefficient = config.terminal_value_loss_coefficient
        self.entropy_penalty_coefficient = config.entropy_penalty_coefficient
        self.entropy_penalty_reference = config.entropy_penalty_reference
        self.actor_temperature = config.actor_temperature
        self.backup_entropy_coefficient = config.backup_entropy_coefficient

    def _setup_model(self) -> None:
        assert self.model is not None
        if not all(hasattr(self.model, name) for name in ("actor", "q1", "q2")):
            raise TypeError("SD-SAC model must expose actor, q1 and q2")
        if self.actor_objective == "soft_q_forward_kl" and not callable(
            getattr(self.model.actor, "log_probabilities", None)
        ):
            raise TypeError("soft_q_forward_kl requires actor.log_probabilities")
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self._setup_optimizers()
        config = EntropyConfig(
            self.initial_entropy_coefficient, self.entropy_learning_rate, self.entropy_mode
        )
        self.log_alpha, self.alpha_optimizer = entropy_state(config, self.device)

    def _setup_optimizers(self) -> None:
        assert self.model is not None
        groups = {
            name: list(getattr(self.model, name).parameters()) for name in ("actor", "q1", "q2")
        }
        owners: set[int] = set()
        for name, parameters in groups.items():
            identifiers = {id(parameter) for parameter in parameters}
            if owners.intersection(identifiers) or len(identifiers) != len(parameters):
                raise ValueError(f"SD-SAC actor, q1 and q2 parameters must be disjoint ({name})")
            owners.update(identifiers)
        self.actor_optimizer = torch.optim.Adam(groups["actor"], lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            groups["q1"] + groups["q2"],
            lr=self.learning_rate,
        )

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        with self.autocast():
            return self._update(batch)

    def _update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        assert self.model is not None
        if batch.metadata.get("n_step", 1) != 1:
            raise ValueError("SD-SAC requires n_step=1; generic n-step omits intermediate entropy")
        prepared = self._prepare_batch(batch)
        alpha = alpha_value(self.log_alpha, self.initial_entropy_coefficient, self.device)
        if alpha.numel() != 1 or not bool(torch.isfinite(alpha) & (alpha > 0)):
            raise ValueError("SD-SAC alpha must be finite and positive")
        critic = self._critic_step(prepared, alpha)
        policy = self._current_policy(prepared)
        if policy[0].shape[1] != critic.action_count:
            raise ValueError("SD-SAC current actor and critic action counts must match")
        self._optimize(critic.loss, self.critic_optimizer)
        # Forward normalizes effective rows, but Adam updates the raw rows.
        # Keep their radius fixed so the effective step cannot shrink as the
        # redundant raw norm drifts. Match the value learner's Simba contract.
        project_hyperspherical_weights(self.model.q1)
        project_hyperspherical_weights(self.model.q2)
        actor = self._actor_step(prepared, alpha, policy=policy)
        self._optimize(actor.loss, self.actor_optimizer)
        project_hyperspherical_weights(self.model.actor)
        alpha_loss = self._entropy_loss(actor)
        polyak_update(self.model, self.target_model, self.target_tau)
        current_alpha = alpha_value(self.log_alpha, self.initial_entropy_coefficient, self.device)
        return self._update_result(
            prepared, _DiscreteUpdate(critic, actor, alpha_loss, current_alpha, alpha)
        )

    def _prepare_batch(self, batch: TrainingBatch) -> SACBatch:
        size = len(batch.transition_ids)
        # A scalar column is supported; a declared or actual temporal batch is not.
        layout_rewards = batch.rewards
        if isinstance(layout_rewards, torch.Tensor) and layout_rewards.shape == (size, 1):
            layout_rewards = layout_rewards.reshape(size)
        self._validate_batch_layout(replace(batch, rewards=layout_rewards))
        if size == 0:
            raise ValueError("SD-SAC batch must contain transitions")
        values = {
            name: _batch_vector(getattr(batch, name), name, size)
            for name in ("actions", "rewards", "bootstrap_discounts", "terminated", "truncated")
        }
        if values["actions"].dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("SD-SAC actions must use an integer dtype, excluding bool")
        for name in ("terminated", "truncated"):
            if values[name].dtype != torch.bool:
                raise TypeError(f"SD-SAC {name} must use bool dtype")
        for name in ("rewards", "bootstrap_discounts"):
            if values[name].is_complex() or values[name].dtype == torch.bool:
                raise TypeError(f"SD-SAC {name} must contain real numeric values")
        weights = batch.importance_weights
        if weights is not None:
            weights = _batch_vector(weights, "importance_weights", size)
            if weights.is_complex() or weights.dtype == torch.bool:
                raise TypeError("SD-SAC importance_weights must contain real numeric values")
        # _batch waits for asynchronous transfer before any tensor-value checks.
        normalized = self._batch(
            replace(
                batch,
                actions=values["actions"],
                rewards=values["rewards"],
                bootstrap_discounts=values["bootstrap_discounts"],
                terminated=values["terminated"],
                truncated=values["truncated"],
                importance_weights=weights,
            )
        )
        prepared = replace(
            discrete_batch(normalized),
            actions=normalized.actions.long(),
            weights=(
                normalized.importance_weights.float()
                if normalized.importance_weights is not None
                else None
            ),
        )
        valid = (
            torch.isfinite(prepared.rewards).all()
            & torch.isfinite(prepared.discounts).all()
            & (prepared.discounts >= 0).all()
            & (prepared.actions >= 0).all()
        )
        if prepared.weights is not None:
            total = prepared.weights.sum()
            valid = (
                valid
                & torch.isfinite(prepared.weights).all()
                & (prepared.weights >= 0).all()
                & torch.isfinite(total)
                & (total > 0)
            )
        if not bool(valid):
            raise ValueError(
                "SD-SAC requires finite rewards/discounts, non-negative discounts/actions, "
                "and finite non-negative importance_weights with positive finite sum"
            )
        if prepared.weights is not None:
            # Preserve relative weights even below the shared mean's denominator floor.
            prepared = replace(prepared, weights=prepared.weights / total)
        return prepared

    def _current_policy(self, batch: SACBatch) -> tuple[torch.Tensor, torch.Tensor]:
        policy = categorical_statistics(self.model.actor, batch.observations)
        shape = _policy_shape(policy, batch.rewards.numel())
        self._desired_entropy(shape[1])
        valid = torch.isfinite(policy[0]).all() & torch.isfinite(policy[1]).all()
        valid = valid & (batch.actions < shape[1]).all()
        if self.entropy_penalty_coefficient and self.entropy_penalty_reference == "behavior":
            reference = self._entropy_reference(batch, batch.rewards)
            valid = valid & (reference <= math.log(shape[1]) + 1e-6).all()
        if not bool(valid):
            raise ValueError("SD-SAC invalid actor probabilities, action index or behavior entropy")
        return policy

    def _critic_targets(self, batch: SACBatch, alpha: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_probabilities, next_log_probabilities = categorical_statistics(
                self.model.actor, batch.next_observations
            )
            next_shape = _policy_shape(
                (next_probabilities, next_log_probabilities), batch.rewards.numel()
            )
            self._desired_entropy(next_probabilities.shape[-1])
            q1 = _action_values(
                self.target_model.q1(batch.next_observations), next_shape, "target q1"
            )
            q2 = _action_values(
                self.target_model.q2(batch.next_observations), next_shape, "target q2"
            )
            next_q = 0.5 * q1 + 0.5 * q2
            # An explicit coefficient decouples the backed-up entropy bonus from the
            # temperature that shapes the actor target; zero backs up E_pi[Q].
            weight = (
                alpha
                if self.backup_entropy_coefficient is None
                else alpha.new_tensor(self.backup_entropy_coefficient)
            )
            next_value = (next_probabilities * (next_q - weight * next_log_probabilities)).sum(1)
            terminal = batch.source.terminated.bool().reshape(-1) & batch.discounts.eq(0)
            continuation = torch.where(terminal, torch.zeros_like(next_value), next_value)
            targets = batch.rewards + batch.discounts * continuation
            if not bool(torch.isfinite(continuation).all() & torch.isfinite(targets).all()):
                raise ValueError("SD-SAC non-finite continuation or critic target")
            return targets

    def _critic_step(self, batch: SACBatch, alpha: torch.Tensor) -> _DiscreteCriticStep:
        targets = self._critic_targets(batch, alpha)
        indices = batch.actions[:, None]
        values1 = self.model.q1(batch.observations)
        if values1.ndim != 2 or values1.shape[1] < 2:
            raise ValueError("SD-SAC q1 must produce [batch, actions] with at least two actions")
        shape = (batch.rewards.numel(), int(values1.shape[1]))
        values1 = _action_values(values1, shape, "q1")
        values2 = _action_values(self.model.q2(batch.observations), shape, "q2")
        valid = torch.isfinite(values1).all() & torch.isfinite(values2).all()
        if not bool(valid & (batch.actions >= 0).all() & (batch.actions < shape[1]).all()):
            raise ValueError("SD-SAC current critics or action indices are invalid")
        q1 = values1.gather(1, indices).squeeze(1)
        q2 = values2.gather(1, indices).squeeze(1)
        losses, diagnostics = self._critic_losses(batch, (q1, q2, targets), shape=shape)
        loss = weighted_mean(losses, batch.weights)
        terminal_loss = loss.new_zeros(())
        if self.terminal_value_loss_coefficient:
            terminal_loss = terminal_value_loss(
                (q1, q2), batch.rewards, batch.source.terminated, batch.weights
            )
            loss = loss + self.terminal_value_loss_coefficient * terminal_loss
        return _DiscreteCriticStep(
            loss,
            q1,
            q2,
            targets,
            {**diagnostics, "loss/terminal_value": terminal_loss.detach()},
            action_count=shape[1],
        )

    def _critic_losses(
        self,
        batch: SACBatch,
        values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        q1, q2, targets = values
        indices = batch.actions[:, None]
        with torch.no_grad():
            values1 = self.target_model.q1(batch.observations)
            values2 = self.target_model.q2(batch.observations)
            if shape is not None:
                _action_values(values1, shape, "current target q1")
                _action_values(values2, shape, "current target q2")
            if not bool(torch.isfinite(values1).all() & torch.isfinite(values2).all()):
                raise ValueError("SD-SAC current target critics produced non-finite values")
            target_q1 = values1.gather(1, indices).squeeze(1)
            target_q2 = values2.gather(1, indices).squeeze(1)
        clipped_q1 = target_q1 + (q1 - target_q1).clamp(-self.q_clip_epsilon, self.q_clip_epsilon)
        clipped_q2 = target_q2 + (q2 - target_q2).clamp(-self.q_clip_epsilon, self.q_clip_epsilon)
        losses = torch.maximum((q1 - targets).square(), (clipped_q1 - targets).square())
        losses = losses + torch.maximum((q2 - targets).square(), (clipped_q2 - targets).square())
        with torch.no_grad():
            outside = torch.stack(
                [
                    (q1 - target_q1).abs() > self.q_clip_epsilon,
                    (q2 - target_q2).abs() > self.q_clip_epsilon,
                ]
            )
            clipped_wins = torch.stack(
                [
                    (clipped_q1 - targets).square() > (q1 - targets).square(),
                    (clipped_q2 - targets).square() > (q2 - targets).square(),
                ]
            )
            terminal = batch.source.terminated.bool().reshape(-1)
            terminal_count = terminal.float().sum()
            terminal_error = (0.5 * (q1 + q2) - targets).abs()
            diagnostics = {
                "critic/clipped_fraction": outside.float().mean(),
                "critic/clip_blocked_gradient_fraction": (outside & clipped_wins).float().mean(),
                "critic/terminal_samples": terminal_count,
                "critic/terminal_td_abs_error_sum": (terminal_error * terminal.float()).sum(),
                "critic/terminal_td_mae": (terminal_error * terminal.float()).sum()
                / terminal_count.clamp_min(1),
            }
        return losses, diagnostics

    def _actor_step(
        self,
        batch: SACBatch,
        alpha: torch.Tensor,
        *,
        policy: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> _DiscreteActorStep:
        # Actor parameters are disjoint from critics; reuse its graph across the
        # critic step while evaluating the actor loss against the updated critics.
        probabilities, log_probabilities = (
            categorical_statistics(self.model.actor, batch.observations)
            if policy is None
            else policy
        )
        shape = _policy_shape((probabilities, log_probabilities), probabilities.shape[0])
        # The actor improves against fixed critic values. Building a critic graph
        # only to detach it wastes memory and compute on every actor update.
        with torch.no_grad():
            q1 = _action_values(self.model.q1(batch.observations), shape, "q1").float()
            q2 = _action_values(self.model.q2(batch.observations), shape, "q2").float()
            q_values = 0.5 * q1 + 0.5 * q2
            matched: EntropyMatchedTarget | None = None
            if self.actor_temperature == "entropy_matched":
                matched = entropy_matched_target(q_values, self._desired_entropy(shape[1]))
                target_log_probabilities = matched.log_probabilities
            else:
                target_log_probabilities = soft_q_log_probabilities(q_values, alpha)
        if self.actor_objective == "soft_q_forward_kl":
            # An explicit variant, not canonical SD-SAC. Detached cross-entropy
            # keeps a gradient toward missed actions even for saturated logits.
            actor_loss = -(target_log_probabilities.exp() * log_probabilities).sum(1).mean()
        else:
            actor_loss = (probabilities * (alpha * log_probabilities - q_values)).sum(1).mean()
        entropy = -(probabilities * log_probabilities).sum(1)
        diagnostics = actor_diagnostics(
            (probabilities.detach(), log_probabilities.detach()), (q1, q2), target_log_probabilities
        )
        if matched is not None:
            diagnostics.update(entropy_matched_diagnostics(matched))
        desired = self._desired_entropy(probabilities.shape[-1])
        diagnostics["policy/entropy_target"] = entropy.new_tensor(desired)
        diagnostics["policy/entropy_target_error"] = entropy.detach().mean() - desired
        diagnostics["loss/entropy_penalty"] = torch.zeros((), device=entropy.device)
        if self.entropy_penalty_coefficient:
            reference = self._entropy_reference(batch, entropy)
            penalty = self.entropy_penalty_coefficient * (entropy - reference).square().mean()
            actor_loss = actor_loss + penalty
            diagnostics["loss/entropy_penalty"] = penalty.detach()
            diagnostics["policy/reference_entropy"] = reference.mean()
            diagnostics["policy/entropy_reference_gap"] = (entropy.detach() - reference).mean()
        return _DiscreteActorStep(actor_loss, entropy, int(probabilities.shape[-1]), diagnostics)

    def _entropy_penalty(self, batch: SACBatch, entropy: torch.Tensor) -> torch.Tensor:
        return (entropy - self._entropy_reference(batch, entropy)).square().mean()

    def _entropy_reference(self, batch: SACBatch, entropy: torch.Tensor) -> torch.Tensor:
        if self.entropy_penalty_reference == "behavior":
            stored = batch.source.metadata.get("behavior_entropies")
            if not isinstance(stored, torch.Tensor) or stored.shape != entropy.shape:
                raise ValueError(
                    "Behavior entropy penalty requires one stored entropy per transition"
                )
            target_entropy = stored.detach().to(device=entropy.device, dtype=entropy.dtype)
            if not bool(torch.isfinite(target_entropy).all() & (target_entropy >= 0).all()):
                raise ValueError("Stored behavior entropies must be finite and non-negative")
            return target_entropy
        with torch.no_grad():
            target_probabilities, target_logs = categorical_statistics(
                self.target_model.actor, batch.observations
            )
            target_entropy = -(target_probabilities * target_logs).sum(1)
        return target_entropy

    def _desired_entropy(self, action_count: int) -> float:
        maximum = math.log(action_count)
        if self.target_entropy is None:
            return 0.98 * maximum
        if self.target_entropy > maximum:
            raise ValueError("SD-SAC target_entropy cannot exceed log(action_count)")
        if self.actor_temperature == "entropy_matched" and self.target_entropy >= maximum:
            # Critic targets ask for this before any optimizer step, so an
            # unreachable matched target fails before weights change.
            raise ValueError("entropy_matched target_entropy must be below log(action_count)")
        return self.target_entropy

    def _entropy_loss(self, actor: _DiscreteActorStep) -> torch.Tensor:
        alpha_loss = torch.zeros((), device=self.device)
        if self.log_alpha is None:
            return alpha_loss
        desired = self._desired_entropy(actor.action_count)
        alpha_loss = (self.log_alpha * (actor.entropy.detach() - desired)).mean()
        assert self.alpha_optimizer is not None
        self._optimize(alpha_loss, self.alpha_optimizer)
        with torch.no_grad():
            if self.entropy_coefficient_min is not None:
                self.log_alpha.clamp_(min=math.log(self.entropy_coefficient_min))
            if self.entropy_coefficient_max is not None:
                self.log_alpha.clamp_(max=math.log(self.entropy_coefficient_max))
        return alpha_loss

    @staticmethod
    def _update_result(
        batch: SACBatch, update: _DiscreteUpdate
    ) -> tuple[Mapping[str, float], PriorityUpdate]:
        metrics = {
            "loss/actor": float(update.actor.loss.item()),
            "loss/critic": float(update.critic.loss.item()),
            "loss/entropy": float(update.alpha_loss.item()),
            "state/alpha": float(update.alpha.item()),
            "state/alpha_used": float(update.alpha_used.item()),
            "policy/entropy": float(update.actor.entropy.detach().mean().item()),
            "critic/q1_mean": float(update.critic.q1.detach().mean().item()),
            "critic/q2_mean": float(update.critic.q2.detach().mean().item()),
            "critic/target_mean": float(update.critic.targets.detach().mean().item()),
            "critic/disagreement": float(
                (update.critic.q1 - update.critic.q2).detach().abs().mean().item()
            ),
            "critic/td_rmse": float(
                (0.5 * (update.critic.q1 + update.critic.q2) - update.critic.targets)
                .detach()
                .square()
                .mean()
                .sqrt()
                .item()
            ),
            "critic/td_bias": float(
                (0.5 * (update.critic.q1 + update.critic.q2) - update.critic.targets)
                .detach()
                .mean()
                .item()
            ),
            **{key: float(value.item()) for key, value in update.actor.diagnostics.items()},
            **{key: float(value.item()) for key, value in update.critic.diagnostics.items()},
        }
        td_errors = (0.5 * (update.critic.q1 + update.critic.q2) - update.critic.targets).abs()
        return metrics, PriorityUpdate(
            batch.source.transition_ids, td_errors.detach().cpu().tolist()
        )

    def policy(self) -> _DiscretePolicy:
        assert self.model is not None
        return _DiscretePolicy(self.model.actor, self.device)

    def state_dict(self) -> Mapping[str, Any]:
        assert self.model is not None
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "scaler": self._scaler_state(),
            "rng": self._rng_state(),
            "sd_sac_options": self._checkpoint_options(),
        }

    def state_dict_for_policy(self, policy_state: Mapping[str, Any]) -> Mapping[str, Any]:
        assert self.model is not None
        state = evaluated_actor_state(self.state_dict(), self.model, policy_state)
        state["actor_optimizer"] = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.actor_learning_rate
        ).state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self._validate_checkpoint_options(state)
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        entropy = EntropyRestoreTarget(self.log_alpha, self.alpha_optimizer, self.device)
        restore_entropy_state(entropy, state)
        self._restore_scaler(state["scaler"])
        self._restore_rng(state["rng"])

    def _checkpoint_options(self) -> Mapping[str, Any]:
        options: dict[str, Any] = {
            "actor_objective": self.actor_objective,
            "learning_rate": self.learning_rate,
            "actor_learning_rate": self.actor_learning_rate,
            "entropy_learning_rate": self.entropy_learning_rate,
            "entropy_coefficient_min": self.entropy_coefficient_min,
            "entropy_coefficient_max": self.entropy_coefficient_max,
            "entropy_mode": self.entropy_mode,
            "initial_entropy_coefficient": self.initial_entropy_coefficient,
            "target_entropy": self.target_entropy,
            "target_tau": self.target_tau,
            "q_clip_epsilon": self.q_clip_epsilon,
            "entropy_penalty_coefficient": self.entropy_penalty_coefficient,
            "entropy_penalty_reference": self.entropy_penalty_reference,
        }
        # Preserve the canonical checkpoint contract at the default.
        if self.terminal_value_loss_coefficient:
            options["terminal_value_loss_coefficient"] = self.terminal_value_loss_coefficient
        if self.actor_temperature != "learned":
            options["actor_temperature"] = self.actor_temperature
        if self.backup_entropy_coefficient is not None:
            options["backup_entropy_coefficient"] = self.backup_entropy_coefficient
        if self.model is not None and any(
            isinstance(module, HypersphericalLinear) for module in self.model.modules()
        ):
            # A saved Adam trajectory without projection is a different optimizer
            # contract. Do not silently resume it under the repaired behavior.
            options["hyperspherical_projection"] = True
        return options

    def _validate_checkpoint_options(self, state: Mapping[str, Any]) -> None:
        for required in ("log_alpha", "alpha_optimizer"):
            if required not in state:
                raise KeyError(required)
        saved = state.get("sd_sac_options")
        if saved is not None:
            if saved != self._checkpoint_options():
                raise ValueError("checkpoint SD-SAC options do not match the learner")
        elif (
            self.actor_objective != "sac"
            or self.terminal_value_loss_coefficient != 0
            or self.entropy_learning_rate != self.learning_rate
            or self.entropy_coefficient_min is not None
            or self.entropy_coefficient_max is not None
            or self.actor_temperature != "learned"
            or self.backup_entropy_coefficient is not None
            or "hyperspherical_projection" in self._checkpoint_options()
        ):
            raise ValueError("legacy checkpoint cannot resume with new SD-SAC options")
        for name, expected_rate in (
            ("actor_optimizer", self.actor_learning_rate),
            ("critic_optimizer", self.learning_rate),
            ("alpha_optimizer", self.entropy_learning_rate),
        ):
            optimizer = state.get(name)
            if name == "alpha_optimizer" and self.entropy_mode == "fixed":
                continue
            if not isinstance(optimizer, Mapping) or not optimizer.get("param_groups"):
                raise ValueError(f"checkpoint is missing {name} parameter groups")
            if any(group.get("lr") != expected_rate for group in optimizer["param_groups"]):
                raise ValueError(f"checkpoint {name} learning rate does not match the learner")
        if self.entropy_mode == "learned":
            log_alpha = state.get("log_alpha")
            if not isinstance(log_alpha, torch.Tensor) or log_alpha.numel() != 1:
                raise ValueError("checkpoint must contain scalar learned log_alpha")
            if not bool(torch.isfinite(log_alpha).all()):
                raise ValueError("checkpoint log_alpha must be finite")
            coefficient = log_alpha.float().exp()
            if not bool(torch.isfinite(coefficient).all()) or not bool(coefficient > 0):
                raise ValueError("checkpoint alpha must be finite and positive in float32")
            for bound, lower in (
                (self.entropy_coefficient_min, True),
                (self.entropy_coefficient_max, False),
            ):
                if bound is not None:
                    limit = log_alpha.new_tensor(math.log(bound))
                    outside = bool(log_alpha < limit) if lower else bool(log_alpha > limit)
                    if outside:
                        raise ValueError("checkpoint log_alpha is outside configured bounds")
