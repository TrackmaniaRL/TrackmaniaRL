"""PPO-only observability adapter; preserve optimization and random draws."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from pilot_tools.ppo_prefix import CampaignPrefixPPO
from trackmaniarl.algorithms.ppo_update import PPOUpdateOwner, PPOUpdater
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.observability.trackers import WandbTracker


def gradient_norm(module: Any) -> float:
    gradients = [p.grad.detach().float().norm().square() for p in module.parameters()
                 if p.grad is not None]
    return float(torch.stack(gradients).sum().sqrt().item()) if gradients else 0.0


class DiagnosticUpdater(PPOUpdater):
    def __init__(self, owner: PPOUpdateOwner) -> None:
        super().__init__(owner)
        self.epochs = 0
        self.norms: list[tuple[float, float, float]] = []

    def _optimize_epoch(self, batch: Any) -> Any:
        self.epochs += 1
        return super()._optimize_epoch(batch)

    def _optimize(self, loss: torch.Tensor) -> None:
        owner = self.owner
        owner.optimizer.zero_grad(set_to_none=True)
        owner.scaler.scale(loss).backward()
        owner.scaler.unscale_(owner.optimizer)
        actor = gradient_norm(owner.model.actor)
        value = gradient_norm(owner.model.value)
        total = torch.nn.utils.clip_grad_norm_(owner.model.parameters(), owner.max_gradient_norm)
        self.norms.append((actor, value, float(total.item())))
        owner.scaler.step(owner.optimizer)
        owner.scaler.update()

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], int]:
        prepared = self._prepare_batch(batch)
        flat = self._flat_batch(prepared)
        outcome = self._optimize_epochs(flat)
        self.owner.observation_normalizer.update(
            prepared.raw_observations, prepared.sample_dimensions
        )
        metrics = dict(self._metrics(outcome))
        metrics.update(self.final_metrics(flat))
        metrics["state/epochs"] = float(self.epochs)
        metrics["state/minibatches"] = float(outcome.updates)
        for column, name in enumerate(("actor", "value", "total")):
            metrics[f"state/gradient_norm_{name}"] = sum(r[column] for r in self.norms) / len(
                self.norms
            )
        return metrics, prepared.rewards.numel()

    def final_metrics(self, batch: Any) -> dict[str, float]:
        # No evaluate_latent_actions call: its entropy sample would advance RNG.
        with torch.no_grad(), self.owner.autocast():
            actor = self.owner.model.actor
            distribution = actor._distribution(batch.observations)
            logs = actor._log_probability(distribution, batch.latent_actions)
            log_ratio = logs - batch.old_log_probabilities
            values = self.owner.model.value(batch.observations)
            target_variance = batch.returns.var(unbiased=False)
            valid = bool(target_variance > 1e-12)
            explained = 1 - (batch.returns - values).var(unbiased=False) / target_variance
            return {
                "state/final_kl": float(((log_ratio.exp() - 1) - log_ratio).mean().item()),
                "state/explained_variance": float(explained.item()) if valid else 0.0,
                "state/explained_variance_valid": float(valid),
                "state/return_mean": float(batch.returns.mean().item()),
                "state/return_std": float(target_variance.sqrt().item()),
                "state/value_mean": float(values.mean().item()),
                "state/value_std": float(values.std(unbiased=False).item()),
            }


class InstrumentedPrefixPPO(CampaignPrefixPPO):
    def update(self, batch: TrainingBatch) -> Mapping[str, float]:
        batch = self._batch(batch)
        self._anneal_learning_rate()
        metrics, count = DiagnosticUpdater(cast(PPOUpdateOwner, self)).update(batch)
        self.processed_transitions += count
        return {**metrics, "state/learning_rate": self._current_learning_rate()}


PPO_METRICS = frozenset({
    "approx_kl", "clip_fraction", "early_stop", "entropy", "learning_rate", "epochs",
    "minibatches", "final_kl", "explained_variance", "explained_variance_valid",
    "return_mean", "return_std", "value_mean", "value_std", "gradient_norm_actor",
    "gradient_norm_value", "gradient_norm_total",
})


class PpoWandbTracker(WandbTracker):
    def _project(self, event: str, payload: Mapping[str, Any], step: int | None) -> dict[str, Any]:
        values = super()._project(event, payload, step)
        if event == "train/update":
            values.update({f"learner/ppo_{key}": payload[f"state/{key}"] for key in PPO_METRICS
                           if f"state/{key}" in payload})
        return values
