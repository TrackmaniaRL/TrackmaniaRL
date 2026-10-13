"""Opt-in PPO epoch transaction; bounds empirical rollout KL, not driving quality."""

from __future__ import annotations

import copy
import math
from typing import Any, cast

import torch

from pilot_tools.ppo_diagnostics import DiagnosticUpdater, InstrumentedPrefixPPO, PpoWandbTracker
from trackmaniarl.algorithms.ppo_support import _index_samples
from trackmaniarl.algorithms.ppo_update import PPOUpdateOwner, PPOUpdater, _TrainingOutcome
from trackmaniarl.core.data import TrainingBatch


class TransactionalPPOUpdater(DiagnosticUpdater):
    norms: list[tuple[float, float, float]]
    epochs: int

    def _full_kl(self, batch: Any) -> float:
        total = 0.0
        count = batch.returns.numel()
        with torch.no_grad(), self.owner.autocast():
            for start in range(0, count, self.owner.minibatch_size):
                indices = torch.arange(start, min(start+self.owner.minibatch_size, count),
                                       device=self.owner.device)
                observation = _index_samples(batch.observations, indices)
                actor = self.owner.model.actor
                distribution = actor._distribution(observation)
                log_ratio = actor._log_probability(distribution, batch.latent_actions[indices])
                log_ratio = log_ratio - batch.old_log_probabilities[indices]
                total += float((log_ratio.exp() - 1 - log_ratio).sum().item())
        return float(total / count)

    def _snapshot(self) -> dict[str, Any]:
        return {"model": copy.deepcopy(self.owner.model.state_dict()),
                "optimizer": copy.deepcopy(self.owner.optimizer.state_dict()),
                "scaler": copy.deepcopy(self.owner.scaler.state_dict()),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.owner.device)
                if self.owner.device.type == "cuda" else None}

    def _restore(self, snapshot: dict[str, Any]) -> None:
        self.owner.model.load_state_dict(snapshot["model"])
        self.owner.optimizer.load_state_dict(snapshot["optimizer"])
        self.owner.scaler.load_state_dict(snapshot["scaler"])
        self.owner.optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(snapshot["cpu_rng"])
        if snapshot["cuda_rng"] is not None:
            torch.cuda.set_rng_state(snapshot["cuda_rng"], self.owner.device)

    def _optimize_epochs(self, batch: Any) -> _TrainingOutcome:
        target = self.owner.target_kl
        if target is None:
            raise ValueError("Transactional PPO requires an explicit target_kl")
        initial = self._full_kl(batch)
        if not math.isfinite(initial) or initial > target:
            raise ValueError("Initial rollout KL exceeds target; reject stale/invalid data")
        totals = torch.zeros(5, device=self.owner.device)
        updates = 0
        self.attempted_epochs = 0
        self.rejected_epochs = 0
        self.rejected_kl = 0.0
        for _ in range(self.owner.update_epochs):
            snapshot = self._snapshot()
            norm_count = len(self.norms)
            epoch_count = self.epochs
            try:
                epoch = self._optimize_epoch(batch)
                self.attempted_epochs += 1
                measured = self._full_kl(batch)
            except BaseException:
                self._restore(snapshot)
                self.norms = self.norms[:norm_count]
                self.epochs = epoch_count
                raise
            if not math.isfinite(measured) or measured > target:
                self._restore(snapshot)
                self.norms = self.norms[:norm_count]
                self.epochs = epoch_count
                self.rejected_epochs += 1
                self.rejected_kl = measured
                break
            totals += epoch.totals
            updates += epoch.updates
        return _TrainingOutcome(totals, updates, bool(self.rejected_epochs))

    @staticmethod
    def _metrics(outcome: _TrainingOutcome) -> dict[str, float]:
        if outcome.updates:
            return dict(PPOUpdater._metrics(outcome))
        return {"loss/policy": 0.0, "loss/value": 0.0, "state/entropy": 0.0,
                "state/approx_kl": 0.0, "state/clip_fraction": 0.0,
                "state/early_stop": float(outcome.stopped_early)}

    def update(self, batch: TrainingBatch) -> tuple[dict[str, float], int]:
        prepared = self._prepare_batch(batch)
        flat = self._flat_batch(prepared)
        outcome = self._optimize_epochs(flat)
        self.owner.observation_normalizer.update(
            prepared.raw_observations, prepared.sample_dimensions
        )
        metrics = self._metrics(outcome)
        metrics.update(self.final_metrics(flat))
        metrics.update({"state/epochs": float(self.epochs),
                        "state/minibatches": float(outcome.updates),
                        "state/attempted_epochs": float(self.attempted_epochs),
                        "state/rejected_epochs": float(self.rejected_epochs),
                        "state/optimization_skipped": float(outcome.updates == 0),
                        "state/rejected_kl_finite": float(math.isfinite(self.rejected_kl)),
                        "state/rejected_kl": self.rejected_kl if math.isfinite(self.rejected_kl)
                        else -1.0})
        for column, name in enumerate(("actor", "value", "total")):
            metrics[f"state/gradient_norm_{name}"] = (
                sum(r[column] for r in self.norms) / len(self.norms) if self.norms else 0.0
            )
        return metrics, prepared.rewards.numel()


class GuardedPrefixPPO(InstrumentedPrefixPPO):
    def update(self, batch: TrainingBatch) -> dict[str, float]:
        prepared = self._batch(batch)
        self._anneal_learning_rate()
        metrics, count = TransactionalPPOUpdater(cast(PPOUpdateOwner, self)).update(prepared)
        self.processed_transitions += count
        return {**metrics, "state/learning_rate": self._current_learning_rate()}


class GuardedPpoWandbTracker(PpoWandbTracker):
    def _project(self, event: str, payload: Any, step: int | None) -> dict[str, Any]:
        values = super()._project(event, payload, step)
        if event == "train/update":
            for name in ("attempted_epochs", "rejected_epochs", "rejected_kl",
                         "optimization_skipped", "rejected_kl_finite"):
                if f"state/{name}" in payload:
                    values[f"learner/ppo_{name}"] = payload[f"state/{name}"]
        return dict(values)
