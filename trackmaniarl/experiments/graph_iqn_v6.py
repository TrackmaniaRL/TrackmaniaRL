"""Incident-gated recovery adapter for the V3 TrackMania policy.

V6 keeps the proven V3 policy immutable and permits a learned correction only
while two independent recovery signals agree: retained-speed loss and the
decaying incident memory introduced by V5.  Ordinary observations therefore
bypass the new branch exactly, even after that branch has learned.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch
from torch.nn import functional as F

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.experiments.graph_iqn_v4 import TrackGnnSimbaEncoderV4
from trackmaniarl.experiments.graph_iqn_v5 import (
    RECOVERY_V5_DIM,
    BoundaryGraphFeaturePipelineV5,
    TrackGnnSimbaEncoderV5,
)
from trackmaniarl.models.composite import BatchLayout, CompositeValueModel
from trackmaniarl.models.contracts import ValuePhase

IncidentGateThresholds = tuple[float, float, float, float]
DEFAULT_INCIDENT_GATE_THRESHOLDS: IncidentGateThresholds = (0.12, 0.25, 0.08, 0.40)


def _ramp(value: torch.Tensor, onset: float, full: float) -> torch.Tensor:
    return torch.clamp((value - onset) / (full - onset), min=0.0, max=1.0)


def _recovery_metric_values(
    values: torch.Tensor, actions: torch.Tensor, loss: torch.Tensor
) -> Mapping[str, float]:
    accuracy = (values.detach().argmax(dim=-1) == actions).float().mean()
    return {
        "recovery/loss": float(loss.detach().cpu()),
        "recovery/action_accuracy": float(accuracy.cpu()),
    }


class IncidentGatedTrackGnnSimbaEncoderV6(TrackGnnSimbaEncoderV5):
    """Apply the recovery residual only after a sustained detected incident."""

    fingerprint_parameters: tuple[str, ...] = (
        *TrackGnnSimbaEncoderV5.fingerprint_parameters,
        "incident_memory_onset",
        "incident_memory_full",
        "retained_deficit_onset",
        "retained_deficit_full",
    )

    def __init__(
        self,
        recovery_residual_scale: float = 0.01,
        incident_gate_thresholds: IncidentGateThresholds = DEFAULT_INCIDENT_GATE_THRESHOLDS,
        **encoder_kwargs: float,
    ) -> None:
        super().__init__(recovery_residual_scale=recovery_residual_scale, **encoder_kwargs)
        self._set_incident_gate_thresholds(incident_gate_thresholds)

    def _set_incident_gate_thresholds(self, thresholds: IncidentGateThresholds) -> None:
        self._validate_thresholds(*thresholds)
        (
            incident_memory_onset,
            incident_memory_full,
            retained_deficit_onset,
            retained_deficit_full,
        ) = thresholds
        self.incident_memory_onset = float(incident_memory_onset)
        self.incident_memory_full = float(incident_memory_full)
        self.retained_deficit_onset = float(retained_deficit_onset)
        self.retained_deficit_full = float(retained_deficit_full)

    @staticmethod
    def _validate_thresholds(*values: float) -> None:
        memory_onset, memory_full, deficit_onset, deficit_full = values
        if not all(isfinite(value) for value in values):
            raise ValueError("incident gate thresholds must be finite")
        if not 0.0 <= memory_onset < memory_full <= 4.0:
            raise ValueError("incident memory thresholds must satisfy 0 <= onset < full <= 4")
        if not 0.0 <= deficit_onset < deficit_full <= 4.0:
            raise ValueError("retained deficit thresholds must satisfy 0 <= onset < full <= 4")

    def incident_gate(self, recovery: torch.Tensor) -> torch.Tensor:
        """Return a conservative agreement gate with shape ``(batch, 1)``."""
        if recovery.ndim != 2 or recovery.shape[1] != RECOVERY_V5_DIM:
            raise ValueError(f"recovery observation must have shape (batch, {RECOVERY_V5_DIM})")
        memory = _ramp(recovery[:, 7], self.incident_memory_onset, self.incident_memory_full)
        retained = _ramp(recovery[:, 6], self.retained_deficit_onset, self.retained_deficit_full)
        return torch.minimum(memory, retained).unsqueeze(-1)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        recovery = observation["recovery"].float()
        gate = self.incident_gate(recovery)
        baseline = TrackGnnSimbaEncoderV4.forward(self, observation)
        correction = self.recovery_residual_scale * torch.tanh(self.recovery_adapter(recovery))
        return baseline + gate * correction


class IncidentRecoveryOnlyDiscreteValueLearner(DiscreteValueLearner):
    """Freeze the source policy and optimize only the incident-gated adapter."""

    def _prepare_model(self) -> None:
        super()._prepare_model()
        assert isinstance(self.model.encoder, IncidentGatedTrackGnnSimbaEncoderV6)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.encoder.recovery_adapter.parameters():
            parameter.requires_grad_(True)

    def supervised_recovery_update(
        self,
        observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Mapping[str, float]:
        """Imitate post-incident human controls without a TD update."""
        self._validate_supervised_batch(observations, actions)
        prepared = {name: value.to(self.device) for name, value in observations.items()}
        labels = actions.to(device=self.device, dtype=torch.int64)
        self.model.train()
        values, loss = self._supervised_recovery_loss(prepared, labels)
        self._optimize_recovery_loss(loss)
        self.update_count += 1
        return _recovery_metric_values(values, labels, loss)

    def _validate_supervised_batch(
        self, observations: Mapping[str, torch.Tensor], actions: torch.Tensor
    ) -> None:
        assert isinstance(self.model, CompositeValueModel)
        if actions.ndim != 1:
            raise ValueError("supervised recovery actions must have shape (batch,)")
        if not actions.numel():
            raise ValueError("supervised recovery batch must not be empty")
        if any(value.shape[0] != len(actions) for value in observations.values()):
            raise ValueError("supervised recovery observation batch dimensions must match")
        if torch.any(actions < 0) or torch.any(actions >= self.model.action_count):
            raise ValueError("supervised recovery action is outside the model action space")

    def _supervised_recovery_loss(
        self, observations: Mapping[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self.autocast():
            values = self._supervised_recovery_values(observations)
            loss = F.cross_entropy(values, actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("supervised recovery loss is non-finite")
        return values, loss

    @torch.no_grad()
    def supervised_recovery_metrics(
        self,
        observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Mapping[str, float]:
        """Measure deterministic imitation loss for an episode-held-out batch."""

        prepared = {name: value.to(self.device) for name, value in observations.items()}
        labels = actions.to(device=self.device, dtype=torch.int64)
        self.model.eval()
        with self.autocast():
            values = self._supervised_recovery_values(prepared)
            loss = F.cross_entropy(values, labels)
        return _recovery_metric_values(values, labels, loss)

    @torch.no_grad()
    def supervised_recovery_predictions(
        self, observations: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return deterministic action IDs for recovery-dataset diagnostics."""

        prepared = {name: value.to(self.device) for name, value in observations.items()}
        self.model.eval()
        with self.autocast():
            values = self._supervised_recovery_values(prepared)
        return values.argmax(dim=-1).detach().cpu()

    def _supervised_recovery_values(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        sequence = self.model.encode_sequence(observations, BatchLayout.FRAMES, burn_in=0)
        features = sequence[:, 0]
        support = self.model.support(features, ValuePhase.EVALUATE)
        return self._masked(self.model.expected_all_actions(features, support, self.neutral_risk))

    def _optimize_recovery_loss(self, loss: torch.Tensor) -> None:
        if self.scaler is None:
            raise RuntimeError("learner setup() must be called before recovery fine-tuning")
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def finish_supervised_recovery(self) -> None:
        """Publish the adapted online weights as an exact evaluation target."""

        self.target_model.load_state_dict(self.model.state_dict(), strict=True)
        self.model.eval()


__all__ = [
    "BoundaryGraphFeaturePipelineV5",
    "IncidentGatedTrackGnnSimbaEncoderV6",
    "IncidentRecoveryOnlyDiscreteValueLearner",
]
