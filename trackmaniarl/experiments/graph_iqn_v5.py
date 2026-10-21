"""Recovery-state and short-memory extension of the V4 graph IQN policy.

V5 adds two initially inert residual paths: explicit frame-to-frame recovery
features and a bounded residual GRU.  A V3/V4 checkpoint therefore produces the
same actions before adaptation, while sequence replay can learn how to recover
from persistent speed loss instead of reacting to one frame in isolation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite
from typing import Any, cast

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.experiments.graph_iqn_v3 import (
    CONTEXT_V3_DIM,
    BoundaryGraphFeaturePipelineV3,
)
from trackmaniarl.experiments.graph_iqn_v4 import TrackGnnSimbaEncoderV4

RECOVERY_V5_LAYOUT: tuple[str, ...] = (
    "decision_interval_error",
    "speed_loss",
    "forward_speed_loss",
    "progress_advance",
    "clearance_loss",
    "edge_approach",
    "retained_speed_deficit",
    "recovery_memory",
)
RECOVERY_V5_DIM = len(RECOVERY_V5_LAYOUT)


@dataclass(frozen=True, slots=True)
class _RecoveryFrame:
    time_ms: float
    speed_mps: float
    forward_mps: float
    nearest: int
    offset: float
    clearance: float


@dataclass(frozen=True, slots=True)
class _RecoverySignals:
    speed_loss: float
    forward_loss: float
    progress: float
    clearance_loss: float
    edge_approach: float
    retained_deficit: float


def _positive_difference(left: float, right: float, scale: float) -> float:
    return max(0.0, left - right) / scale


class BoundaryGraphFeaturePipelineV5(BoundaryGraphFeaturePipelineV3):
    """V3 features plus causal, compact recovery-state indicators."""

    schema_version = "5"

    @staticmethod
    def _build_observation_space() -> spaces.Dict:
        base = BoundaryGraphFeaturePipelineV3._build_observation_space()
        return spaces.Dict(
            {
                "physics": base["physics"],
                "track": base["track"],
                "context": base["context"],
                "recovery": spaces.Box(-4.0, 4.0, (RECOVERY_V5_DIM,), dtype=np.float32),
            }
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        self._recovery_time_ms: float | None = None
        self._recovery_speed_mps = 0.0
        self._recovery_forward_mps = 0.0
        self._recovery_progress_index = 0
        self._recovery_clearance = 1.0
        self._recovery_abs_offset = 0.0
        self._recovery_peak_speed_mps = 0.0
        self._recovery_memory = 0.0

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if isinstance(observation, Mapping):
            return self._validate_prepared_v5(observation)
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        prepared = super().transform_observation(values)
        recovery = self._recovery_v5(values, self._progress_index)
        return {
            "context": prepared["context"],
            "physics": prepared["physics"],
            "recovery": torch.from_numpy(recovery),
            "track": prepared["track"],
        }

    def _validate_prepared_v5(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        required = {"physics", "track", "context", "recovery"}
        if set(observation) != required:
            raise ValueError(
                "prepared V5 observation requires physics, track, context and recovery"
            )
        result = self._prepared_tensors(observation)
        self._validate_v5_tensors(result)
        return result

    @staticmethod
    def _prepared_tensors(observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "context": torch.as_tensor(observation["context"], dtype=torch.float32),
            "physics": torch.as_tensor(observation["physics"], dtype=torch.float32),
            "recovery": torch.as_tensor(observation["recovery"], dtype=torch.float32),
            "track": torch.as_tensor(observation["track"], dtype=torch.float32),
        }

    @staticmethod
    def _validate_v5_tensors(result: Mapping[str, torch.Tensor]) -> None:
        expected = {
            "physics": (60,),
            "track": (3, 88),
            "context": (CONTEXT_V3_DIM,),
            "recovery": (RECOVERY_V5_DIM,),
        }
        if any(result[name].shape != shape for name, shape in expected.items()):
            raise ValueError("prepared V5 observation has invalid shape")
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError("prepared V5 observation contains non-finite values")

    def _recovery_v5(self, values: np.ndarray, nearest: int) -> np.ndarray:
        frame = self._recovery_frame(values, nearest)
        if self._recovery_time_ms is None:
            features, dt_s = np.zeros(RECOVERY_V5_DIM, dtype=np.float32), 0.05
        else:
            dt_s = max((frame.time_ms - self._recovery_time_ms) / 1_000.0, 0.001)
            features = self._recovery_delta(frame, dt_s)
        self._commit_recovery(frame, dt_s)
        return np.clip(features, -4.0, 4.0)

    def _recovery_frame(self, values: np.ndarray, nearest: int) -> _RecoveryFrame:
        heading = values[[10, 12]]
        heading /= max(float(np.linalg.norm(heading)), 1.0e-6)
        forward_mps = float(values[[7, 9]] @ heading)
        road = self._road_context(values[4:7], nearest)
        return _RecoveryFrame(
            float(values[3]),
            float(values[16]),
            forward_mps,
            nearest,
            float(road[0]),
            float(road[4]),
        )

    def _recovery_delta(self, frame: _RecoveryFrame, dt_s: float) -> np.ndarray:
        signals = self._recovery_signals(frame, dt_s)
        incident = max(
            signals.speed_loss,
            signals.forward_loss,
            signals.clearance_loss,
            signals.edge_approach,
        )
        self._update_recovery_memory(incident, dt_s)
        return self._recovery_features(signals, dt_s)

    def _recovery_signals(self, frame: _RecoveryFrame, dt_s: float) -> _RecoverySignals:
        decayed_peak = self._recovery_peak_speed_mps * exp(-dt_s / 2.0)
        return _RecoverySignals(
            _positive_difference(self._recovery_speed_mps, frame.speed_mps, 20.0),
            _positive_difference(self._recovery_forward_mps, frame.forward_mps, 20.0),
            _positive_difference(frame.nearest, self._recovery_progress_index, 5.0),
            _positive_difference(self._recovery_clearance, frame.clearance, 1.0),
            _positive_difference(abs(frame.offset), self._recovery_abs_offset, 0.5),
            _positive_difference(decayed_peak, frame.speed_mps, 30.0),
        )

    def _recovery_features(self, signals: _RecoverySignals, dt_s: float) -> np.ndarray:
        return np.asarray(
            (
                dt_s / 0.05 - 1.0,
                signals.speed_loss,
                signals.forward_loss,
                signals.progress,
                signals.clearance_loss,
                signals.edge_approach,
                signals.retained_deficit,
                self._recovery_memory,
            ),
            dtype=np.float32,
        )

    def _update_recovery_memory(self, incident: float, dt_s: float) -> None:
        decay = exp(-dt_s / 0.75)
        self._recovery_memory = decay * self._recovery_memory + (1.0 - decay) * incident

    def _commit_recovery(self, frame: _RecoveryFrame, dt_s: float) -> None:
        decayed_peak = self._recovery_peak_speed_mps * exp(-dt_s / 2.0)
        self._recovery_peak_speed_mps = max(frame.speed_mps, decayed_peak)
        self._recovery_time_ms = frame.time_ms
        self._recovery_speed_mps = frame.speed_mps
        self._recovery_forward_mps = frame.forward_mps
        self._recovery_progress_index = frame.nearest
        self._recovery_clearance = frame.clearance
        self._recovery_abs_offset = abs(frame.offset)


class TrackGnnSimbaEncoderV5(TrackGnnSimbaEncoderV4):
    """V4 encoder plus an initially inert recovery-state correction."""

    fingerprint_parameters: tuple[str, ...] = (
        *TrackGnnSimbaEncoderV4.fingerprint_parameters,
        "recovery_residual_scale",
    )

    def __init__(
        self,
        context_residual_scale: float = 1.0,
        spatial_residual_scale: float = 1.0,
        recovery_residual_scale: float = 0.02,
    ) -> None:
        super().__init__(context_residual_scale, spatial_residual_scale)
        if not isfinite(recovery_residual_scale) or not 0.0 < recovery_residual_scale <= 1.0:
            raise ValueError("recovery_residual_scale must be finite and in (0, 1]")
        self.recovery_residual_scale = float(recovery_residual_scale)
        self.recovery_adapter = nn.Sequential(
            nn.Linear(RECOVERY_V5_DIM, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, self.output_dim),
        )
        final = cast(nn.Linear, self.recovery_adapter[-1])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        recovery = observation["recovery"].float()
        if recovery.ndim != 2 or recovery.shape[1] != RECOVERY_V5_DIM:
            raise ValueError(f"recovery observation must have shape (batch, {RECOVERY_V5_DIM})")
        if any(
            observation[name].shape[0] != recovery.shape[0]
            for name in ("track", "physics", "context")
        ):
            raise ValueError("V5 observation batch dimensions must match")
        baseline = super().forward(observation)
        correction = self.recovery_residual_scale * torch.tanh(self.recovery_adapter(recovery))
        return baseline + correction


class RecoveryTemporalOnlyDiscreteValueLearner(DiscreteValueLearner):
    """Train only V5 recovery features and residual recurrent memory."""

    def _prepare_model(self) -> None:
        super()._prepare_model()
        assert isinstance(self.model.encoder, TrackGnnSimbaEncoderV5)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.encoder.recovery_adapter.parameters():
            parameter.requires_grad_(True)
        for parameter in self.model.temporal.parameters():
            parameter.requires_grad_(True)
