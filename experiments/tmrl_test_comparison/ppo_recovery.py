"""Opt-in PPO-only memory retention while an observed incident remains unresolved."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5


@dataclass
class StalledIncidentMemory:
    memory: float = 0.0
    deficit: float = 0.0

    def apply(self, recovery: np.ndarray, speed_mps: float, clearance: float) -> np.ndarray:
        # Only an already detected incident near a road edge may be retained.
        # Moving or leaving the edge clears it; ordinary driving is unchanged.
        if abs(speed_mps) >= 2.0 or clearance >= 0.15:
            self.memory = self.deficit = 0.0
            return recovery
        if recovery[7] > 0.12 and recovery[6] > 0.08:
            self.memory = max(self.memory, float(recovery[7]))
            self.deficit = max(self.deficit, float(recovery[6]))
        if not self.memory:
            return recovery
        result = recovery.copy()
        result[7] = max(float(result[7]), self.memory)
        result[6] = max(float(result[6]), self.deficit)
        return result


class PersistentPpoRecoveryPipeline(BoundaryGraphFeaturePipelineV5):
    def reset_episode(self) -> None:
        super().reset_episode()
        self._stalled_incident = StalledIncidentMemory()

    def _recovery_v5(self, values: np.ndarray, nearest: int) -> np.ndarray:
        recovery = super()._recovery_v5(values, nearest)
        clearance = float(self._road_context(values[4:7], nearest)[4])
        return self._stalled_incident.apply(recovery, float(values[16]), clearance)

    def transform_observation(self, observation: Any) -> Any:
        # Stored prepared observations are already causal; never update memory
        # again when the learner reads a batch or validate/collate a replay row.
        if isinstance(observation, Mapping):
            return self._validate_prepared_v5(observation)
        return super().transform_observation(observation)
