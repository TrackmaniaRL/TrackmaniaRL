"""The team PPO's stalled-incident memory applied before native V5 preparation."""

import numpy as np

from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGraphFeaturePipeline,
)


class NativePersistentPpoRecoveryPipeline(NativeGraphFeaturePipeline):
    """Preserve incident memory while stalled at the road edge, per episode."""

    def reset_episode(self) -> None:
        super().reset_episode()
        self._stalled_memory = 0.0
        self._stalled_deficit = 0.0

    def _recovery_v5(self, values: np.ndarray, nearest: int) -> np.ndarray:
        recovery = super()._recovery_v5(values, nearest)
        clearance = float(self._road_context(values[4:7], nearest)[4])
        if abs(float(values[16])) >= 2.0 or clearance >= 0.15:
            self._stalled_memory = self._stalled_deficit = 0.0
            return recovery
        if recovery[7] > 0.12 and recovery[6] > 0.08:
            self._stalled_memory = max(self._stalled_memory, float(recovery[7]))
            self._stalled_deficit = max(self._stalled_deficit, float(recovery[6]))
        if not self._stalled_memory:
            return recovery
        result = recovery.copy()
        result[7] = max(float(result[7]), self._stalled_memory)
        result[6] = max(float(result[6]), self._stalled_deficit)
        return result
