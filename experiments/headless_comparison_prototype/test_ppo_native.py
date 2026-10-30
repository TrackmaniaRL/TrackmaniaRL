"""Exercise native PPO memory without invoking the engine."""

import numpy as np
import pytest

from experiments.headless_comparison_prototype.ppo_native import NativePersistentPpoRecoveryPipeline
from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5


def test_stalled_memory_is_retained_and_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(NativePersistentPpoRecoveryPipeline)
    pipeline._stalled_memory = pipeline._stalled_deficit = 0.0
    recovery = np.zeros(8, dtype=np.float32)
    monkeypatch.setattr(BoundaryGraphFeaturePipelineV5, "_recovery_v5", lambda *_: recovery.copy())
    monkeypatch.setattr(pipeline, "_road_context", lambda *_: [0, 0, 0, 0, 0.1])
    values = np.zeros(33, dtype=np.float32)
    recovery[6], recovery[7] = 0.3, 0.5
    first = pipeline._recovery_v5(values, 0)
    recovery[:] = 0
    retained = pipeline._recovery_v5(values, 0)
    np.testing.assert_array_equal(first, retained)
    values[16] = 2
    assert not pipeline._recovery_v5(values, 0).any()
    values[16] = 0
    assert not pipeline._recovery_v5(values, 0).any()
