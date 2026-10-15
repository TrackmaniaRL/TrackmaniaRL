from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from tests.unit.trackmania._lidar_fixtures import _asset
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.runtime_validation import validate_resolved_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.project.scaffold_run_templates import _trackmania_sensor_config


class SensorEnvironment:
    def __init__(self, *, fusion: bool) -> None:
        self.fusion = fusion
        self.steps = 0

    def _observation(self) -> Any:
        telemetry = np.zeros(33, dtype=np.float32)
        telemetry[12] = 1
        telemetry[3] = self.steps * 50
        if not self.fusion:
            return telemetry
        return {
            "telemetry": telemetry,
            "images": np.full((8, 8, 3), 40 + self.steps * 10, dtype=np.uint8),
        }

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.steps = 0
        return self._observation(), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        assert np.isfinite(action).all()
        return self._observation(), 1.0, False, self.steps == 3, {}

    def close(self) -> None:
        pass


class SensorEnvironmentFactory:
    def __init__(self, *, fusion: bool) -> None:
        self.fusion = fusion

    def create(self, *, seed: int) -> SensorEnvironment:
        return SensorEnvironment(fusion=self.fusion)


@pytest.mark.parametrize(
    "algorithm", ["q", "qr", "iqn", "fqf", "sac", "redq", "tqc", "sd-sac", "ppo"]
)
@pytest.mark.parametrize("fusion", [False, True])
def test_sensor_update_train_and_resume(tmp_path: Path, algorithm: str, *, fusion: bool) -> None:
    geometry = _asset(tmp_path)
    config = yaml.safe_load(_trackmania_sensor_config(algorithm, fusion=fusion))
    config["artifacts_dir"] = str(tmp_path / "artifacts")
    config.pop("evaluation")
    components = config["components"]
    components.pop("evaluator")
    pipeline = components["feature_pipeline"]["kwargs"]
    lidar = pipeline["lidar" if fusion else "config"]
    lidar.update(geometry_path=str(geometry), expected_map_uid="trackmaniarl-test")
    if fusion:
        pipeline["vision"] = {"width": 8, "height": 8}
    components["environment"] = {
        "class_path": f"{__name__}:SensorEnvironmentFactory",
        "kwargs": {"fusion": fusion},
    }
    components["replay_store"]["kwargs"] = {"capacity": 16}
    learner = components["learner"]["kwargs"]
    learner["execution"] = {"device": "cpu", "precision": "float32"}
    if algorithm == "ppo":
        learner.update(update_epochs=1, minibatch_size=4)
    if algorithm in {"redq", "tqc"}:
        components["model_factory"]["kwargs"]["config"] = {"critic_count": 2}
    config["training"].update(
        total_transitions=8,
        batch_size=1 if algorithm == "ppo" else 2,
        sequence_length=4 if algorithm == "ppo" else 1,
        n_step=1,
        warmup_transitions=2,
        checkpoint_interval_updates=1,
    )
    spec = RunSpec.model_validate(config)
    run = resolve_run(spec)
    try:
        assert all(np.isfinite(v) for v in validate_resolved_run(run).values())
    finally:
        run.logger.close()
    run = resolve_run(spec)
    try:
        result = Trainer(run).train()
        assert result.transitions == 8
        assert result.updates > 0
        gradients = {
            n: p.grad for n, p in run.learner.model.named_parameters() if p.grad is not None
        }
        assert gradients
        assert all(torch.isfinite(g).all() for g in gradients.values())
        if fusion:
            for branch in ("vision.convolution", "lidar.sensor"):
                assert any(branch in n and torch.count_nonzero(g) for n, g in gradients.items())
    finally:
        run.logger.close()
    resumed = resolve_run(spec)
    try:
        continuation = Trainer(resumed, resume_checkpoint=result.checkpoints[0]).train()
        assert continuation.transitions == result.transitions
        assert continuation.updates == result.updates
    finally:
        resumed.logger.close()
