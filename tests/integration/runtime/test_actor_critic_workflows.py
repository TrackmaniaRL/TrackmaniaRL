from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.integration.trackmania.test_scaffold_evaluation import _evaluation_suite, _patch_geometry
from trackmaniarl.core.contracts import EvaluatorRuntimeRequest, PolicyMode
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.runtime_validation import validate_resolved_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.project.scaffold import create_project
from trackmaniarl.project.scaffold_run_templates import _trackmania_actor_critic_config
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator


class TelemetryEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.steps = 0
        return np.zeros(33, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, int):
            assert 0 <= action < 78
        else:
            assert np.asarray(action).shape == (3,)
            assert np.all(action >= np.array([0, 0, -1]))
            assert np.all(action <= np.ones(3))
        self.steps += 1
        return (
            np.full(33, self.steps / 10.0, dtype=np.float32),
            1.0,
            self.steps == 4,
            False,
            {"termination_reason": "finished" if self.steps == 4 else "", "race_time_ms": 40.0},
        )

    def close(self) -> None:
        pass


class TelemetryEnvironmentFactory:
    def create(self, *, seed: int, evaluation_map: Any = None) -> TelemetryEnvironment:
        return TelemetryEnvironment()


def _spec(algorithm: str, directory: Path) -> RunSpec:
    config = yaml.safe_load(_trackmania_actor_critic_config(algorithm))
    config["artifacts_dir"] = str(directory)
    config.pop("evaluation")
    components = config["components"]
    components.pop("evaluator")
    components["environment"] = {"class_path": f"{__name__}:TelemetryEnvironmentFactory"}
    components["replay_store"]["kwargs"] = {"capacity": 32}
    model_kwargs: dict[str, Any] = {"hidden_dim": 8}
    if algorithm == "redq":
        model_kwargs["critic_count"] = 3
    if algorithm == "tqc":
        model_kwargs = {"config": {"hidden_dim": 8, "critics": 2, "quantiles": 5}}
    components["model_factory"]["kwargs"] = model_kwargs
    config["training"].update(
        total_transitions=16,
        batch_size=2,
        n_step=1,
        warmup_transitions=2,
        updates_per_transition=0.5,
        checkpoint_interval_updates=1,
    )
    return RunSpec.model_validate(config)


@pytest.mark.parametrize("algorithm", ["sac", "redq", "tqc", "sd-sac"])
def test_bundled_actor_critic_collects_updates_resumes_and_evaluates(
    algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_geometry(monkeypatch)
    spec = _spec(algorithm, tmp_path)
    run = resolve_run(spec)
    try:
        result = Trainer(run).train()
        assert result.updates > 1
        state = run.learner.state_dict()
        observation = run.feature_pipeline.transform_observation(np.zeros(33))
        policy_action = run.learner.policy().act(observation, PolicyMode.EVALUATION)
        run.learner.load_state_dict(state)
        np.testing.assert_array_equal(
            run.learner.policy().act(observation, PolicyMode.EVALUATION), policy_action
        )
    finally:
        run.logger.close()
    restored = resolve_run(spec)
    try:
        continued = Trainer(restored, resume_checkpoint=result.checkpoints[0]).train()
        assert continued.transitions == 16
        assert continued.updates == result.updates
        evaluator = TrackmaniaEvaluator(
            EvaluatorRuntimeRequest(
                suite=_evaluation_suite(tmp_path, trials_per_map=2),
                environment_factory=restored.environment_factory,
                feature_pipeline=restored.feature_pipeline,
            )
        )
        metrics = evaluator.evaluate(restored.learner.policy())
        assert metrics["eval/finish_rate"] == 1.0
        assert metrics["eval/finish_time_s"] == pytest.approx(0.04)
    finally:
        restored.logger.close()


@pytest.mark.parametrize("algorithm", ["sac", "redq", "tqc", "sd-sac"])
def test_generated_actor_critic_configuration_validates(algorithm: str, tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent", template="trackmania")
    spec = RunSpec.from_yaml(target / f"run-{algorithm}.yaml")
    spec = spec.model_copy(
        update={"training": spec.training.model_copy(update={"batch_size": 2, "n_step": 1})}
    )
    run = resolve_run(spec, base_dir=target)
    try:
        metrics = validate_resolved_run(run)
        assert all(np.isfinite(value) for value in metrics.values())
        assert "loss/critic" in metrics
    finally:
        run.logger.close()
