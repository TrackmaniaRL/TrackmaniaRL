from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from tests.integration.trackmania.test_scaffold_evaluation import _evaluation_suite, _patch_geometry
from trackmaniarl.commands.assets import _trackmania_factory
from trackmaniarl.core.contracts import EvaluatorRuntimeRequest
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.runtime_validation import validate_resolved_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.core.training_loop import _close_session, _start_session, _TrainingSession
from trackmaniarl.project.scaffold import create_project
from trackmaniarl.project.scaffold_run_templates import (
    _trackmania_config,
    _trackmania_ppo_config,
    _trackmania_vision_config,
)
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator


class ImageEnvironment:
    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        self.step_count = 0
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action: Any) -> tuple[np.ndarray[Any, Any], float, bool, bool, dict[str, Any]]:
        if np.asarray(action).ndim == 0:
            assert 0 <= int(action) < 78
        else:
            assert np.asarray(action).shape == (3,)
            assert np.all(np.asarray(action) >= [0, 0, -1])
            assert np.all(np.asarray(action) <= [1, 1, 1])
        self.step_count += 1
        image = np.full((8, 8, 3), self.step_count * 20, dtype=np.uint8)
        return image, 1.0, False, self.step_count == 3, {}

    def close(self) -> None:
        pass


class ImageEnvironmentFactory:
    def create(self, *, seed: int, evaluation_map: Any = None) -> ImageEnvironment:
        return ImageEnvironment()


@pytest.mark.parametrize("algorithm", ["q", "qr", "iqn", "fqf", "sac", "redq", "tqc", "sd-sac"])
def test_off_policy_vision_update_train_and_resume(tmp_path: Path, algorithm: str) -> None:
    config = yaml.safe_load(_trackmania_vision_config(algorithm))
    config["artifacts_dir"] = str(tmp_path)
    config.pop("evaluation")
    components = config["components"]
    components.pop("evaluator")
    components["environment"] = {"class_path": f"{__name__}:ImageEnvironmentFactory"}
    components["feature_pipeline"]["kwargs"] = {"config": {"width": 8, "height": 8}}
    components["replay_store"]["kwargs"] = {"capacity": 16}
    components["learner"]["kwargs"]["execution"] = {"device": "cpu", "precision": "float32"}
    if algorithm in {"sac", "redq", "tqc", "sd-sac"}:
        components["model_factory"]["kwargs"]["config"] = {"hidden_dim": 8, "critic_count": 2}
    config["training"].update(
        total_transitions=8,
        batch_size=2,
        n_step=1,
        warmup_transitions=2,
        updates_per_transition=1.0,
        checkpoint_interval_updates=2,
    )
    spec = RunSpec.model_validate(config)
    run = resolve_run(spec)
    try:
        metrics = validate_resolved_run(run)
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        run.logger.close()
    run = resolve_run(spec)
    try:
        result = Trainer(run).train()
        assert result.transitions == 8
        assert result.updates > 0
        model = run.learner.model
        image_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "convolution" in name and parameter.grad is not None
        ]
        assert image_gradients
        assert all(torch.isfinite(gradient).all() for gradient in image_gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in image_gradients)
    finally:
        run.logger.close()
    resumed = resolve_run(spec)
    try:
        continuation = Trainer(resumed, resume_checkpoint=result.checkpoints[0]).train()
        assert continuation.transitions == result.transitions
        assert continuation.updates == result.updates
    finally:
        resumed.logger.close()


@pytest.mark.parametrize("algorithm", ["q", "qr", "iqn", "fqf"])
def test_vision_value_models_support_recurrent_replay(tmp_path: Path, algorithm: str) -> None:
    config = yaml.safe_load(_trackmania_vision_config(algorithm))
    config["artifacts_dir"] = str(tmp_path)
    config.pop("evaluation")
    components = config["components"]
    components.pop("evaluator")
    components["feature_pipeline"]["kwargs"] = {"config": {"width": 8, "height": 8}}
    components["model_factory"]["kwargs"]["temporal"] = {
        "class_path": "trackmaniarl.models.temporal:GruTemporalCore",
        "kwargs": {"input_dim": 256, "hidden_dim": 256},
    }
    components["learner"]["kwargs"]["execution"] = {"device": "cpu", "precision": "float32"}
    components["learner"]["kwargs"]["burn_in"] = 1
    config["training"].update(batch_size=2, sequence_length=4, n_step=1)
    run = resolve_run(RunSpec.model_validate(config), base_dir=tmp_path)
    try:
        metrics = validate_resolved_run(run)
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        run.logger.close()


def _vision_spec(tmp_path: Path) -> RunSpec:
    config = yaml.safe_load(_trackmania_ppo_config(vision=True))
    config["artifacts_dir"] = str(tmp_path)
    config.pop("evaluation")
    components = config["components"]
    components.pop("evaluator")
    components["environment"] = {"class_path": f"{__name__}:ImageEnvironmentFactory"}
    components["model_factory"]["kwargs"] = {"hidden_dim": 8}
    components["feature_pipeline"]["kwargs"] = {"config": {"width": 8, "height": 8}}
    components["replay_store"]["kwargs"] = {"capacity": 8}
    components["learner"]["kwargs"].update(update_epochs=1, minibatch_size=4)
    config["training"].update(total_transitions=8, sequence_length=4, checkpoint_interval_updates=1)
    return RunSpec.model_validate(config)


def test_vision_ppo_trains_and_resumes_from_a_midrun_checkpoint(tmp_path: Path) -> None:
    spec = _vision_spec(tmp_path)
    run = resolve_run(spec)
    try:
        result = Trainer(run).train()
        final = run.checkpoint_codec.load(result.checkpoints[-1])
    finally:
        run.logger.close()
    assert result.updates == 2
    assert result.transitions == 8
    assert final["learner"]["processed_transitions"] == 8
    resumed = resolve_run(spec)
    try:
        continuation = Trainer(resumed, resume_checkpoint=result.checkpoints[0]).train()
        assert continuation.updates == 2
        assert continuation.transitions == 8
        assert resumed.learner.state_dict()["processed_transitions"] == 8
        observation = resumed.feature_pipeline.transform_observation(
            np.zeros((8, 8, 3), dtype=np.uint8)
        )
        assert np.isfinite(resumed.learner.policy().act(observation)).all()
    finally:
        resumed.logger.close()


def test_vision_ppo_validation_round_trips_model_and_optimizer(tmp_path: Path) -> None:
    run = resolve_run(_vision_spec(tmp_path))
    try:
        metrics = validate_resolved_run(run)
        assert all(np.isfinite(value) for value in metrics.values())
        assert "loss/policy" in metrics
    finally:
        run.logger.close()


def test_restored_vision_ppo_evaluates_every_trial_with_the_trackmania_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_geometry(monkeypatch)
    run = resolve_run(_vision_spec(tmp_path))
    try:
        validate_resolved_run(run)
        evaluator = TrackmaniaEvaluator(
            EvaluatorRuntimeRequest(
                suite=_evaluation_suite(tmp_path, trials_per_map=4),
                environment_factory=run.environment_factory,
                feature_pipeline=run.feature_pipeline,
                run_dir=tmp_path / "evaluation",
            )
        )
        metrics = evaluator.evaluate(run.learner.policy())
        assert metrics["eval/finish_rate"] == 0.0
        assert metrics["eval/reward"] == 3.0
        timeline = tmp_path / "evaluation" / "evaluation-timeline.jsonl"
        events = [json.loads(line) for line in timeline.read_text().splitlines()]
        completed = [event for event in events if event["event"] == "end"]
        assert len(completed) == 4
        assert all(event["result"]["steps"] == 3 for event in completed)
    finally:
        run.logger.close()


def test_vision_iqn_uses_existing_composite_value_training(tmp_path: Path) -> None:
    config = yaml.safe_load(_trackmania_config())
    config["artifacts_dir"] = str(tmp_path)
    config.pop("evaluation")
    components = config["components"]
    components.pop("evaluator")
    components.pop("environment")
    components["feature_pipeline"] = {
        "class_path": "trackmaniarl.trackmania.vision:VisionFeaturePipeline",
        "kwargs": {"config": {"width": 8, "height": 8}},
    }
    components["model_factory"]["kwargs"]["encoder"] = {
        "class_path": "trackmaniarl.trackmania.vision_models:VisionSensorEncoder"
    }
    components["replay_store"]["kwargs"] = {"capacity": 16}
    config["training"].update(batch_size=2, n_step=1)
    run = resolve_run(RunSpec.model_validate(config))
    try:
        metrics = validate_resolved_run(run)
        assert all(np.isfinite(value) for value in metrics.values())
        assert "loss/value" in metrics
    finally:
        run.logger.close()


@pytest.mark.parametrize("vision", [False, True])
def test_generated_ppo_configs_resolve_without_live_capture(
    tmp_path: Path, *, vision: bool
) -> None:
    target = create_project(tmp_path / "project", "agent", template="trackmania")
    path = target / ("run-ppo-vision.yaml" if vision else "run-ppo.yaml")
    run = resolve_run(RunSpec.model_validate(yaml.safe_load(path.read_text())), base_dir=target)
    try:
        assert run.learner.on_policy
        assert run.spec.training.sequence_length == run.replay_store.capacity
    finally:
        run.logger.close()


def test_ppo_rejects_off_policy_sampler_and_insufficient_capacity(tmp_path: Path) -> None:
    config = _vision_spec(tmp_path).model_dump(mode="json")
    config["components"]["sampler"] = {"class_path": "trackmaniarl.core.replay:UniformSampler"}
    config["training"]["sequence_length"] = 1
    with pytest.raises(ValueError, match="OnPolicySequenceSampler"):
        resolve_run(RunSpec.model_validate(config))
    config = _vision_spec(tmp_path).model_dump(mode="json")
    config["components"]["replay_store"]["kwargs"]["capacity"] = 2
    run = resolve_run(RunSpec.model_validate(config))
    try:
        with pytest.raises(ValueError, match="capacity"):
            Trainer(run)
    finally:
        run.logger.close()


def test_camera_config_preserves_reward_discount_validation_and_track_checks(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(_trackmania_ppo_config(vision=True))
    config["artifacts_dir"] = str(tmp_path / "artifacts")
    config["components"]["environment"]["kwargs"]["config"]["reward_gamma"] = 0.5
    with pytest.raises(ValueError, match="reward_gamma"):
        resolve_run(RunSpec.model_validate(config), base_dir=tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config))
    factory = _trackmania_factory(config_path)
    assert factory.config.geometry_path == tmp_path / "assets" / "my-map.geometry.npz"


def test_on_policy_startup_and_close_failures_release_artifact_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    writer = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(
        "trackmaniarl.core.training_loop.AsyncEpisodeWriter", lambda *args, **kwargs: writer
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr("trackmaniarl.core.training_loop._start_on_policy_collection", fail)
    run = resolve_run(_vision_spec(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="camera unavailable"):
            _start_session(Trainer(run))
        assert closed == [True]
        session = _TrainingSession(writer, rollout_environment=SimpleNamespace(close=fail))
        with pytest.raises(RuntimeError, match="camera unavailable"):
            _close_session(session)
        assert closed == [True, True]
    finally:
        run.logger.close()
