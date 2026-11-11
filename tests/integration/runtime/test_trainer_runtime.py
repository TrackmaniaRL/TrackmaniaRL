"""Trainer lifecycle tests for the isolated runtime."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.integration.runtime.core_runtime_support import (
    FakeEnvironment,
    PpoFakeEnvironment,
    runtime_spec,
)
from trackmaniarl.core.builtins import SmokeLearner
from trackmaniarl.core.contracts import Policy
from trackmaniarl.core.runtime import ResolvedRun, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.core.training_support import TrainingResult
from trackmaniarl.trackmania.evaluation import EvaluationCancelledError

_FAKE_ENVIRONMENT = "tests.integration.runtime.core_runtime_support:FakeEnvironmentFactory"
_FAILING_ENVIRONMENT = "tests.integration.runtime.core_runtime_support:FailingEnvironmentFactory"
_PPO_COMPONENTS = {
    "learner": {
        "class_path": (
            "trackmaniarl.algorithms.proximal_policy_optimization:ProximalPolicyOptimization"
        ),
        "kwargs": {"update_epochs": 1, "minibatch_size": 2},
    },
    "environment": {
        "class_path": ("tests.integration.runtime.core_runtime_support:PpoFakeEnvironmentFactory")
    },
    "model_factory": {
        "class_path": "trackmaniarl.trackmania.baseline:TelemetryPpoModelFactory",
        "kwargs": {"input_dim": 33, "hidden_dim": 8},
    },
    "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
    "sampler": {"class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler"},
    "feature_pipeline": {"class_path": "trackmaniarl.trackmania.features:TelemetryFeaturePipeline"},
}
_PPO_TRAINING = {
    "total_transitions": 4,
    "max_episode_steps": 2,
    "batch_size": 2,
    "sequence_length": 2,
    "checkpoint_interval_updates": None,
}
_CHECKPOINT_TRAINING = {
    "total_transitions": 8,
    "max_episode_steps": 2,
    "batch_size": 4,
    "warmup_transitions": 4,
    "updates_per_transition": 1.0,
    "checkpoint_interval_updates": 2,
}


def _ppo_spec(tmp_path: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "ppo-train",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": _PPO_COMPONENTS,
            "training": _PPO_TRAINING,
        }
    )


def _trainer_spec(
    tmp_path: Path, training: Mapping[str, object], environment_class: str
) -> RunSpec:
    payload = runtime_spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {"class_path": environment_class}
    payload["training"] = dict(training)
    return RunSpec.model_validate(payload)


def _train_and_close(run: ResolvedRun, resume_checkpoint: Path | None = None) -> TrainingResult:
    try:
        return Trainer(run, resume_checkpoint=resume_checkpoint).train()
    finally:
        run.logger.close()


def _event_payloads(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _event_names(run_dir: Path) -> list[str]:
    return [str(event["event"]) for event in _event_payloads(run_dir)]


class _InterruptingFactory:
    def __init__(self) -> None:
        self.created = 0

    def create(self, *, seed: int) -> FakeEnvironment:
        del seed
        self.created += 1
        if self.created > 2:
            raise KeyboardInterrupt
        return FakeEnvironment()


class _ManifestAwareSmokeLearner(SmokeLearner):
    def __init__(self) -> None:
        super().__init__()
        self.manifest_present_at_setup = False

    def setup(self, context: Mapping[str, Any]) -> None:
        self.manifest_present_at_setup = (Path(context["run_dir"]) / "manifest.json").is_file()
        super().setup(context)


def _assert_interrupted_state(state: Mapping[str, Any]) -> None:
    assert state["counters"]["transitions"] == 4
    assert state["counters"]["episodes"] == 2


def test_local_trainer_updates_ppo_once_per_fresh_episode(tmp_path: Path) -> None:
    result = _train_and_close(resolve_run(_ppo_spec(tmp_path)))
    assert result.episodes == 2
    assert result.updates == 2


def test_ppo_logs_segments_without_presenting_them_as_single_races(tmp_path: Path) -> None:
    run = resolve_run(_ppo_spec(tmp_path))
    result = _train_and_close(run)
    events = _event_payloads(run.run_dir)
    rollouts = [event["payload"] for event in events if event["event"] == "train/rollout"]
    assert len(rollouts) == result.updates == 2
    assert all(item["completed_episodes"] == 1 for item in rollouts)
    assert all(item["rollout_boundaries"] == 0 for item in rollouts)
    assert "train/episode" not in [event["event"] for event in events]


def test_ppo_stop_file_finishes_rollout_and_saves_checkpoint(tmp_path: Path) -> None:
    run = resolve_run(_ppo_spec(tmp_path))
    stop = tmp_path / "STOP"
    original_update = run.learner.update

    def update(batch: Any) -> Any:
        result = original_update(batch)
        stop.touch()
        return result

    run.learner.update = update
    try:
        result = Trainer(run, stop_file=stop).train()
        assert result.transitions == 2
        assert result.updates == 1
        assert result.checkpoints[-1].is_file()
        state = run.checkpoint_codec.load(result.checkpoints[-1])
        assert state["counters"]["transitions"] == 2
        assert "train/stopped" in _event_names(run.run_dir)
    finally:
        run.logger.close()


class _StopDuringEvaluation:
    def __init__(self, stop_file: Path) -> None:
        self.stop_file = stop_file
        self.stop_requested: Callable[[], bool] = lambda: False
        self.checkpoint: Path | None = None
        self.checkpoint_bytes: bytes | None = None

    def set_stop_requested(self, stop_requested: Callable[[], bool]) -> None:
        self.stop_requested = stop_requested

    def set_checkpoint(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def evaluate(self, policy: Policy) -> Mapping[str, float]:
        del policy
        assert self.checkpoint is not None
        self.checkpoint_bytes = self.checkpoint.read_bytes()
        assert not self.stop_requested()
        self.stop_file.touch()
        if self.stop_requested():
            raise EvaluationCancelledError("Evaluation stopped; the suite is incomplete")
        return {"eval/finish_rate": 1.0}


@pytest.mark.parametrize("evaluation_interval", [None, 1], ids=["final", "periodic"])
def test_ppo_stop_during_evaluation_preserves_checkpoint(
    tmp_path: Path, evaluation_interval: int | None
) -> None:
    spec = _ppo_spec(tmp_path)
    training = spec.training.model_copy(update={"evaluate_every_episodes": evaluation_interval})
    stop = tmp_path / "STOP"
    evaluator = _StopDuringEvaluation(stop)
    run = replace(resolve_run(spec.model_copy(update={"training": training})), evaluator=evaluator)
    try:
        with pytest.raises(EvaluationCancelledError, match="suite is incomplete"):
            Trainer(run, stop_file=stop).train()
        assert evaluator.checkpoint is not None
        assert evaluator.checkpoint.read_bytes() == evaluator.checkpoint_bytes
        state = run.checkpoint_codec.load(evaluator.checkpoint)
        expected_updates = 2 if evaluation_interval is None else 1
        assert state["counters"]["updates"] == expected_updates
        assert state["counters"]["transitions"] == expected_updates * 2
        assert "eval/suite" not in _event_names(run.run_dir)
    finally:
        run.logger.close()


def test_trainer_collects_updates_and_checkpoints(tmp_path: Path) -> None:
    spec = _trainer_spec(tmp_path, _CHECKPOINT_TRAINING, _FAKE_ENVIRONMENT)
    learner = _ManifestAwareSmokeLearner()
    run = replace(resolve_run(spec), learner=learner)
    result = _train_and_close(run)
    assert learner.manifest_present_at_setup
    assert (result.transitions, result.updates) == (8, 4)
    assert len(result.checkpoints) == len(set(result.checkpoints))
    names = _event_names(run.run_dir)
    assert names.count("train/checkpoint_completed") >= len(result.checkpoints)
    assert "train/checkpoint_failed" not in names
    resumed_learner = _ManifestAwareSmokeLearner()
    resumed_run = replace(resolve_run(spec), learner=resumed_learner)
    resumed = _train_and_close(resumed_run, result.checkpoints[-1])
    assert resumed_learner.manifest_present_at_setup
    assert (resumed.transitions, resumed.updates) == (result.transitions, result.updates)


def test_local_checkpoint_retention_runs_only_after_successful_saves(tmp_path: Path) -> None:
    training = {
        **_CHECKPOINT_TRAINING,
        "checkpoint_interval_updates": 1,
        "checkpoint_keep_last": 1,
    }
    run = resolve_run(_trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT))

    result = _train_and_close(run)

    files = sorted((run.run_dir / "checkpoints").glob("update-*.pt"))
    assert files == [run.run_dir / "checkpoints" / "update-00000004.pt"]
    assert result.checkpoints == tuple(files)
    retention = _retention_events(run.run_dir)
    assert retention
    assert retention[-1]["payload"]["removed_count"] == 1


def test_local_milestones_trigger_without_periodic_saves_and_survive_retention(
    tmp_path: Path,
) -> None:
    training = {
        **_CHECKPOINT_TRAINING,
        "checkpoint_interval_updates": None,
        "checkpoint_keep_last": 1,
        "checkpoint_milestones": [5, 7],
    }
    run = resolve_run(_trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT))
    result = _train_and_close(run)
    milestones = sorted((run.run_dir / "checkpoints").glob("milestone-*.pt"))
    assert len(milestones) == 2
    assert result.checkpoints[-1].is_file()
    assert _event_names(run.run_dir).count("train/checkpoint_milestone") == 2


def _retention_events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        event
        for event in _event_payloads(run_dir)
        if event["event"] == "train/checkpoint_retention"
    ]


def test_trainer_checkpoints_latest_completed_episode_on_interrupt(tmp_path: Path) -> None:
    training = {
        "total_transitions": 10,
        "max_episode_steps": 2,
        "batch_size": 4,
        "warmup_transitions": 4,
        "updates_per_transition": 0.25,
        "checkpoint_interval_updates": 100,
    }
    resolved = resolve_run(_trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT))
    run = replace(resolved, environment_factory=_InterruptingFactory())
    checkpoint = run.run_dir / "checkpoints" / "update-00000000.pt"
    try:
        with pytest.raises(KeyboardInterrupt):
            Trainer(run).train()
        state = run.checkpoint_codec.load(checkpoint)
    finally:
        run.logger.close()
    _assert_interrupted_state(state)


def test_trainer_emits_run_failure_event(tmp_path: Path) -> None:
    training = {
        "total_transitions": 2,
        "max_episode_steps": 2,
        "batch_size": 1,
        "warmup_transitions": 1,
    }
    run = resolve_run(_trainer_spec(tmp_path, training, _FAILING_ENVIRONMENT))
    try:
        with pytest.raises(RuntimeError, match="simulated environment failure"):
            Trainer(run).train()
    finally:
        run.logger.close()
    events = _event_payloads(run.run_dir)
    failure = next(item for item in events if item["event"] == "run/failure")
    assert sum(item["event"] == "run/failure" for item in events) == 1
    assert failure["payload"]["exception_type"] == "RuntimeError"
    assert failure["payload"]["message"] == "simulated environment failure"


class _InterruptAfterFirstStep(FakeEnvironment):
    def step(self, action: object) -> tuple[dict[str, float], float, bool, bool, dict[str, object]]:
        if self.step_index == 1:
            raise KeyboardInterrupt
        return super().step(action)


class _InterruptAfterFirstStepFactory:
    def __init__(self) -> None:
        self.created = 0

    def create(self, *, seed: int) -> FakeEnvironment:
        del seed
        self.created += 1
        return FakeEnvironment() if self.created == 1 else _InterruptAfterFirstStep()


def test_local_resume_after_interrupt_during_episode(tmp_path: Path) -> None:
    training = {
        "total_transitions": 4,
        "max_episode_steps": 2,
        "batch_size": 1,
        "warmup_transitions": 1,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": 1,
    }
    spec = _trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT)
    resolved = resolve_run(spec)
    interrupted = replace(resolved, environment_factory=_InterruptAfterFirstStepFactory())
    checkpoint = interrupted.run_dir / "checkpoints" / "update-00000001.pt"
    try:
        with pytest.raises(KeyboardInterrupt):
            Trainer(interrupted).train()
    finally:
        interrupted.logger.close()
    resumed = resolve_run(spec)
    try:
        result = Trainer(resumed, resume_checkpoint=checkpoint).train()
    finally:
        resumed.logger.close()
    assert result.transitions == 4


def test_local_interrupt_before_first_checkpoint_reports_no_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _trainer_spec(tmp_path, _CHECKPOINT_TRAINING, _FAKE_ENVIRONMENT)
    resolved = resolve_run(spec)
    factory = _InterruptAfterFirstStepFactory()
    factory.created = 1
    interrupted = replace(resolved, environment_factory=factory)
    try:
        with pytest.raises(KeyboardInterrupt):
            Trainer(interrupted).train()
    finally:
        interrupted.logger.close()
    assert not list((interrupted.run_dir / "checkpoints").glob("*.pt"))
    output = capsys.readouterr().out
    assert "no new checkpoint was saved" in output
    assert "No previous checkpoint is known" in output


def test_resume_consumes_update_credit_after_final_collection(tmp_path: Path) -> None:
    training = {
        "total_transitions": 2,
        "max_episode_steps": 2,
        "batch_size": 1,
        "warmup_transitions": 1,
        "updates_per_transition": 3.0,
        "checkpoint_interval_updates": 1,
    }
    spec = _trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT)
    run = resolve_run(spec)
    try:
        original = Trainer(run).train()
    finally:
        run.logger.close()
    checkpoint = run.run_dir / "checkpoints" / "update-00000001.pt"
    stop_file = tmp_path / "stop"
    stop_file.touch()
    stopped = resolve_run(spec)
    try:
        stopped_result = Trainer(stopped, resume_checkpoint=checkpoint, stop_file=stop_file).train()
    finally:
        stopped.logger.close()
    assert stopped_result.updates == 1
    resumed = resolve_run(spec)
    try:
        result = Trainer(resumed, resume_checkpoint=checkpoint).train()
    finally:
        resumed.logger.close()
    assert original.updates == 3
    assert result.updates == original.updates


def test_resume_consumes_credit_before_collecting_again(tmp_path: Path) -> None:
    training = {
        "total_transitions": 4,
        "max_episode_steps": 2,
        "batch_size": 1,
        "warmup_transitions": 1,
        "updates_per_transition": 3.0,
        "checkpoint_interval_updates": 1,
    }
    spec = _trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT)
    run = resolve_run(spec)
    _train_and_close(run)
    checkpoint = run.run_dir / "checkpoints" / "update-00000001.pt"
    resumed = resolve_run(spec)
    seen_updates: list[int] = []

    class _RecordingFactory:
        def create(self, *, seed: int) -> FakeEnvironment:
            del seed
            seen_updates.append(int(resumed.learner.state_dict()["updates"]))
            return FakeEnvironment()

    resumed = replace(resumed, environment_factory=_RecordingFactory())
    _train_and_close(resumed, checkpoint)
    assert seen_updates[0] == 3


class _InterruptPpoEnvironment(PpoFakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.total_steps = 0

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, object]]:
        if self.total_steps == 3:
            raise KeyboardInterrupt
        self.total_steps += 1
        return super().step(action)


class _InterruptPpoFactory:
    def create(self, *, seed: int) -> _InterruptPpoEnvironment:
        del seed
        return _InterruptPpoEnvironment()


def test_on_policy_resume_uses_last_complete_checkpoint_after_interrupt(tmp_path: Path) -> None:
    payload = _ppo_spec(tmp_path).model_dump(mode="json")
    payload["training"]["checkpoint_interval_updates"] = 1
    spec = RunSpec.model_validate(payload)
    resolved = resolve_run(spec)
    interrupted = replace(resolved, environment_factory=_InterruptPpoFactory())
    checkpoint = interrupted.run_dir / "checkpoints" / "update-00000001.pt"
    try:
        with pytest.raises(KeyboardInterrupt):
            Trainer(interrupted).train()
    finally:
        interrupted.logger.close()
    state = interrupted.checkpoint_codec.load(checkpoint)
    assert state["counters"]["transitions"] == 2
    resumed = resolve_run(spec)
    try:
        result = Trainer(resumed, resume_checkpoint=checkpoint).train()
    finally:
        resumed.logger.close()
    assert result.transitions == 4


def test_on_policy_resume_rejects_pending_updates_without_rollout(tmp_path: Path) -> None:
    spec = _ppo_spec(tmp_path)
    run = resolve_run(spec)
    result = _train_and_close(run)
    checkpoint = result.checkpoints[-1]
    state = dict(run.checkpoint_codec.load(checkpoint))
    state["counters"]["fractional_updates"] = 1.0
    run.checkpoint_codec.save(state, checkpoint)
    resumed = resolve_run(spec)
    try:
        with pytest.raises(ValueError, match="pending updates"):
            Trainer(resumed, resume_checkpoint=checkpoint).train()
    finally:
        resumed.logger.close()


@pytest.mark.parametrize("on_policy", [False, True])
def test_interrupt_during_update_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, on_policy: bool
) -> None:
    if on_policy:
        payload = _ppo_spec(tmp_path).model_dump(mode="json")
        payload["training"]["checkpoint_interval_updates"] = 1
        spec = RunSpec.model_validate(payload)
    else:
        training = {
            "total_transitions": 4,
            "max_episode_steps": 2,
            "batch_size": 1,
            "warmup_transitions": 1,
            "updates_per_transition": 1.0,
            "checkpoint_interval_updates": 1,
        }
        spec = _trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT)
    run = resolve_run(spec)
    update = run.learner.update
    calls = 0

    def interrupted_update(batch: Any) -> Any:
        nonlocal calls
        result = update(batch)
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(run.learner, "update", interrupted_update)
    checkpoint = run.run_dir / "checkpoints" / "update-00000001.pt"
    try:
        with pytest.raises(KeyboardInterrupt):
            Trainer(run).train()
    finally:
        run.logger.close()
    state = run.checkpoint_codec.load(checkpoint)
    assert state["counters"]["updates"] == 1
    assert state["counters"]["transitions"] == 2
    assert _train_and_close(resolve_run(spec), checkpoint).transitions == 4
