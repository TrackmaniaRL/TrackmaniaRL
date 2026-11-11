"""Local periodic-evaluation recovery across interruption and resume."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.integration.runtime.core_runtime_support import FakeEnvironment, runtime_spec
from trackmaniarl.core.runtime import ResolvedRun, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer


class _InterruptedError(RuntimeError):
    pass


class _CollectionReachedError(RuntimeError):
    pass


class _Evaluator:
    def __init__(
        self,
        events: list[str],
        *,
        fail: bool = False,
        stop: Path | None = None,
    ) -> None:
        self.events = events
        self.fail = fail
        self.stop = stop
        self.checkpoint: Path | None = None
        self.checkpoint_hash: str | None = None

    def set_checkpoint(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint)

    def evaluate(self, policy: object) -> dict[str, float]:
        del policy
        self.events.append("evaluation")
        assert self.checkpoint is not None
        self.checkpoint_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if self.fail:
            raise _InterruptedError("evaluation interrupted")
        if self.stop is not None:
            self.stop.touch()
        return {"eval/finish_rate": 1.0}


class _Factory:
    def __init__(self, events: list[str], *, fail_on_create: bool = False) -> None:
        self.events = events
        self.fail_on_create = fail_on_create

    def create(self, *, seed: int) -> FakeEnvironment:
        del seed
        self.events.append("collect")
        if self.fail_on_create:
            raise _CollectionReachedError
        return FakeEnvironment()


def _spec(tmp_path: Path, *, warmup: int = 2, keep: int | None = None) -> RunSpec:
    payload = runtime_spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {
        "class_path": "tests.integration.runtime.core_runtime_support:FakeEnvironmentFactory"
    }
    payload["components"]["evaluator"] = {
        "class_path": "tests.integration.runtime.core_runtime_support:RecordingEvaluator"
    }
    payload["training"] = {
        "total_transitions": 10,
        "max_episode_steps": 2,
        "batch_size": 1,
        "warmup_transitions": warmup,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": 1,
        "evaluate_every_episodes": 2,
        "save_final_checkpoint": False,
        "checkpoint_keep_last": keep,
    }
    return RunSpec.model_validate(payload)


def _run(spec: RunSpec, evaluator: _Evaluator, factory: _Factory) -> ResolvedRun:
    return replace(resolve_run(spec), evaluator=evaluator, environment_factory=factory)


def _train(run: ResolvedRun, *, checkpoint: Path | None = None, stop: Path | None = None) -> Any:
    try:
        return Trainer(run, resume_checkpoint=checkpoint, stop_file=stop).train()
    finally:
        run.logger.close()


def _pending_checkpoint(run: ResolvedRun) -> Path:
    candidates = sorted((run.run_dir / "checkpoints").glob("update-*.pt"))
    pending = [
        path
        for path in candidates
        if run.checkpoint_codec.load(path).get("pending_periodic_evaluation") is True
    ]
    assert pending
    return pending[-1]


def _interrupted_checkpoint(spec: RunSpec) -> Path:
    events: list[str] = []
    run = _run(spec, _Evaluator(events, fail=True), _Factory(events))
    with pytest.raises(_InterruptedError, match="evaluation interrupted"):
        _train(run)
    assert events[-1] == "evaluation"
    return _pending_checkpoint(run)


def _completed_checkpoint(spec: RunSpec, stop: Path) -> tuple[Path, _Evaluator]:
    events: list[str] = []
    evaluator = _Evaluator(events, stop=stop)
    run = _run(spec, evaluator, _Factory(events))
    result = _train(run, stop=stop)
    assert result.transitions == 4
    assert events[-1] == "evaluation"
    return _pending_checkpoint(run), evaluator


def test_resume_runs_interrupted_evaluation_before_collecting(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    checkpoint = _interrupted_checkpoint(spec)
    stop = tmp_path / "stop"
    events: list[str] = []
    run = _run(spec, _Evaluator(events, stop=stop), _Factory(events))

    result = _train(run, checkpoint=checkpoint, stop=stop)

    assert events == ["evaluation"]
    assert result.transitions == 4
    assert result.updates == 2


def test_completed_evaluation_is_not_repeated_before_next_collection(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    stop = tmp_path / "stop"
    checkpoint, _ = _completed_checkpoint(spec, stop)
    stop.unlink()
    events: list[str] = []
    run = _run(spec, _Evaluator(events), _Factory(events, fail_on_create=True))

    with pytest.raises(_CollectionReachedError):
        _train(run, checkpoint=checkpoint, stop=stop)

    assert events == ["collect"]


@pytest.mark.parametrize("receipt_state", ["missing", "corrupt", "wrong_hash"])
def test_missing_or_invalid_receipt_retries_evaluation(tmp_path: Path, receipt_state: str) -> None:
    spec = _spec(tmp_path)
    stop = tmp_path / "stop"
    checkpoint, _ = _completed_checkpoint(spec, stop)
    receipt = checkpoint.with_suffix(checkpoint.suffix + ".evaluation.json")
    assert receipt.is_file()
    if receipt_state == "missing":
        receipt.unlink()
    elif receipt_state == "corrupt":
        receipt.write_text("{", encoding="utf-8")
    else:
        receipt.write_text(
            '{"checkpoint_sha256":"wrong","run_fingerprint":"wrong"}', encoding="utf-8"
        )
    stop.unlink()
    events: list[str] = []
    run = _run(spec, _Evaluator(events, stop=stop), _Factory(events))

    result = _train(run, checkpoint=checkpoint, stop=stop)

    assert events == ["evaluation"]
    assert result.transitions == 4


def test_stop_at_resume_preserves_pending_evaluation(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    checkpoint = _interrupted_checkpoint(spec)
    stop = tmp_path / "stop"
    stop.touch()
    events: list[str] = []
    run = _run(spec, _Evaluator(events), _Factory(events))

    _train(run, checkpoint=checkpoint, stop=stop)

    assert events == []
    stop.unlink()
    resumed_events: list[str] = []
    resumed = _run(spec, _Evaluator(resumed_events, stop=stop), _Factory(resumed_events))
    _train(resumed, checkpoint=checkpoint, stop=stop)
    assert resumed_events == ["evaluation"]


def test_evaluation_checkpoint_survives_same_update_and_retention(tmp_path: Path) -> None:
    spec = _spec(tmp_path, warmup=100, keep=1)
    stop = tmp_path / "stop"
    events: list[str] = []
    evaluator = _Evaluator(events)
    factory = _Factory(events)
    original_create = factory.create

    def create(*, seed: int) -> FakeEnvironment:
        environment = original_create(seed=seed)
        if events.count("collect") == 3:
            original_step = environment.step

            def stop_after_episode(
                action: object,
            ) -> tuple[Any, float, bool, bool, dict[str, object]]:
                result = original_step(action)
                if result[2]:
                    stop.touch()
                return result

            environment.step = stop_after_episode  # type: ignore[method-assign]
        return environment

    factory.create = create  # type: ignore[method-assign]
    training = spec.training.model_copy(update={"save_final_checkpoint": True})
    run = _run(spec.model_copy(update={"training": training}), evaluator, factory)

    result = _train(run, stop=stop)

    assert (result.transitions, result.updates) == (6, 0)
    assert evaluator.checkpoint is not None
    assert evaluator.checkpoint_hash is not None
    assert (
        hashlib.sha256(evaluator.checkpoint.read_bytes()).hexdigest() == evaluator.checkpoint_hash
    )
    state = run.checkpoint_codec.load(evaluator.checkpoint)
    assert state["counters"]["transitions"] == 4


def test_each_evaluation_receives_current_checkpoint_without_new_updates(tmp_path: Path) -> None:
    spec = _spec(tmp_path, warmup=100, keep=1)
    training = spec.training.model_copy(update={"evaluate_every_episodes": 1})
    spec = spec.model_copy(update={"training": training})
    stop = tmp_path / "stop"
    events: list[str] = []
    seen: list[int] = []

    class _CapturingEvaluator(_Evaluator):
        def evaluate(self, policy: object) -> dict[str, float]:
            result = super().evaluate(policy)
            assert self.checkpoint is not None
            seen.append(run.checkpoint_codec.load(self.checkpoint)["counters"]["transitions"])
            if len(seen) == 2:
                stop.touch()
            return result

    evaluator = _CapturingEvaluator(events)
    run = _run(spec, evaluator, _Factory(events))

    result = _train(run, stop=stop)

    assert (result.transitions, result.updates) == (4, 0)
    assert seen == [2, 4]


def test_interruption_after_evaluation_before_receipt_retries_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, keep=1)
    events: list[str] = []
    evaluator = _Evaluator(events)
    run = _run(spec, evaluator, _Factory(events))
    original_record = Trainer._record_evaluation

    def interrupt_record(*args: object) -> None:
        del args
        raise _InterruptedError("before completion receipt")

    monkeypatch.setattr(Trainer, "_record_evaluation", interrupt_record)
    with pytest.raises(_InterruptedError, match="before completion receipt"):
        _train(run)
    checkpoint = _pending_checkpoint(run)
    assert events[-1] == "evaluation"
    monkeypatch.setattr(Trainer, "_record_evaluation", original_record)
    stop = tmp_path / "stop"
    resumed_events: list[str] = []
    resumed = _run(spec, _Evaluator(resumed_events, stop=stop), _Factory(resumed_events))

    result = _train(resumed, checkpoint=checkpoint, stop=stop)

    assert resumed_events == ["evaluation"]
    assert result.transitions == 4
    snapshots = list((run.run_dir / "evaluation-checkpoints").glob("snapshot-*.pt"))
    assert len(snapshots) == 1


def test_legacy_checkpoint_without_pending_marker_keeps_previous_resume_behavior(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    stop = tmp_path / "stop"
    checkpoint, _ = _completed_checkpoint(spec, stop)
    run = resolve_run(spec)
    try:
        state = run.checkpoint_codec.load(checkpoint)
        state.pop("pending_periodic_evaluation")
        run.checkpoint_codec.save(state, checkpoint)
    finally:
        run.logger.close()
    stop.unlink()
    events: list[str] = []
    resumed = _run(spec, _Evaluator(events), _Factory(events, fail_on_create=True))

    with pytest.raises(_CollectionReachedError):
        _train(resumed, checkpoint=checkpoint, stop=stop)

    assert events == ["collect"]


def test_final_evaluation_has_current_checkpoint_without_final_checkpoint_setting(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, warmup=100)
    training = spec.training.model_copy(update={"evaluate_every_episodes": None})
    spec = spec.model_copy(update={"training": training})
    events: list[str] = []
    evaluator = _Evaluator(events)
    run = _run(spec, evaluator, _Factory(events))

    result = _train(run)

    assert result.transitions == 10
    assert evaluator.checkpoint is not None
    state = run.checkpoint_codec.load(evaluator.checkpoint)
    assert state["counters"]["transitions"] == 10
    assert (
        hashlib.sha256(evaluator.checkpoint.read_bytes()).hexdigest() == evaluator.checkpoint_hash
    )


def test_ppo_pending_evaluation_runs_before_resumed_rollout(tmp_path: Path) -> None:
    from tests.integration.runtime.core_runtime_support import PpoFakeEnvironment
    from tests.integration.runtime.test_trainer_runtime import _ppo_spec

    payload = _ppo_spec(tmp_path).model_dump(mode="json")
    payload["components"]["evaluator"] = {
        "class_path": "tests.integration.runtime.core_runtime_support:RecordingEvaluator"
    }
    payload["training"].update(
        {
            "evaluate_every_episodes": 1,
            "checkpoint_interval_updates": 1,
            "save_final_checkpoint": False,
        }
    )
    spec = RunSpec.model_validate(payload)
    events: list[str] = []

    class _PpoEnvironment(PpoFakeEnvironment):
        def step(self, action: object) -> Any:
            events.append("step")
            return super().step(action)

    class _PpoFactory:
        def create(self, *, seed: int) -> _PpoEnvironment:
            del seed
            return _PpoEnvironment()

    first = replace(
        resolve_run(spec),
        evaluator=_Evaluator(events, fail=True),
        environment_factory=_PpoFactory(),
    )
    with pytest.raises(_InterruptedError, match="evaluation interrupted"):
        _train(first)
    checkpoint = _pending_checkpoint(first)
    events.clear()
    stop = tmp_path / "stop"
    resumed = replace(
        resolve_run(spec),
        evaluator=_Evaluator(events, stop=stop),
        environment_factory=_PpoFactory(),
    )

    result = _train(resumed, checkpoint=checkpoint, stop=stop)

    assert events == ["evaluation"]
    assert (result.transitions, result.updates) == (2, 1)


def test_transferred_checkpoint_without_snapshot_retries_evaluation(tmp_path: Path) -> None:
    import json
    import shutil

    spec = _spec(tmp_path)
    stop = tmp_path / "stop"
    checkpoint, _ = _completed_checkpoint(spec, stop)
    receipt = checkpoint.with_suffix(checkpoint.suffix + ".evaluation.json")
    snapshot_name = json.loads(receipt.read_text(encoding="utf-8"))["snapshot_name"]
    snapshot = spec.artifacts_dir / spec.run_id / "evaluation-checkpoints" / snapshot_name
    snapshot.unlink()
    transfer = tmp_path / "transferred"
    transfer.mkdir()
    copied_checkpoint = transfer / checkpoint.name
    shutil.copyfile(checkpoint, copied_checkpoint)
    shutil.copyfile(
        receipt, copied_checkpoint.with_suffix(copied_checkpoint.suffix + ".evaluation.json")
    )
    stop.unlink()
    events: list[str] = []
    resumed = _run(spec, _Evaluator(events, stop=stop), _Factory(events))

    result = _train(resumed, checkpoint=copied_checkpoint, stop=stop)

    assert events == ["evaluation"]
    assert result.transitions == 4


def test_required_evaluation_artifact_rejects_receipt_without_artifact_hash(tmp_path: Path) -> None:
    import json

    spec = _spec(tmp_path)
    stop = tmp_path / "stop"
    checkpoint, _ = _completed_checkpoint(spec, stop)
    receipt = checkpoint.with_suffix(checkpoint.suffix + ".evaluation.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["artifact_sha256"] is None
    stop.unlink()
    events: list[str] = []

    class _ArtifactEvaluator(_Evaluator):
        requires_evaluation_artifact = True

    resumed = _run(spec, _ArtifactEvaluator(events, fail=True), _Factory(events))
    with pytest.raises(_InterruptedError, match="evaluation interrupted"):
        _train(resumed, checkpoint=checkpoint, stop=stop)
    assert events == ["evaluation"]


def test_custom_evaluator_without_checkpoint_hook_uses_no_snapshot(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    stop = tmp_path / "stop"
    events: list[str] = []

    class _NoCheckpointEvaluator:
        def evaluate(self, policy: object) -> dict[str, float]:
            del policy
            events.append("evaluation")
            stop.touch()
            return {"eval/finish_rate": 1.0}

    first = replace(
        resolve_run(spec), evaluator=_NoCheckpointEvaluator(), environment_factory=_Factory(events)
    )
    result = _train(first, stop=stop)
    checkpoint = _pending_checkpoint(first)
    assert result.transitions == 4
    assert not (first.run_dir / "evaluation-checkpoints").exists()
    stop.unlink()
    events.clear()
    resumed = replace(
        resolve_run(spec),
        evaluator=_NoCheckpointEvaluator(),
        environment_factory=_Factory(events, fail_on_create=True),
    )
    with pytest.raises(_CollectionReachedError):
        _train(resumed, checkpoint=checkpoint, stop=stop)
    assert events == ["collect"]


class _ArtifactEvaluator(_Evaluator):
    requires_evaluation_artifact = True

    def __init__(self, events: list[str], run_dir: Path, *, stop: Path | None = None) -> None:
        super().__init__(events, stop=stop)
        self.run_dir = run_dir

    def evaluate(self, policy: object) -> dict[str, float]:
        import json

        result = super().evaluate(policy)
        assert self.checkpoint is not None
        assert self.checkpoint_hash is not None
        payload = {
            "status": "complete",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
        }
        (self.run_dir / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")
        return result


@pytest.mark.parametrize("artifact_state", ["missing", "corrupt", "moved"])
def test_builtin_style_artifact_is_required_for_completed_receipt(
    tmp_path: Path, artifact_state: str
) -> None:
    spec = _spec(tmp_path, keep=1)
    stop = tmp_path / "stop"
    events: list[str] = []
    run_dir = spec.artifacts_dir / spec.run_id
    first = _run(spec, _ArtifactEvaluator(events, run_dir, stop=stop), _Factory(events))
    _train(first, stop=stop)
    checkpoint = _pending_checkpoint(first)
    stop.unlink()
    events.clear()
    verified = _run(
        spec, _ArtifactEvaluator(events, run_dir), _Factory(events, fail_on_create=True)
    )
    with pytest.raises(_CollectionReachedError):
        _train(verified, checkpoint=checkpoint, stop=stop)
    assert events == ["collect"]

    artifact = run_dir / "evaluation.json"
    if artifact_state == "missing":
        artifact.unlink()
    elif artifact_state == "corrupt":
        artifact.write_text("{", encoding="utf-8")
    else:
        artifact.rename(tmp_path / "moved-evaluation.json")
    events.clear()
    resumed = _run(spec, _ArtifactEvaluator(events, run_dir, stop=stop), _Factory(events))
    result = _train(resumed, checkpoint=checkpoint, stop=stop)
    assert events == ["evaluation"]
    assert result.transitions == 4


def test_custom_evaluator_retains_existing_builtin_artifact_snapshot(tmp_path: Path) -> None:
    spec = _spec(tmp_path, keep=1)
    stop = tmp_path / "stop"
    events: list[str] = []
    run_dir = spec.artifacts_dir / spec.run_id
    builtin_style = _ArtifactEvaluator(events, run_dir, stop=stop)
    first = _run(spec, builtin_style, _Factory(events))
    _train(first, stop=stop)
    checkpoint = _pending_checkpoint(first)
    assert builtin_style.checkpoint is not None
    original_snapshot = builtin_style.checkpoint
    original_hash = builtin_style.checkpoint_hash
    assert original_hash is not None
    checkpoint.with_suffix(checkpoint.suffix + ".evaluation.json").unlink()
    stop.unlink()
    events.clear()
    custom = _Evaluator(events, stop=stop)
    resumed = _run(spec, custom, _Factory(events))

    _train(resumed, checkpoint=checkpoint, stop=stop)

    assert events == ["evaluation"]
    assert original_snapshot.is_file()
    assert hashlib.sha256(original_snapshot.read_bytes()).hexdigest() == original_hash
    assert custom.checkpoint is not None
    assert custom.checkpoint.is_file()
    assert custom.checkpoint != original_snapshot
