from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import BytesValue

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Context,
    _Logger,
    _Pipeline,
    _SlowLearner,
)
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.environment_errors import EnvironmentPausedError
from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_background import _accept_evaluation_request
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig


class AttestedEnvironment:
    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return (
            np.array([1.0], dtype=np.float32),
            1.0,
            True,
            False,
            {
                "termination_reason": "finished",
                "race_time_ms": 2000.0,
            },
        )


class AttestedFactory:
    def validate_scheduled_evaluation_map(self, evaluation_map: Any, pipeline: Any) -> None:
        if (
            evaluation_map.expected_map_uid != "test-map"
            or evaluation_map.reward_points_path.read_text(encoding="utf-8") != "test-map"
            or type(pipeline) is not _Pipeline
        ):
            raise ValueError("unexpected test map or pipeline")

    def create(self, *, seed: int) -> AttestedEnvironment:
        del seed
        return AttestedEnvironment()


class TestLearner(_SlowLearner):
    __test__ = False

    def state_dict_for_policy(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"value": state["value"]}


def _config(tmp_path: Path) -> Path:
    (tmp_path / "reward.txt").write_text("test-map", encoding="utf-8")
    module = "tests.integration.runtime.test_distributed_evaluation_integration"
    config = {
        "api_version": "2.0",
        "run_id": "suite-integration",
        "artifacts_dir": str(tmp_path),
        "components": {
            "environment": {"class_path": f"{module}:AttestedFactory"},
            "evaluator": {"class_path": f"{module}:AttestedFactory"},
            "learner": {"class_path": f"{module}:TestLearner"},
            "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
            "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
            "feature_pipeline": {
                "class_path": "tests.integration.runtime.distributed_runtime_support:_Pipeline"
            },
        },
        "training": {
            "total_transitions": 10,
            "warmup_transitions": 10,
            "evaluate_every_episodes": 1,
            "checkpoint_interval_updates": None,
        },
        "evaluation": {
            "name": "test",
            "version": "1",
            "trials_per_map": 2,
            "maps": [
                {"id": "test", "expected_map_uid": "test-map", "reward_points_path": "reward.txt"}
            ],
        },
    }
    path = tmp_path / "run.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _coordinator(path: Path, fingerprint: str) -> Coordinator:
    spec = RunSpec.from_yaml(path)
    pipeline = _Pipeline()
    run = ResolvedRun(
        spec=spec,
        run_dir=path.parent / spec.run_id,
        learner=TestLearner(),
        environment_factory=AttestedFactory(),
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline, seed=0),
        feature_pipeline=pipeline,
        logger=_Logger(),
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
        base_dir=path.parent,
    )
    return Coordinator(run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, fingerprint))


def _submit(coordinator: Coordinator, payload: bytes) -> dict[str, Any]:
    response = coordinator._submit(
        BytesValue(value=payload), cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}"))
    )
    return coordinator.codec.decode(response.value)


def _close(coordinator: Coordinator) -> None:
    coordinator._checkpoint_writer.close()
    coordinator.journal.close()


def test_attested_single_map_assignment_actor_spool_and_wal_restore(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    actor = ActorRuntime(
        ActorProcessRequest(config_path, "127.0.0.1:1", "test-actor", _DISTRIBUTED_TOKEN)
    )
    coordinator = _coordinator(config_path, actor.fingerprint)
    try:
        pipeline, factory = actor._components()
        coordinator._evaluation_due.add(actor.actor_id)
        coordinator.counters.policy_version = 1
        coordinator._policy_payload = coordinator.codec.encode({"value": 7})
        request = {
            **actor._request_base(),
            "sequence": 0,
            "policy_version": 1,
            "transitions": [],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        assignment = _submit(coordinator, coordinator.codec.encode(request))
        assert assignment["evaluate"] is True
        assert assignment["evaluation_assignment_id"]
        _accept_evaluation_request(actor, assignment)
        actor.sequence = 1
        actor._evaluate(factory.create(seed=0), pipeline)
        payload = actor.queue.get_nowait().read_bytes()
        result = coordinator.codec.decode(payload)
        assert len(result["evaluations"]) == 2
        assert {item["evaluation_assignment_id"] for item in result["evaluations"]} == {
            assignment["evaluation_assignment_id"]
        }
        assert all(item["finished"] for item in result["evaluations"])
        assert coordinator.codec.decode(result["evaluation_snapshot"]) == {"value": 7}
        assert _submit(coordinator, payload)["accepted"] is True
        assert actor.actor_id not in coordinator._evaluation_assignments
    finally:
        actor.client.close()
        _close(coordinator)

    restored = _coordinator(config_path, actor.fingerprint)
    try:
        restored._recover_journal(0)
        restored._checkpoint_writer.wait()
        assert restored.counters.evaluations == 2
        assert restored.counters.evaluation_finishes == 2
        assert restored.counters.journal_applied_frontier == 2
        assert restored._evaluation_policy_states[1] == {"value": 7}
        assert restored._best_evaluation is not None
        assert not restored.journal.pending_evaluation_promotions()
    finally:
        _close(restored)


@pytest.mark.parametrize(
    "stop_point",
    ["before_reset", "after_reset", "between_trials", "during_step", "timeout", "paused"],
)
def test_stopped_evaluation_retries_same_assignment_after_actor_restart(
    tmp_path: Path, stop_point: str
) -> None:
    config_path = _config(tmp_path)
    request = ActorProcessRequest(config_path, "127.0.0.1:1", "test-actor", _DISTRIBUTED_TOKEN)
    actor = ActorRuntime(request)
    coordinator = _coordinator(config_path, actor.fingerprint)
    resumed_actor: ActorRuntime | None = None
    try:
        pipeline, _ = actor._components()
        coordinator._evaluation_due.add(actor.actor_id)
        coordinator.counters.policy_version = 1
        coordinator._policy_payload = coordinator.codec.encode({"value": 7})
        initial = {
            **actor._request_base(),
            "sequence": 0,
            "policy_version": 1,
            "transitions": [],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        assignment = _submit(coordinator, coordinator.codec.encode(initial))
        _accept_evaluation_request(actor, assignment)
        actor.sequence = 1

        class _StopEnvironment(AttestedEnvironment):
            def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
                result = super().reset(seed=seed)
                if stop_point == "after_reset":
                    actor.stop.set()
                return result

            def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
                observation, reward, terminated, truncated, info = super().step(action)
                actor.stop.set()
                if stop_point == "timeout":
                    raise TimeoutError("stop during telemetry wait")
                if stop_point == "paused":
                    raise EnvironmentPausedError("stop during pause")
                if stop_point == "during_step":
                    return observation, reward, False, False, {}
                return observation, reward, terminated, truncated, info

        if stop_point == "before_reset":
            actor.stop.set()
        actor._evaluate(_StopEnvironment(), pipeline)
        assert actor.queue.empty()
        assert actor.actor_id in coordinator._evaluation_assignments

        resumed_actor = ActorRuntime(request)
        resumed_pipeline, resumed_factory = resumed_actor._components()
        retry = {**initial, **resumed_actor._request_base()}
        response = _submit(coordinator, coordinator.codec.encode(retry))
        assert response["evaluate"] is True
        assert response["evaluation_assignment_id"] == assignment["evaluation_assignment_id"]
        assert response["evaluation_snapshot"] == assignment["evaluation_snapshot"]
        _accept_evaluation_request(resumed_actor, response)
        resumed_actor.sequence = 1
        resumed_actor._evaluate(resumed_factory.create(seed=0), resumed_pipeline)
        payload = resumed_actor.queue.get_nowait().read_bytes()
        result = coordinator.codec.decode(payload)
        assert len(result["evaluations"]) == 2
        assert all(item["finished"] for item in result["evaluations"])
        assert _submit(coordinator, payload)["accepted"] is True
        assert actor.actor_id not in coordinator._evaluation_assignments
    finally:
        actor.client.close()
        if resumed_actor is not None:
            resumed_actor.client.close()
        _close(coordinator)

    restored = _coordinator(config_path, actor.fingerprint)
    try:
        restored._recover_journal(0)
        assert restored.counters.evaluations == 2
        assert restored.counters.evaluation_finishes == 2
    finally:
        _close(restored)
