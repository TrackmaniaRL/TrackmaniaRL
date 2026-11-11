from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Logger,
    _Pipeline,
)
from tests.integration.runtime.test_distributed_evaluation_integration import (
    AttestedEnvironment,
    AttestedFactory,
    TestLearner,
    _config,
)
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.trackmania.reward_points import save_reward_points


class SlowEnvironment(AttestedEnvironment):
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        time.sleep(0.02)
        return super().step(action)


class SlowFactory(AttestedFactory):
    def create(self, *, seed: int) -> SlowEnvironment:
        del seed
        return SlowEnvironment()

    def validate_scheduled_evaluation_map(self, evaluation_map: Any, pipeline: Any) -> None:
        assert evaluation_map.expected_map_uid == "test-map"
        assert evaluation_map.reward_points_path.is_file()
        assert type(pipeline) is _Pipeline


def _coordinator(path: Path, fingerprint: str, checkpoint: Path | None = None) -> Coordinator:
    spec = RunSpec.from_yaml(path)
    pipeline = _Pipeline()
    run = ResolvedRun(
        spec=spec,
        run_dir=path.parent / spec.run_id,
        learner=TestLearner(),
        environment_factory=SlowFactory(),
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline, seed=0),
        feature_pipeline=pipeline,
        logger=_Logger(),
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
        base_dir=path.parent,
    )
    return Coordinator(
        run,
        CoordinatorConfig(
            "127.0.0.1:0", _DISTRIBUTED_TOKEN, fingerprint, resume_checkpoint=checkpoint
        ),
    )


def _wait_until(predicate: Any, threads: tuple[threading.Thread, ...], timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if not all(thread.is_alive() for thread in threads):
            raise AssertionError("runtime stopped before reaching the expected state")
        if time.monotonic() >= deadline:
            raise AssertionError("runtime did not reach the expected state before timeout")
        time.sleep(0.02)


def test_scheduled_evaluation_over_grpc_survives_full_resume(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reward_path = tmp_path / "reward.npz"
    save_reward_points(
        reward_path, np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32), map_uid="test-map"
    )
    config["evaluation"]["maps"][0]["reward_points_path"] = str(reward_path)
    config["components"]["environment"]["class_path"] = (
        "tests.integration.runtime.test_scheduled_evaluation_grpc_resume:SlowFactory"
    )
    config["training"].update(
        total_transitions=10000,
        warmup_transitions=1,
        batch_size=1,
        updates_per_transition=0.25,
        checkpoint_interval_updates=None,
    )
    config["distributed"] = {
        "rollout_chunk_transitions": 1,
        "rollout_flush_s": 0.05,
        "policy_refresh_s": 0.05,
        "heartbeat_s": 0.1,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    actor = ActorRuntime(
        ActorProcessRequest(config_path, "127.0.0.1:1", "grpc-actor", _DISTRIBUTED_TOKEN)
    )
    coordinator = _coordinator(config_path, actor.fingerprint)
    _run_until_evaluations(coordinator, actor, 2)
    previous_evaluations = coordinator.counters.evaluations
    previous_finishes = coordinator.counters.evaluation_finishes
    previous_transitions = coordinator.counters.transitions
    checkpoints = sorted(
        (tmp_path / "suite-integration" / "checkpoints").glob("distributed-update-*.pt")
    )
    assert checkpoints

    resumed_actor = ActorRuntime(
        ActorProcessRequest(config_path, "127.0.0.1:1", "grpc-actor", _DISTRIBUTED_TOKEN)
    )
    resumed = _coordinator(config_path, resumed_actor.fingerprint, checkpoints[-1])
    _run_until_evaluations(resumed, resumed_actor, previous_finishes + 2)
    assert resumed.counters.evaluation_finishes >= previous_finishes + 2
    assert resumed.counters.evaluations >= previous_evaluations + 2
    assert resumed.counters.transitions > previous_transitions


def _run_until_evaluations(coordinator: Coordinator, actor: ActorRuntime, target: int) -> None:
    stop = threading.Event()
    coordinator.external_stop = stop
    errors: list[BaseException] = []

    def run_coordinator() -> None:
        try:
            coordinator.run_forever()
        except BaseException as exc:
            errors.append(exc)

    def run_actor() -> None:
        try:
            actor.run_forever()
        except BaseException as exc:
            errors.append(exc)

    server = threading.Thread(target=run_coordinator, daemon=True)
    server.start()
    actor_thread: threading.Thread | None = None
    try:
        _wait_until(lambda: coordinator._bound_port is not None or bool(errors), (server,))
        if errors:
            raise errors[0]
        if coordinator.resume_checkpoint is not None:
            assert coordinator.counters.evaluations >= 2
            assert coordinator.counters.transitions > 0
            assert len(coordinator.run.replay_store) > 0
        # The actor's client must connect to the actual bound gRPC endpoint.
        actor.client.close()
        from trackmaniarl.distributed.actor_transport import Client

        actor.client = Client(
            f"127.0.0.1:{coordinator.bound_port}", _DISTRIBUTED_TOKEN, actor.codec
        )
        actor_thread = threading.Thread(target=run_actor, daemon=True)
        actor_thread.start()
        _wait_until(
            # Actual DNFs count as evaluations without counting as finishes.
            lambda: coordinator.counters.evaluation_finishes >= target or bool(errors),
            (server, actor_thread),
        )
        assert not errors, errors
    finally:
        actor.stop.set()
        if actor_thread is not None:
            actor_thread.join(timeout=15)
        stop.set()
        server.join(timeout=15)
    assert actor_thread is not None
    assert not actor_thread.is_alive()
    assert not server.is_alive()
    assert not errors
