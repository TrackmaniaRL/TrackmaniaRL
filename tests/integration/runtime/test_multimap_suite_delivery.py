"""End-to-end suite delivery with a simulated map backend and real WAL.

The game and plugin are simulated; these tests cannot attest a real game.
"""

from __future__ import annotations

import json
import os
import socket
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Any

import numpy as np
import pytest

from tests.integration.runtime.distributed_evaluation_support import _DISTRIBUTED_TOKEN
from tests.integration.runtime.distributed_runtime_support import _Pipeline
from tests.integration.runtime.test_distributed_evaluation_integration import (
    AttestedEnvironment,
    _close,
    _config,
    _coordinator,
    _submit,
)
from trackmaniarl.distributed import actor_evaluation, evaluation_contract
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_background import _accept_evaluation_request
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.trackmania.reward_points import save_reward_points


class _SimulatedMapFactory:
    def __init__(self, events: list[str], actor: ActorRuntime, *, interrupt: bool) -> None:
        self.events, self.actor, self.interrupt = events, actor, interrupt

    def require_switch_capacity(self, training: Any, required: int) -> None:
        assert required == 3

    @contextmanager
    def staged_suite(self, suite: Any, base: Path, assets: str) -> Iterator[Any]:
        assert assets
        assert base.is_dir()
        yield suite

    def evaluation_pipeline(self, training: Any, map_spec: Any) -> _Pipeline:
        return _Pipeline()

    @contextmanager
    def evaluation_map(self, training: Any, map_spec: Any) -> Iterator[Any]:
        self.events.append(f"switch:{map_spec.id}")
        actor, interrupt = self.actor, self.interrupt

        class Environment(AttestedEnvironment):
            def step(self, action: Any) -> Any:
                result = super().step(action)
                if interrupt:
                    actor.stop.set()
                return result

        yield Environment()

    def verify_evaluation_map(self, training: Any, map_spec: Any) -> None:
        self.events.append(f"verify:{map_spec.id}")

    def restore_training(self, training: Any) -> None:
        self.events.append("restore")

    def abort_evaluation(self, training: Any, error: BaseException) -> None:
        self.events.append("abort")


@pytest.mark.parametrize("mismatch", [False, True])
def test_experimental_factory_integration_with_explicit_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mismatch: bool
) -> None:
    from tests.integration.runtime.distributed_runtime_support import _Logger
    from tests.integration.runtime.test_distributed_evaluation_integration import TestLearner
    from trackmaniarl.core.builtins import IdentityFeaturePipeline, TorchCheckpointCodec
    from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
    from trackmaniarl.core.runtime import ResolvedRun
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.coordinator_types import CoordinatorConfig

    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("training", "one", "two"):
        track = tmp_path / f"{name}.gbx"
        track.write_bytes(name.encode())
        save_reward_points(
            tmp_path / f"{name}.npz",
            np.asarray([[0, 0, 0], [20, 0, 0]], dtype=np.float32),
            map_uid=name,
            map_path=track,
        )
    config["components"]["environment"] = {
        "class_path": (
            "trackmaniarl.trackmania.multimap_environment:OpenPlanetMultiMapEnvironmentFactory"
        ),
        "kwargs": {
            "config": {
                "expected_map_uid": "training",
                "reward_points_path": "training.npz",
                "action_repeat_frames": 1,
                "minimum_finish_steps": 1,
            },
            "training_map_path": "training.gbx",
            "journal_path": "new-switch.sqlite",
            "instance_id": "isolated-test",
            "staging_root": str(tmp_path),
            "enable_experimental_multimap": True,
            "assume_exclusive_no_unobserved_map_changes": True,
        },
    }
    config["components"]["feature_pipeline"] = {
        "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
    }
    config["evaluation"]["maps"] = [
        {
            "id": name,
            "expected_map_uid": name,
            "map_path": f"{name}.gbx",
            "reward_points_path": f"{name}.npz",
        }
        for name in ("one", "two")
    ]
    path.write_text(json.dumps(config), encoding="utf-8")
    actor = ActorRuntime(ActorProcessRequest(path, "127.0.0.1:1", "actor", _DISTRIBUTED_TOKEN))
    # The simulated learner is deliberately not on the real policy allowlist.
    # Real first-party policy isolation is covered in the unit suite.
    monkeypatch.setattr(actor_evaluation, "_supports_multimap_policy", lambda _: True)
    monkeypatch.setattr(actor_evaluation, "_validate_model_isolation", lambda *_: None)
    try:
        pipeline, factory = actor._components()
        run = ResolvedRun(
            spec=actor.spec,
            run_dir=tmp_path / "coordinator",
            learner=TestLearner(),
            environment_factory=factory,
            model_factory=None,
            replay_store=InMemoryReplayStore(),
            sampler=UniformSampler(pipeline, seed=0),
            feature_pipeline=pipeline,
            logger=_Logger(),
            checkpoint_codec=TorchCheckpointCodec(),
            evaluator=None,
            base_dir=tmp_path,
        )
        coordinator = Coordinator(
            run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, actor.fingerprint)
        )
        # Admission does not contact the game or create a switch journal.
        assert isinstance(pipeline, IdentityFeaturePipeline)
        assert not (tmp_path / "new-switch.sqlite").exists()
        if os.name != "nt":
            _close(coordinator)
            return  # The staging backend is Windows-only.

        from tests.unit.trackmania.test_map_switch_controller import FakePlugin
        from trackmaniarl.distributed.actor_transport import Client
        from trackmaniarl.trackmania import map_switch_controller, multimap_environment
        from trackmaniarl.trackmania.bound_telemetry import BoundTelemetryError
        from trackmaniarl.trackmania.environment import OpenPlanetEnvironment
        from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal

        class FakeController:
            def __init__(self) -> None:
                self.frames = 0

            def reset(self) -> None:
                self.frames = 0

            def apply(self, action: Any) -> None:
                pass

            def apply_discrete(self, action: Any) -> None:
                pass

            def consume_collision(self) -> bool:
                return False

            def confirm_finish(self) -> None:
                pass

            def close(self) -> None:
                pass

        plugin = FakePlugin()
        controller = FakeController()

        class FakeSession:
            def verify_loaded_map(self, uid: str) -> None:
                assert plugin.uid == uid

            def confirm_ready(self, uid: str) -> None:
                assert plugin.uid == uid

        # The only network listener is this test-owned, ephemeral P3 endpoint.
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.1)
        stop_server = Event()
        requests: list[tuple[str, str, bool]] = []
        server_errors: list[BaseException] = []

        def serve_p3() -> None:
            try:
                while not stop_server.is_set():
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    with connection:
                        connection.settimeout(0.1)
                        stream = bytearray()
                        while not stop_server.is_set():
                            try:
                                data = connection.recv(4096)
                            except TimeoutError:
                                continue
                            except ConnectionResetError:
                                break  # A rejected frame may close with unread payload bytes.
                            if not data:
                                break
                            stream.extend(data)
                            while b"\n" in stream:
                                line, _, rest = stream.partition(b"\n")
                                stream = bytearray(rest)
                                request = json.loads(line)
                                assert set(request) == {"command", "request_id", "wait_ready"}
                                assert request["command"] == "frame"
                                assert type(request["wait_ready"]) is bool
                                controller.frames += 1
                                requests.append(
                                    (plugin.uid, plugin.operation, request["wait_ready"])
                                )
                                uid = plugin.uid
                                if mismatch and uid == "one" and controller.frames == 3:
                                    uid = "wrong-map"
                                header = {
                                    "protocol_version": "3",
                                    "request_id": request["request_id"],
                                    "instance_epoch": plugin.instance,
                                    "operation_id": plugin.operation,
                                    "map_uid": uid,
                                    "state": "ready",
                                    "payload_bytes": 132,
                                }
                                values = [0.0] * 33
                                values[2] = float(controller.frames >= 3)
                                values[3] = float(100 * controller.frames)
                                values[4] = 20.0 if controller.frames >= 3 else 0.0
                                connection.sendall(
                                    json.dumps(header).encode()
                                    + b"\n"
                                    + struct.pack("<33f", *values)
                                )
            except BaseException as error:
                server_errors.append(error)

        server = Thread(target=serve_p3, daemon=True)
        server.start()
        monkeypatch.setattr(
            multimap_environment, "_BOUND_TELEMETRY_PORT", listener.getsockname()[1]
        )
        monkeypatch.setattr(map_switch_controller, "_registry_root", lambda: tmp_path / "registry")
        monkeypatch.setattr(multimap_environment, "OpenPlanetSwitchTransport", lambda *_: plugin)
        monkeypatch.setattr(
            OpenPlanetEnvironment, "_create_session", staticmethod(lambda _: FakeSession())
        )
        monkeypatch.setattr(
            multimap_environment.OpenPlanetEnvironmentFactory,
            "create",
            lambda self, **kwargs: OpenPlanetEnvironment(self.config, controller),
        )
        # This learner uses a test policy; real first-party isolation is covered
        # separately. The suite loop, factory, staging, journal and RPC are real.
        MapSwitchJournal(tmp_path / "new-switch.sqlite", "isolated-test", create=True)
        try:
            training = factory.create(seed=0)
        except BaseException:
            stop_server.set()
            server.join(timeout=5)
            listener.close()
            _close(coordinator)
            raise
        client = None
        try:
            coordinator._start_server()  # port 0 asks the OS for an unused port
            client = Client(f"127.0.0.1:{coordinator.bound_port}", _DISTRIBUTED_TOKEN, actor.codec)
            coordinator._evaluation_due.add(actor.actor_id)
            coordinator.counters.policy_version = 1
            coordinator._policy_payload = coordinator.codec.encode({"value": 7})
            submit = {
                **actor._request_base(),
                "sequence": 0,
                "policy_version": 1,
                "transitions": [],
                "episodes": [],
                "evaluations": [],
                "evaluation_snapshot": b"",
            }
            assignment = client.call("Submit", submit)
            _accept_evaluation_request(actor, assignment)
            actor.sequence = 1
            if mismatch:
                with pytest.raises(BoundTelemetryError, match="identity differs"):
                    actor._evaluate(training, pipeline)
                assert actor.queue.empty()
                assert plugin.uid == "one"
                assert plugin.switch_count == 2  # initial training and first evaluation map
                assert training.switch._telemetry_quarantine_path.exists()
                assert [(uid, ready) for uid, _, ready in requests] == [
                    ("one", True),
                    ("one", False),
                    ("one", True),
                    ("one", False),
                    ("one", False),
                ]
                return
            actor._evaluate(training, pipeline)
            payload = actor.queue.get_nowait().read_bytes()
            result = actor.codec.decode(payload)
            assert [(item["map_id"], item["trial_index"]) for item in result["evaluations"]] == [
                ("one", 0),
                ("one", 1),
                ("two", 0),
                ("two", 1),
            ]
            assert plugin.uid == "training"  # restore happens before spool
            training.reset(seed=1)
            _, reward, terminated, truncated, _ = training.step(0)
            assert np.isfinite(reward)
            assert terminated
            assert not truncated
            assert plugin.uid == "training"
            assert [(uid, ready) for uid, _, ready in requests] == [
                (uid, ready)
                for uid, flags in (
                    ("one", (True, False, True, False, False, True, False, False)),
                    ("two", (True, False, True, False, False, True, False, False)),
                    ("training", (True, False, False)),
                )
                for ready in flags
            ]
            assert all(operation for _, operation, _ in requests)
            assert actor.codec.decode(result["evaluation_snapshot"]) == {"value": 7}
            assert all(
                item["evaluation_assets_sha256"] == assignment["evaluation_assets_sha256"]
                and item["evaluation_assignment_id"] == assignment["evaluation_assignment_id"]
                for item in result["evaluations"]
            )
            assert client.call("Submit", result)["accepted"] is True
            assert client.call("Submit", result)["accepted"] is True
        finally:
            training.close()
            stop_server.set()
            server.join(timeout=5)
            listener.close()
            assert not server.is_alive()
            assert not server_errors
            if client is not None:
                client.close()
            if coordinator._server is not None:
                coordinator._server.stop(0).wait()
                coordinator._rpc_executor.shutdown(wait=True)
            _close(coordinator)
        restored = Coordinator(
            run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, actor.fingerprint)
        )
        try:
            restored._recover_journal(0)
            assert restored.counters.evaluations == 4
            assert restored.counters.evaluation_finishes == 4
        finally:
            _close(restored)
    finally:
        actor.client.close()


def test_interrupted_suite_retries_same_assignment_and_recovers_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("one", "two"):
        (tmp_path / f"{name}.gbx").write_bytes(name.encode())
        (tmp_path / f"{name}.txt").write_bytes((name + "-reward").encode())
    config["evaluation"]["maps"] = [
        {
            "id": name,
            "expected_map_uid": f"uid-{name}",
            "map_path": f"{name}.gbx",
            "reward_points_path": f"{name}.txt",
        }
        for name in ("one", "two")
    ]
    path.write_text(json.dumps(config), encoding="utf-8")
    # Test the real suite loop and delivery, but do not claim backend attestation.
    monkeypatch.setattr(evaluation_contract, "validate_scheduled_evaluation", lambda *args: None)
    monkeypatch.setattr(actor_evaluation, "_supports_multimap_policy", lambda _: True)
    monkeypatch.setattr(actor_evaluation, "_validate_model_isolation", lambda *_: None)
    request = ActorProcessRequest(path, "127.0.0.1:1", "actor", _DISTRIBUTED_TOKEN)
    actor = ActorRuntime(request)
    coordinator = _coordinator(path, actor.fingerprint)
    events: list[str] = []
    try:
        pipeline, factory = actor._components()
        actor._multi_map_factory = _SimulatedMapFactory(events, actor, interrupt=True)
        coordinator._evaluation_due.add(actor.actor_id)
        coordinator.counters.policy_version = 1
        coordinator._policy_payload = coordinator.codec.encode({"value": 7})
        submit = {
            **actor._request_base(),
            "sequence": 0,
            "policy_version": 1,
            "transitions": [],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        assignment = _submit(coordinator, coordinator.codec.encode(submit))
        _accept_evaluation_request(actor, assignment)
        actor._evaluate(factory.create(seed=0), pipeline)
        assert events == ["switch:one", "restore"]
        assert actor.queue.empty()
    finally:
        actor.client.close()
        _close(coordinator)

    resumed = ActorRuntime(request)
    coordinator = _coordinator(path, resumed.fingerprint)
    events = []
    try:
        pipeline, factory = resumed._components()
        resumed._multi_map_factory = _SimulatedMapFactory(events, resumed, interrupt=False)
        submit = {
            **resumed._request_base(),
            "sequence": 0,
            "policy_version": 1,
            "transitions": [],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        retry = _submit(coordinator, coordinator.codec.encode(submit))
        assert retry["evaluation_assignment_id"] == assignment["evaluation_assignment_id"]
        assert retry["evaluation_assets_sha256"] == assignment["evaluation_assets_sha256"]
        _accept_evaluation_request(resumed, retry)
        resumed.sequence = 1
        resumed._evaluate(factory.create(seed=0), pipeline)
        assert events == ["switch:one", "verify:one", "switch:two", "verify:two", "restore"]
        payload = resumed.queue.get_nowait().read_bytes()
        result = coordinator.codec.decode(payload)
        assert [(x["map_id"], x["trial_index"]) for x in result["evaluations"]] == [
            ("one", 0),
            ("one", 1),
            ("two", 0),
            ("two", 1),
        ]
        assert coordinator.codec.decode(result["evaluation_snapshot"]) == {"value": 7}
        assert _submit(coordinator, payload)["accepted"] is True
        assert _submit(coordinator, payload)["accepted"] is True
    finally:
        resumed.client.close()
        _close(coordinator)

    restored = _coordinator(path, resumed.fingerprint)
    try:
        restored._recover_journal(0)
        assert restored.counters.evaluations == 4
        assert restored.counters.evaluation_finishes == 4
    finally:
        _close(restored)
