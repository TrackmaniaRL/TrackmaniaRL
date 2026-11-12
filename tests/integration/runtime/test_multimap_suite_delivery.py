"""End-to-end suite delivery with a simulated map backend and real WAL.

This does not attest a real game or bypass production's multi-map guard.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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


class _SimulatedMapFactory:
    def __init__(self, events: list[str], actor: ActorRuntime, *, interrupt: bool) -> None:
        self.events, self.actor, self.interrupt = events, actor, interrupt

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
