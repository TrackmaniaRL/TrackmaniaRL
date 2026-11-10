from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from google.protobuf.wrappers_pb2 import BytesValue

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Context,
    _ephemeral_port,
    _resolved_run,
    _transition,
    _TransitionSpec,
)
from trackmaniarl.distributed.actor_metrics import summarize_episode
from trackmaniarl.distributed.coordinator import (
    Coordinator,
)
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    transition_to_wire,
)


def _coordinator(tmp_path: Path, run_id: str) -> Coordinator:
    training = {
        "total_transitions": 10,
        "warmup_transitions": 10,
        "checkpoint_interval_updates": None,
    }
    run = _resolved_run(tmp_path, run_id, training)
    config = CoordinatorConfig(f"127.0.0.1:{_ephemeral_port()}", _DISTRIBUTED_TOKEN, "fingerprint")
    return Coordinator(run, config)


def _base_payload(sequence: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": "actor",
        "session_id": "session",
        "sequence": sequence,
        "policy_version": 0,
        "transitions": [],
        "episodes": [],
        "evaluations": [],
        "evaluation_snapshot": b"",
    }


def _request(coordinator: Coordinator, payload: dict[str, Any]) -> BytesValue:
    return BytesValue(value=coordinator.codec.encode(payload))


def _submit(coordinator: Coordinator, request: BytesValue) -> BytesValue:
    context = cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}"))
    return coordinator._submit(request, context)


def _close(coordinator: Coordinator) -> None:
    coordinator._checkpoint_writer.close()
    coordinator.journal.close()


def _apply_malformation(
    coordinator: Coordinator, malformation: str, payload: dict[str, Any]
) -> None:
    transition = payload["transitions"][0]
    if malformation == "transitions_not_list":
        payload["transitions"] = {}
        return
    _apply_payload_malformation(coordinator, malformation, payload)
    if malformation == "non_finite_reward":
        transition["reward"] = torch.tensor(float("nan"))
    elif malformation == "invalid_action":
        transition["action"] = b"not-a-numeric-pytree"
    elif malformation == "wrong_owner_transition":
        transition["episode_id"] = "other/session/episode"


def _apply_payload_malformation(
    coordinator: Coordinator, malformation: str, payload: dict[str, Any]
) -> None:
    if malformation == "incomplete_episode":
        _set_incomplete_episode(payload)
    elif malformation == "missing_time_attack_reward":
        _set_episode_without_time_attack_reward(payload)
    elif malformation == "missing_evaluation_snapshot":
        _set_evaluation_without_snapshot(payload)
    elif malformation == "invalid_evaluation_snapshot":
        _set_invalid_snapshot(coordinator, payload)


def _malformed_payload(coordinator: Coordinator, malformation: str) -> dict[str, Any]:
    payload = _payload_with_transition()
    _apply_malformation(coordinator, malformation, payload)
    return payload


def _payload_with_transition() -> dict[str, Any]:
    transition = transition_to_wire(_transition(_TransitionSpec("actor", 0, 1.0)))
    payload = _base_payload(0)
    payload["transitions"] = [transition]
    return payload


def _set_incomplete_episode(payload: dict[str, Any]) -> None:
    payload["transitions"] = []
    payload["episodes"] = [{"finished": True, "finish_time_s": 1.0}]


def _set_episode_without_time_attack_reward(payload: dict[str, Any]) -> None:
    payload["transitions"] = []
    summary = summarize_episode(
        0.0,
        {"policy_version": 0, "termination_reason": "finished", "race_time_ms": 1_000.0},
        1,
    )
    summary["episode_id"] = "actor/session/episode"
    summary.pop("reward/time_attack_terminal")
    payload["episodes"] = [summary]


def _set_evaluation_without_snapshot(payload: dict[str, Any]) -> None:
    payload["transitions"] = []
    payload["evaluations"] = [
        {"finished": True, "finish_time_s": 1.0, "policy_version": 1, "steps": 1}
    ]


def _set_invalid_snapshot(coordinator: Coordinator, payload: dict[str, Any]) -> None:
    payload["transitions"] = []
    payload["evaluations"] = [
        {"finished": True, "finish_time_s": 1.0, "policy_version": 1, "steps": 1}
    ]
    payload["evaluation_snapshot"] = coordinator.codec.encode(["not", "a", "mapping"])


def _empty_request(coordinator: Coordinator, sequence: int) -> BytesValue:
    return _request(coordinator, _base_payload(sequence))


def _transition_request(coordinator: Coordinator, sequence: int) -> BytesValue:
    payload = _base_payload(sequence)
    spec = _TransitionSpec("actor", sequence, 1.0)
    payload["transitions"] = [transition_to_wire(_transition(spec))]
    return _request(coordinator, payload)


def _fail_append(session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
    del session_id, sequence, payload
    raise OSError("simulated WAL I/O failure")


def _induce_incidents(coordinator: Coordinator, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    lag = coordinator.run.spec.distributed.hard_policy_lag_updates
    coordinator.counters.updates = lag + 1
    rejected = _submit(coordinator, _transition_request(coordinator, 0))
    coordinator.counters.updates = 0
    monkeypatch.setattr(coordinator.journal, "append", _fail_append)
    with pytest.raises(OSError, match="simulated WAL I/O failure"):
        _submit(coordinator, _transition_request(coordinator, 1))
    return coordinator.codec.decode(rejected.value)


_MALFORMATIONS = (
    "transitions_not_list",
    "non_finite_reward",
    "invalid_action",
    "wrong_owner_transition",
    "incomplete_episode",
    "missing_time_attack_reward",
    "missing_evaluation_snapshot",
    "invalid_evaluation_snapshot",
)


@pytest.mark.parametrize("malformation", _MALFORMATIONS)
def test_malformed_rollout_is_rejected_before_wal(tmp_path: Path, malformation: str) -> None:
    coordinator = _coordinator(tmp_path, f"invalid-before-wal-{malformation}")
    request = _request(coordinator, _malformed_payload(coordinator, malformation))
    try:
        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
            _submit(coordinator, request)
        assert not coordinator.journal.has_rows()
    finally:
        _close(coordinator)


def test_evaluation_waits_for_the_first_trained_policy_snapshot(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-after-warmup")
    codec = coordinator.codec
    coordinator._evaluation_due.add("actor")

    try:
        before_training = codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        assert before_training["evaluate"] is False
        assert "actor" in coordinator._evaluation_due

        coordinator.counters.policy_version = 1
        coordinator._policy_payload = codec.encode({"model": {}})
        after_training = codec.decode(_submit(coordinator, _empty_request(coordinator, 1)).value)
        assert after_training["evaluate"] is True
        assert "actor" not in coordinator._evaluation_due
    finally:
        _close(coordinator)


def test_lost_evaluation_response_is_replayed_after_restart_and_policy_lag(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-retry")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    snapshot = coordinator.codec.encode({"model": {}})
    coordinator._policy_payload = snapshot
    request = _transition_request(coordinator, 0)
    try:
        first = coordinator.codec.decode(_submit(coordinator, request).value)
        assert first["evaluate"] is True
        assert first["evaluation_snapshot"] == snapshot
    finally:
        _close(coordinator)

    restarted = _coordinator(tmp_path, "evaluation-retry")
    try:
        restarted.counters.updates = restarted.run.spec.distributed.hard_policy_lag_updates + 1
        retry = restarted.codec.decode(_submit(restarted, request).value)
        assert retry["accepted"] is True
        assert retry["duplicate"] is True
        assert retry["evaluate"] is True
        assert retry["evaluation_policy_version"] == 1
        assert retry["evaluation_snapshot"] == snapshot
        assert retry["evaluation_assignment_id"] == first["evaluation_assignment_id"]
    finally:
        _close(restarted)


def test_pending_evaluation_is_reassigned_to_restarted_actor(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-actor-restart")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    snapshot = coordinator.codec.encode({"model": {}})
    coordinator._policy_payload = snapshot
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        assert first["evaluate"] is True
        new_session = _base_payload(0)
        new_session["session_id"] = "new-session"
        response = coordinator.codec.decode(
            _submit(coordinator, _request(coordinator, new_session)).value
        )
        assert response["evaluate"] is True
        assert response["evaluation_snapshot"] == snapshot
        old_retry = coordinator.codec.decode(
            _submit(coordinator, _empty_request(coordinator, 0)).value
        )
        assert old_retry["evaluate"] is False
    finally:
        _close(coordinator)


def test_evaluation_result_acknowledges_assignment(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-ack")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        assignment = coordinator.codec.decode(
            _submit(coordinator, _empty_request(coordinator, 0)).value
        )
        payload = _base_payload(1)
        payload["evaluations"] = [
            summarize_episode(
                0.0,
                {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
                1,
            )
        ]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        payload["evaluations"][0]["evaluation_assignment_id"] = assignment[
            "evaluation_assignment_id"
        ]
        response = coordinator.codec.decode(
            _submit(coordinator, _request(coordinator, payload)).value
        )
        assert response["evaluate"] is False
        assert "actor" not in coordinator._evaluation_assignments
        coordinator.journal.close()
        from trackmaniarl.distributed.journal import RolloutJournal

        coordinator.journal = RolloutJournal(coordinator.journal.path)
        from trackmaniarl.distributed.coordinator_delivery import load

        load(coordinator)
        assert "actor" not in coordinator._evaluation_assignments
    finally:
        _close(coordinator)


def test_retry_of_old_result_does_not_acknowledge_new_assignment(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-result-retry")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        assignment = coordinator.codec.decode(
            _submit(coordinator, _empty_request(coordinator, 0)).value
        )
        coordinator._evaluation_due.add("actor")
        payload = _base_payload(1)
        payload["evaluations"] = [
            summarize_episode(
                0.0,
                {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
                1,
            )
        ]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        payload["evaluations"][0]["evaluation_assignment_id"] = assignment[
            "evaluation_assignment_id"
        ]
        request = _request(coordinator, payload)
        first = coordinator.codec.decode(_submit(coordinator, request).value)
        retry = coordinator.codec.decode(_submit(coordinator, request).value)
        assert first["evaluate"] is True
        assert retry["evaluate"] is True
        assert retry["evaluation_snapshot"] == first["evaluation_snapshot"]
        assert "actor" in coordinator._evaluation_assignments
    finally:
        _close(coordinator)


def test_delayed_same_version_result_cannot_ack_new_assignment(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-out-of-order-results")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})

    def result(sequence: int, assignment_id: str) -> dict[str, Any]:
        payload = _base_payload(sequence)
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        summary["evaluation_assignment_id"] = assignment_id
        payload["evaluations"] = [summary]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        return payload

    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        coordinator._evaluation_due.add("actor")
        second = coordinator.codec.decode(
            _submit(
                coordinator, _request(coordinator, result(1, first["evaluation_assignment_id"]))
            ).value
        )
        assert second["evaluate"] is True
        assert second["evaluation_assignment_id"] != first["evaluation_assignment_id"]
        _submit(coordinator, _request(coordinator, result(2, first["evaluation_assignment_id"])))
        assert (
            coordinator._evaluation_assignments["actor"]["assignment_id"]
            == second["evaluation_assignment_id"]
        )
        _submit(coordinator, _request(coordinator, result(3, second["evaluation_assignment_id"])))
        assert "actor" not in coordinator._evaluation_assignments
    finally:
        _close(coordinator)


def test_retired_session_cannot_ack_reassigned_evaluation(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-retired-result")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        request = _base_payload(0)
        request["session_id"] = "new-session"
        reassigned = coordinator.codec.decode(
            _submit(coordinator, _request(coordinator, request)).value
        )
        assert reassigned["evaluation_assignment_id"] == first["evaluation_assignment_id"]
        stale = _base_payload(1)
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        summary["evaluation_assignment_id"] = first["evaluation_assignment_id"]
        stale["evaluations"] = [summary]
        stale["evaluation_snapshot"] = coordinator._policy_payload
        _submit(coordinator, _request(coordinator, stale))
        assert coordinator._evaluation_assignments["actor"]["session"] == "new-session"
    finally:
        _close(coordinator)


def test_legacy_assignment_without_identity_accepts_legacy_result(tmp_path: Path) -> None:
    from trackmaniarl.distributed import coordinator_delivery

    coordinator = _coordinator(tmp_path, "evaluation-legacy-result")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    payload = _base_payload(1)
    assignment = {
        "session": payload["session_id"],
        "sequence": 0,
        "assigned_row_id": 0,
        "version": 1,
        "snapshot": coordinator._policy_payload,
        "retired_sessions": [],
    }
    try:
        coordinator_delivery.save(coordinator, set(), {"actor": assignment})
        payload["evaluations"] = [
            summarize_episode(
                0.0,
                {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
                1,
            )
        ]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        _submit(coordinator, _request(coordinator, payload))
        assert "actor" not in coordinator._evaluation_assignments
    finally:
        _close(coordinator)


def test_recovery_schedules_unprocessed_evaluation_once(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from trackmaniarl.distributed import coordinator_delivery, coordinator_episode_ingest

    coordinator = _coordinator(tmp_path, "evaluation-schedule-recovery")
    original_run = coordinator.run
    coordinator.run = cast(
        Any,
        SimpleNamespace(spec=SimpleNamespace(training=SimpleNamespace(evaluate_every_episodes=1))),
    )
    value = {"actor_id": "actor", "episodes": [{"finished": False}]}
    try:
        coordinator._recovering = True
        coordinator_episode_ingest.ingest_episode_summaries(coordinator, value, 5)
        assert coordinator._evaluation_due == {"actor"}
        assert coordinator._evaluation_schedule_frontier == 5
        coordinator_delivery.save(coordinator, set(), {})
        coordinator.counters.episodes = 0
        coordinator_episode_ingest.ingest_episode_summaries(coordinator, value, 5)
        assert not coordinator._evaluation_due
        coordinator_episode_ingest.ingest_episode_summaries(coordinator, value, 6)
        assert coordinator._evaluation_due == {"actor"}
    finally:
        coordinator.run = original_run
        _close(coordinator)


def test_rollout_and_wal_failure_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = _coordinator(tmp_path, "runtime-incidents")
    try:
        _assert_incidents(coordinator, _induce_incidents(coordinator, monkeypatch))
    finally:
        _close(coordinator)


def _assert_incidents(coordinator: Coordinator, rejected: dict[str, Any]) -> None:
    assert rejected["reason"] == "hard_policy_lag"
    logger = coordinator.run.logger
    assert logger.events.count("distributed/rollout_rejected") == 1
    assert logger.events.count("distributed/wal_error") == 1
    incidents = dict(logger.records)
    assert incidents["distributed/rollout_rejected"]["reason"] == "hard_policy_lag"
    assert incidents["distributed/wal_error"]["operation"] == "append"


def _stale_evaluation_payload(coordinator: Coordinator) -> dict[str, Any]:
    maximum = coordinator.run.spec.distributed.hard_policy_lag_updates
    coordinator.counters.updates = maximum + 1
    payload = _base_payload(0)
    evaluation = summarize_episode(
        0.0,
        {"policy_version": 0, "termination_reason": "finished", "race_time_ms": 36_000.0},
        1,
    )
    evaluation.pop("termination")
    payload["evaluations"] = [evaluation]
    payload["evaluation_snapshot"] = coordinator.codec.encode({"model": {}})
    return payload


def test_stale_evaluation_only_payload_bypasses_hard_policy_lag(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "stale-evaluation")
    run = coordinator.run
    codec = coordinator.codec
    try:
        payload = _stale_evaluation_payload(coordinator)
        response = _submit(coordinator, _request(coordinator, payload))
        decoded = codec.decode(response.value)
        assert decoded["accepted"] is True
        assert decoded["force_refresh"] is True
        assert coordinator.journal.has_rows()
        assert run.logger.events.count("distributed/rollout_rejected") == 0
    finally:
        _close(coordinator)
