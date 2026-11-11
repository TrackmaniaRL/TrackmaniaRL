from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
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


def test_restarted_actor_acknowledges_assignment_from_recovered_spool(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-recovered-spool")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        old_request = _empty_request(coordinator, 5)
        original = coordinator.codec.decode(_submit(coordinator, old_request).value)
        # The actor restarts and retries its durable old-session request, then
        # executes the assignment in the new session.
        retried = coordinator.codec.decode(_submit(coordinator, old_request).value)
        assert retried["evaluation_assignment_id"] == original["evaluation_assignment_id"]
        payload = _base_payload(0)
        payload["session_id"] = "new-session"
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        summary["evaluation_assignment_id"] = original["evaluation_assignment_id"]
        payload["evaluations"] = [summary]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        response = coordinator.codec.decode(
            _submit(coordinator, _request(coordinator, payload)).value
        )
        assert response["evaluate"] is False
        assert "actor" not in coordinator._evaluation_assignments
        # A retry after the response is lost must not recreate the assignment.
        retry = coordinator.codec.decode(_submit(coordinator, _request(coordinator, payload)).value)
        assert retry["evaluate"] is False
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


def test_evaluation_snapshot_must_match_assignment_before_ack(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-snapshot-mismatch")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode(
        {"weights": torch.tensor([1.0, 2.0]), "metadata": {"a": 1, "b": 2}}
    )
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        payload = _base_payload(1)
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        summary["evaluation_assignment_id"] = first["evaluation_assignment_id"]
        payload["evaluations"] = [summary]
        payload["evaluation_snapshot"] = coordinator.codec.encode(
            {"weights": torch.tensor([1.0, 3.0]), "metadata": {"a": 1, "b": 2}}
        )
        with pytest.raises(RuntimeError, match=r"FAILED_PRECONDITION.*snapshot"):
            _submit(coordinator, _request(coordinator, payload))
        assert "actor" in coordinator._evaluation_assignments
        assert [row for row, _ in coordinator.journal.rows_after(0)] == [1]

        # Mapping order and encoding differ, but the policy state is identical.
        payload["evaluation_snapshot"] = coordinator.codec.encode(
            {"metadata": {"b": 2, "a": 1}, "weights": torch.tensor([1.0, 2.0])}
        )
        request = _request(coordinator, payload)
        accepted = coordinator.codec.decode(_submit(coordinator, request).value)
        retry = coordinator.codec.decode(_submit(coordinator, request).value)
        assert accepted["accepted"] is True
        assert retry["duplicate"] is True
        assert "actor" not in coordinator._evaluation_assignments
    finally:
        _close(coordinator)


def test_actor_evaluation_preserves_assigned_state_before_exploration_and_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Lock
    from types import SimpleNamespace

    from trackmaniarl.distributed import actor_evaluation

    class ExploratoryStatefulPolicy:
        def __init__(self) -> None:
            self.weights = torch.tensor([0.0])
            self.epsilon = 0.0

        def load_state(self, state: dict[str, Any]) -> None:
            self.weights = state["weights"]
            self.epsilon = state["epsilon"]

        def export_state(self) -> dict[str, Any]:
            return {"weights": self.weights, "epsilon": self.epsilon}

        def set_exploration_epsilon(self, value: float) -> None:
            self.epsilon = value

        def act(self) -> None:
            self.weights.add_(1)

    coordinator = _coordinator(tmp_path, "evaluation-stateful-actor")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode(
        {"weights": torch.tensor([3.0]), "epsilon": 0.5}
    )
    try:
        assignment = coordinator.codec.decode(
            _submit(coordinator, _empty_request(coordinator, 0)).value
        )
        policy = ExploratoryStatefulPolicy()
        spooled: list[Any] = []
        runtime = SimpleNamespace(
            spec=SimpleNamespace(evaluation=None),
            codec=coordinator.codec,
            _evaluation_request_lock=Lock(),
            _evaluation_request=(
                assignment["evaluation_snapshot"],
                assignment["evaluation_policy_version"],
                assignment["evaluation_assignment_id"],
            ),
            _new_policy=lambda: policy,
            _active_evaluation_assignment_id="",
            _active_evaluation_initial_snapshot=b"",
            _spool=spooled.append,
        )
        runtime._evaluation_policy = lambda: actor_evaluation.evaluation_policy(runtime)

        def trials(plan: Any, count: int) -> list[dict[str, Any]]:
            assert count == 1
            assert plan.policy.epsilon == 0.5
            plan.policy.act()
            plan.policy.act()
            return [
                summarize_episode(
                    0.0,
                    {
                        "policy_version": 1,
                        "termination_reason": "finished",
                        "race_time_ms": 36_000.0,
                    },
                    1,
                )
            ]

        monkeypatch.setattr(actor_evaluation, "_evaluation_trials", trials)
        actor_evaluation.evaluate(runtime, object(), object())
        result = _base_payload(1)
        result["evaluations"] = spooled[0].evaluations
        result["evaluation_snapshot"] = spooled[0].evaluation_snapshot
        response = coordinator.codec.decode(
            _submit(coordinator, _request(coordinator, result)).value
        )
        assert response["accepted"] is True
        assert "actor" not in coordinator._evaluation_assignments
        assert policy.weights.item() == 5.0
        assert policy.epsilon == 0.5
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


def test_reassigned_evaluation_is_counted_once_even_after_recovery(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-deduplicate")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        new_request = _base_payload(0)
        new_request["session_id"] = "new-session"
        _submit(coordinator, _request(coordinator, new_request))
        for session in ("session", "new-session"):
            payload = _base_payload(1)
            payload["session_id"] = session
            summary = summarize_episode(
                0.0,
                {
                    "policy_version": 1,
                    "termination_reason": "finished",
                    "race_time_ms": 10_000.0 if session == "session" else 36_000.0,
                },
                1,
            )
            summary["evaluation_assignment_id"] = first["evaluation_assignment_id"]
            payload["evaluations"] = [summary]
            payload["evaluation_snapshot"] = coordinator._policy_payload
            _submit(coordinator, _request(coordinator, payload))
        coordinator._recovering = True
        for row_id, raw in coordinator.journal.rows_after(0):
            coordinator._ingest(coordinator.codec.decode(raw), row_id)
        assert coordinator.counters.evaluations == 1
        assert coordinator.counters.evaluation_finishes == 1
        assert coordinator.counters.evaluation_bucket_finishes["sub_36"] == 0
    finally:
        _close(coordinator)

    # The durable marker must not suppress replay of the canonical row when a
    # crash occurs before its effects reach a checkpoint.
    coordinator = _coordinator(tmp_path, "evaluation-deduplicate")
    try:
        coordinator._recovering = True
        for row_id, raw in coordinator.journal.rows_after(0):
            coordinator._ingest(coordinator.codec.decode(raw), row_id)
        assert coordinator.counters.evaluations == 1
        assert coordinator.counters.evaluation_finishes == 1
        assert coordinator.counters.evaluation_bucket_finishes["sub_36"] == 0
        coordinator.journal.prune(coordinator.counters.journal_applied_frontier)
    finally:
        _close(coordinator)

    coordinator = _coordinator(tmp_path, "evaluation-deduplicate")
    try:
        assert coordinator.journal.evaluation_row_accepted(3) is False
        assert coordinator.journal.evaluation_row_accepted(4) is True
    finally:
        _close(coordinator)


def test_identified_evaluation_batch_cannot_mix_assignments(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-mixed-assignment")
    try:
        payload = _base_payload(0)
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        payload["evaluations"] = [
            {**summary, "evaluation_assignment_id": "one"},
            {**summary, "evaluation_assignment_id": "two"},
        ]
        payload["evaluation_snapshot"] = coordinator.codec.encode({"model": {}})
        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT: invalid rollout payload"):
            _submit(coordinator, _request(coordinator, payload))
    finally:
        _close(coordinator)


def test_evaluation_admission_is_atomic_with_journal_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-admission-atomic")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        payload = _base_payload(1)
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        summary["evaluation_assignment_id"] = first["evaluation_assignment_id"]
        payload["evaluations"] = [summary]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        request = _request(coordinator, payload)

        def fail(*_: object) -> bool:
            raise RuntimeError("simulated admission failure")

        with monkeypatch.context() as context:
            context.setattr(coordinator.journal, "_claim_canonical_evaluation", fail)
            with pytest.raises(RuntimeError, match="simulated admission failure"):
                _submit(coordinator, request)
        assert not coordinator.journal.has_receipt("session", 1, request.value)
        assert len(list(coordinator.journal.rows_after(0))) == 1
        response = coordinator.codec.decode(_submit(coordinator, request).value)
        assert response["accepted"] is True
        coordinator._recovering = True
        for row_id, raw in coordinator.journal.rows_after(0):
            coordinator._ingest(coordinator.codec.decode(raw), row_id)
        assert coordinator.counters.evaluations == 1
    finally:
        _close(coordinator)


def test_evaluation_admission_serializes_reassignment_after_durable_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-admission-concurrency")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    committed, release, reassignment_started, reassignment_done = (Event() for _ in range(4))
    failures: list[BaseException] = []
    workers: list[Thread] = []
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        payload = _base_payload(1)
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        summary["evaluation_assignment_id"] = first["evaluation_assignment_id"]
        payload["evaluations"] = [summary]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        original_append = coordinator.journal.append

        def delayed_append(*args: Any, **kwargs: Any) -> tuple[int, bool]:
            result = original_append(*args, **kwargs)
            if kwargs.get("evaluation") is not None:
                committed.set()
                if not release.wait(5):
                    raise TimeoutError("test failed to release admission")
            return result

        monkeypatch.setattr(coordinator.journal, "append", delayed_append)

        def submit_result() -> None:
            try:
                _submit(coordinator, _request(coordinator, payload))
            except BaseException as exc:
                failures.append(exc)

        def reassign() -> None:
            try:
                request = _base_payload(0)
                request["session_id"] = "new-session"
                reassignment_started.set()
                _submit(coordinator, _request(coordinator, request))
            except BaseException as exc:
                failures.append(exc)
            finally:
                reassignment_done.set()

        workers.append(Thread(target=submit_result))
        workers[-1].start()
        assert committed.wait(5)
        workers.append(Thread(target=reassign))
        workers[-1].start()
        assert reassignment_started.wait(5)
        assert not reassignment_done.wait(0.2)
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        assert not failures
        assert "actor" not in coordinator._evaluation_assignments
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
        _close(coordinator)


def test_earlier_delayed_submit_cannot_reassign_newer_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-delayed-older-submit")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    committed, release = Event(), Event()
    failures: list[BaseException] = []
    worker: Thread | None = None
    try:
        original_append = coordinator.journal.append

        def delayed_append(*args: Any, **kwargs: Any) -> tuple[int, bool]:
            result = original_append(*args, **kwargs)
            if args[0] == "session":
                committed.set()
                if not release.wait(5):
                    raise TimeoutError("test failed to release older submit")
            return result

        monkeypatch.setattr(coordinator.journal, "append", delayed_append)

        def submit_older() -> None:
            try:
                _submit(coordinator, _empty_request(coordinator, 0))
            except BaseException as exc:
                failures.append(exc)

        worker = Thread(target=submit_older)
        worker.start()
        assert committed.wait(5)
        later = _base_payload(0)
        later["session_id"] = "new-session"
        response = coordinator.codec.decode(
            _submit(coordinator, _request(coordinator, later)).value
        )
        assert response["evaluate"] is True
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert not failures
        assert coordinator._evaluation_assignments["actor"]["session"] == "new-session"
    finally:
        release.set()
        if worker is not None:
            worker.join(5)
        _close(coordinator)


def test_delayed_intermediate_session_cannot_reverse_newer_handoff(tmp_path: Path) -> None:
    from trackmaniarl.distributed.coordinator_submission import _assign_evaluation

    coordinator = _coordinator(tmp_path, "evaluation-session-order")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        _submit(coordinator, _empty_request(coordinator, 0))
        delayed = _base_payload(0)
        delayed["session_id"] = "session-b"
        delayed_request = _request(coordinator, delayed)
        row_id, _ = coordinator.journal.append("session-b", 0, delayed_request.value)
        current = _base_payload(0)
        current["session_id"] = "session-c"
        response = coordinator.codec.decode(
            _submit(coordinator, _request(coordinator, current)).value
        )
        assert response["evaluate"] is True
        assert coordinator._evaluation_assignments["actor"]["session"] == "session-c"
        delayed_response = _assign_evaluation(coordinator, delayed, row_id)
        assert not delayed_response.evaluate
        assert coordinator._evaluation_assignments["actor"]["session"] == "session-c"
    finally:
        _close(coordinator)

    restarted = _coordinator(tmp_path, "evaluation-session-order")
    try:
        assert restarted._evaluation_assignments["actor"]["session"] == "session-c"
        delayed_response = _assign_evaluation(restarted, delayed, row_id)
        assert not delayed_response.evaluate
        assert restarted._evaluation_assignments["actor"]["session"] == "session-c"
    finally:
        _close(restarted)


def test_evaluation_admission_survives_crash_before_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.distributed import coordinator_submission

    coordinator = _coordinator(tmp_path, "evaluation-admission-crash")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        payload = _base_payload(1)
        summary = summarize_episode(
            0.0,
            {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
            1,
        )
        summary["evaluation_assignment_id"] = first["evaluation_assignment_id"]
        payload["evaluations"] = [summary]
        payload["evaluation_snapshot"] = coordinator._policy_payload
        request = _request(coordinator, payload)

        def fail(*_: object) -> None:
            raise RuntimeError("simulated crash after append")

        with monkeypatch.context() as context:
            context.setattr(coordinator_submission, "_assign_evaluation", fail)
            with pytest.raises(RuntimeError, match="simulated crash after append"):
                _submit(coordinator, request)
        assert coordinator.journal.has_receipt("session", 1, request.value)
    finally:
        _close(coordinator)

    restarted = _coordinator(tmp_path, "evaluation-admission-crash")
    try:
        restarted._recovering = True
        for row_id, raw in restarted.journal.rows_after(0):
            restarted._ingest(restarted.codec.decode(raw), row_id)
        assert restarted.counters.evaluations == 1
        response = restarted.codec.decode(_submit(restarted, request).value)
        assert response["accepted"] is True
        assert response["duplicate"] is True
        assert "actor" not in restarted._evaluation_assignments
    finally:
        _close(restarted)


def test_late_result_from_other_session_does_not_reassign_new_evaluation(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-stale-session")
    coordinator._evaluation_due.add("actor")
    coordinator.counters.policy_version = 1
    coordinator._policy_payload = coordinator.codec.encode({"model": {}})
    try:
        first = coordinator.codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        coordinator._evaluation_due.add("actor")

        def result(session: str, sequence: int, assignment: str) -> dict[str, Any]:
            payload = _base_payload(sequence)
            payload["session_id"] = session
            summary = summarize_episode(
                0.0,
                {"policy_version": 1, "termination_reason": "finished", "race_time_ms": 36_000.0},
                1,
            )
            summary["evaluation_assignment_id"] = assignment
            payload["evaluations"] = [summary]
            payload["evaluation_snapshot"] = coordinator._policy_payload
            return payload

        second = coordinator.codec.decode(
            _submit(
                coordinator,
                _request(coordinator, result("session", 1, first["evaluation_assignment_id"])),
            ).value
        )
        _submit(
            coordinator,
            _request(coordinator, result("obsolete-session", 0, first["evaluation_assignment_id"])),
        )
        assert coordinator._evaluation_assignments["actor"]["session"] == "session"
        assert (
            coordinator._evaluation_assignments["actor"]["assignment_id"]
            == second["evaluation_assignment_id"]
        )
        _submit(
            coordinator,
            _request(coordinator, result("session", 2, second["evaluation_assignment_id"])),
        )
        assert "actor" not in coordinator._evaluation_assignments
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
