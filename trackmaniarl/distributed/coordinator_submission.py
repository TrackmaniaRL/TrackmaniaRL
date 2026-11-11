"""Durable rollout submission handling for coordinator RPCs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from queue import Full
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import grpc
import numpy as np
import torch
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.distributed import coordinator_delivery
from trackmaniarl.distributed.codec import WirePayloadTooLargeError
from trackmaniarl.distributed.coordinator_support import _RolloutRejection, snapshot_value
from trackmaniarl.distributed.coordinator_validation import (
    _evaluation_admission_error,
    _validate_submit_payload,
)
from trackmaniarl.distributed.journal import JournalPayloadConflictError

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator


@dataclass(frozen=True, slots=True)
class _Submission:
    coordinator: Coordinator
    value: Mapping[str, Any]
    lag: int
    stop: bool


@dataclass(frozen=True, slots=True)
class _EvaluationAssignment:
    evaluate: bool
    policy_version: int
    snapshot: bytes
    assignment_id: str = ""


@dataclass(frozen=True, slots=True)
class _AcceptedSubmission:
    submission: _Submission
    inserted: bool
    evaluation: _EvaluationAssignment
    context: grpc.ServicerContext[Any, Any]


def submit(
    coordinator: Coordinator,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> BytesValue:
    # Many concurrent tiny Tensor operations cause GIL handoff contention.
    # Keep every decode/schema/numeric check, but serialize this CPU work.
    # Release before taking coordinator/journal locks or serving responses.
    with coordinator._submission_decode_lock:
        value = coordinator._request(message, context)
        _validate_submission(coordinator, value, context)
    submission = _submission(coordinator, value)
    if _hard_lagged(submission) and not _previously_accepted(submission, message, context):
        return _reject_hard_lag(submission, context)
    evaluations = value["evaluations"]
    identified_result = bool(evaluations and evaluations[0].get("evaluation_assignment_id"))
    with coordinator._lock:
        assignment_active = str(value["actor_id"]) in coordinator._evaluation_assignments
        if identified_result or assignment_active:
            # Identified results and potential session reassignments must be
            # ordered with durable admission while an assignment is active.
            row_id, inserted = _append_journal(submission, message, context)
            evaluation = _assign_evaluation(coordinator, value, row_id)
    if not (identified_result or assignment_active):
        # Unrelated rollout writes must not block the coordinator lock while
        # waiting for storage, nor delay concurrent journal submissions.
        row_id, inserted = _append_journal(submission, message, context)
        evaluation = _assign_evaluation(coordinator, value, row_id)
    if inserted:
        _wake_learner(coordinator, row_id)
    return _accepted_response(_AcceptedSubmission(submission, inserted, evaluation, context))


def _validate_submission(
    coordinator: Coordinator,
    value: Mapping[str, Any],
    context: grpc.ServicerContext[Any, Any],
) -> None:
    try:
        _validate_submit_payload(
            value, coordinator.codec, pipeline=getattr(coordinator.run, "feature_pipeline", None)
        )
    except WirePayloadTooLargeError as exc:
        context.abort(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "distributed request exceeds the configured size limit",
        )
        raise AssertionError("gRPC abort returned") from exc
    except (KeyError, TypeError, ValueError) as exc:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid rollout payload")
        raise AssertionError("gRPC abort returned") from exc
    error = _evaluation_admission_error(coordinator.run.spec, value["evaluations"])
    if error is not None:
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, error)
        raise AssertionError("gRPC abort returned")


def _submission(coordinator: Coordinator, value: Mapping[str, Any]) -> _Submission:
    policy_version = int(value["policy_version"])
    with coordinator._lock:
        lag = max(0, coordinator.counters.updates - policy_version)
        stop = coordinator._should_stop()
    return _Submission(coordinator, value, lag, stop)


def _hard_lagged(submission: _Submission) -> bool:
    maximum = submission.coordinator.run.spec.distributed.hard_policy_lag_updates
    return bool(submission.value["transitions"]) and submission.lag > maximum


def _previously_accepted(
    submission: _Submission, message: BytesValue, context: grpc.ServicerContext[Any, Any]
) -> bool:
    try:
        return submission.coordinator.journal.has_receipt(
            str(submission.value["session_id"]), int(submission.value["sequence"]), message.value
        )
    except JournalPayloadConflictError as exc:
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise AssertionError("gRPC abort returned") from exc


def _reject_hard_lag(
    submission: _Submission, context: grpc.ServicerContext[Any, Any]
) -> BytesValue:
    coordinator = submission.coordinator
    maximum = coordinator.run.spec.distributed.hard_policy_lag_updates
    coordinator._log_rollout_rejected(
        submission.value,
        _RolloutRejection(
            "hard_policy_lag",
            {"policy_lag_updates": submission.lag, "hard_policy_lag_updates": maximum},
        ),
    )
    return coordinator._response(_hard_lag_payload(submission), context)


def _hard_lag_payload(submission: _Submission) -> dict[str, object]:
    coordinator = submission.coordinator
    return {
        "accepted": False,
        "reason": "hard_policy_lag",
        "force_refresh": True,
        "stop": submission.stop,
        "policy_lag_updates": submission.lag,
        "evaluate": False,
        "evaluation_policy_version": coordinator.counters.policy_version,
        "evaluation_snapshot": b"",
    }


def _append_journal(
    submission: _Submission,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> tuple[int, bool]:
    coordinator = submission.coordinator
    value = submission.value
    evaluations = value["evaluations"]
    assignment_id = evaluations[0].get("evaluation_assignment_id") if evaluations else None
    previous = (
        coordinator._evaluation_assignments.get(str(value["actor_id"])) if assignment_id else None
    )
    eligible = _matches_assignment(previous, value) if assignment_id else False
    if eligible and previous is not None:
        assigned = coordinator.codec.decode(previous["snapshot"])
        returned = coordinator.codec.decode(value["evaluation_snapshot"])
        if not _policy_states_equal(assigned, returned):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "evaluation policy snapshot does not match the assigned policy",
            )
            raise AssertionError("gRPC abort returned")
    try:
        if assignment_id:
            assignment_row = int(previous["assigned_row_id"]) if previous is not None else 0
            return coordinator.journal.append(
                value["session_id"],
                value["sequence"],
                message.value,
                evaluation=(str(value["actor_id"]), assignment_id, eligible, assignment_row),
            )
        return coordinator.journal.append(value["session_id"], value["sequence"], message.value)
    except JournalPayloadConflictError as exc:
        coordinator._log_rollout_rejected(value, _RolloutRejection("payload_conflict"))
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise AssertionError("gRPC abort returned") from exc
    except Exception as exc:
        coordinator._log_wal_error("append", exc)
        raise


def _policy_states_equal(left: Any, right: Any) -> bool:
    """Compare decoded policy values without relying on serialized byte order."""
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_policy_states_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_policy_states_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right))
        )
    return type(left) is type(right) and bool(left == right)


def _wake_learner(coordinator: Coordinator, row_id: int) -> None:
    with suppress(Full):
        coordinator._rollouts.put_nowait((row_id, monotonic()))


def _matches_assignment(previous: Mapping[str, Any] | None, value: Mapping[str, Any]) -> bool:
    if previous is None or not value["evaluations"]:
        return False
    session = str(value["session_id"])
    same_session = session == previous["session"]
    eligible = (same_session and int(value["sequence"]) > previous["sequence"]) or (
        not same_session
        and bool(previous.get("assignment_id"))
        and session not in previous.get("retired_sessions", [])
    )
    return bool(eligible) and all(
        int(item["policy_version"]) == previous["version"]
        and item.get("evaluation_assignment_id") == previous.get("assignment_id")
        for item in value["evaluations"]
    )


def _assign_evaluation(
    coordinator: Coordinator, value: Mapping[str, Any], row_id: int
) -> _EvaluationAssignment:
    with coordinator._lock:
        actor_id = str(value["actor_id"])
        assignments = dict(coordinator._evaluation_assignments)
        previous = assignments.get(actor_id)
        evaluations = value["evaluations"]
        if previous is not None and row_id < previous.get(
            "session_row_id", previous["assigned_row_id"]
        ):
            # An earlier WAL append can finish later than the request that
            # created this assignment; it cannot take over its session.
            return _EvaluationAssignment(False, coordinator.counters.policy_version, b"")
        if (
            previous is not None
            and row_id > previous["assigned_row_id"]
            and _matches_assignment(previous, value)
        ):
            del assignments[actor_id]
            coordinator_delivery.save(coordinator, set(coordinator._evaluation_due), assignments)
            previous = None
        if previous is not None:
            same_request = (
                previous["session"] == value["session_id"]
                and previous["sequence"] == value["sequence"]
            )
            if (
                not evaluations
                and previous["session"] != value["session_id"]
                and value["session_id"] not in previous.get("retired_sessions", [])
            ):
                previous = {
                    **previous,
                    "session": str(value["session_id"]),
                    "sequence": int(value["sequence"]),
                    "session_row_id": row_id,
                    "retired_sessions": [
                        *previous.get("retired_sessions", []),
                        previous["session"],
                    ],
                }
                assignments[actor_id] = previous
                coordinator_delivery.save(
                    coordinator, set(coordinator._evaluation_due), assignments
                )
                same_request = True
            if same_request:
                return _EvaluationAssignment(
                    True,
                    previous["version"],
                    previous["snapshot"],
                    previous.get("assignment_id", ""),
                )
            return _EvaluationAssignment(False, coordinator.counters.policy_version, b"")
        evaluate = (
            actor_id in coordinator._evaluation_due and coordinator.counters.policy_version > 0
        )
        version = coordinator.counters.policy_version
        snapshot = coordinator._policy_payload if evaluate else b""
        assignment_id = uuid4().hex if evaluate else ""
        if evaluate:
            _remember_evaluation_policy(coordinator, version, snapshot)
            assignments[actor_id] = {
                "session": str(value["session_id"]),
                "sequence": int(value["sequence"]),
                "assigned_row_id": row_id,
                "session_row_id": row_id,
                "version": version,
                "snapshot": snapshot,
                "assignment_id": assignment_id,
                "retired_sessions": [],
            }
            due = coordinator._evaluation_due - {actor_id}
            coordinator_delivery.save(coordinator, due, assignments)
    return _EvaluationAssignment(evaluate, version, snapshot, assignment_id)


def _remember_evaluation_policy(coordinator: Coordinator, version: int, snapshot: bytes) -> None:
    policy_state = coordinator.codec.decode(snapshot)
    if not isinstance(policy_state, Mapping):
        raise ValueError("published policy snapshot must decode to a mapping")
    coordinator._evaluation_policy_states[version] = snapshot_value(policy_state)
    while len(coordinator._evaluation_policy_states) > 16:
        oldest = next(iter(coordinator._evaluation_policy_states))
        coordinator._evaluation_policy_states.pop(oldest)


def _accepted_response(accepted: _AcceptedSubmission) -> BytesValue:
    submission = accepted.submission
    evaluation = accepted.evaluation
    coordinator = submission.coordinator
    payload = {
        "accepted": True,
        "duplicate": not accepted.inserted,
        "force_refresh": submission.lag > coordinator.run.spec.distributed.soft_policy_lag_updates,
        "stop": submission.stop,
        "policy_lag_updates": submission.lag,
        "evaluate": evaluation.evaluate,
        "evaluation_policy_version": evaluation.policy_version,
        "evaluation_snapshot": evaluation.snapshot,
        "evaluation_assignment_id": evaluation.assignment_id,
    }
    return coordinator._response(payload, accepted.context)
