from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from trackmaniarl.core.checkpoint_milestones import CheckpointProgress, preserve_milestones
from trackmaniarl.core.checkpoints import prune_checkpoint_family
from trackmaniarl.core.fingerprint import fingerprint_details, fingerprint_mismatch_details
from trackmaniarl.distributed.coordinator_support import (
    _CheckpointWrite,
    _Counters,
    load_state_dict,
    snapshot_value,
    state_dict,
)
from trackmaniarl.distributed.coordinator_types import ReplayRestoreMode

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")

_CHECKPOINT_SCHEMA_VERSION = "2.0"
_JOURNAL_CONTRACT_VERSION = 2
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "journal_contract_version",
        "journal_id",
        "run_fingerprint",
        "learner",
        "replay_store",
        "sampler",
        "distributed",
        "evaluated_policy_version",
    }
)
_DISTRIBUTED_KEYS = frozenset(
    {
        "transitions",
        "episodes",
        "finishes",
        "best_finish_time_s",
        "evaluations",
        "evaluation_finishes",
        "evaluation_bucket_finishes",
        "updates",
        "update_credit",
        "journal_applied_frontier",
        "policy_version",
        "actor_sequences",
    }
)
_DISTRIBUTED_OPTIONAL_KEYS = frozenset(
    {
        "best_evaluation_rank",
        "fastest_evaluation_rank",
        "consecutive_evaluation_passes",
        "evaluation_stop_reason",
    }
)


class EvaluationCheckpointKind(StrEnum):
    RELIABLE = "best-eval"
    FASTEST = "fastest-eval"


@dataclass(frozen=True, slots=True)
class EvaluatedPolicyCheckpoint:
    state: Mapping[str, Any]
    version: int
    kind: EvaluationCheckpointKind
    on_saved: Callable[[Path], None] | None = None
    on_failed: Callable[[BaseException], None] | None = None
    suppress_events: bool = False


@dataclass(frozen=True, slots=True)
class _CheckpointPlan:
    coordinator: Coordinator
    evaluated_policy: EvaluatedPolicyCheckpoint | None
    path: Path
    started_at: float
    transitions: int


@dataclass(frozen=True, slots=True)
class _CheckpointCallbacks:
    plan: _CheckpointPlan
    frontier: int
    update: int

    def saved(self) -> None:
        coordinator = self.plan.coordinator
        if self.plan.evaluated_policy is None:
            progress = CheckpointProgress(
                coordinator._segment_initial_transitions, self.plan.transitions, self.update
            )
            saved = preserve_milestones(
                self.plan.path, coordinator.run.spec.training.checkpoint_milestones, progress
            )
            for path in saved:
                coordinator.run.logger.log(
                    "train/checkpoint_milestone", {"path": str(path)}, step=self.update
                )
        # A leader checkpoint may be only one of two promotions for the same
        # evaluation and contains evaluated policy weights rather than the
        # current learner state. Only a full training checkpoint can retire WAL.
        if self.plan.evaluated_policy is None:
            try:
                coordinator.journal.prune(self.frontier)
            except Exception as exc:
                coordinator._log_wal_error("prune", exc)
                raise
        _apply_checkpoint_retention(self.plan, self.update)
        evaluated = self.plan.evaluated_policy
        if evaluated is None or not evaluated.suppress_events:
            _log_checkpoint_completed(self)
        if evaluated is not None and evaluated.on_saved is not None:
            evaluated.on_saved(self.plan.path)

    def failed(self, exc: BaseException) -> None:
        coordinator = self.plan.coordinator
        coordinator.run.logger.log(
            "train/checkpoint_failed",
            {
                "path": str(self.plan.path),
                "journal_applied_frontier": self.frontier,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            step=self.update,
        )
        evaluated = self.plan.evaluated_policy
        if evaluated is not None and evaluated.on_failed is not None:
            evaluated.on_failed(exc)


def _log_checkpoint_completed(callbacks: _CheckpointCallbacks) -> None:
    plan = callbacks.plan
    plan.coordinator.run.logger.log(
        "train/checkpoint_completed",
        {
            "path": str(plan.path),
            "journal_applied_frontier": callbacks.frontier,
            "duration_s": perf_counter() - plan.started_at,
        },
        step=callbacks.update,
    )


def _apply_checkpoint_retention(plan: _CheckpointPlan, update: int) -> None:
    coordinator = plan.coordinator
    keep = coordinator.run.spec.training.checkpoint_keep_last
    if keep is None:
        return
    result = prune_checkpoint_family(plan.path, coordinator.run.run_dir / "checkpoints", keep)
    if not result.removed:
        return
    removed = [str(path) for path in result.removed]
    coordinator.run.logger.log(
        "train/checkpoint_retention",
        {"family": result.family, "removed_count": len(removed), "paths": removed},
        step=update,
    )
    logger.info("Checkpoint retention removed: %s", ", ".join(removed))


@dataclass(frozen=True, slots=True)
class _RestoreRequest:
    coordinator: Coordinator
    path: Path
    mode: ReplayRestoreMode


def checkpoint(
    coordinator: Coordinator,
    evaluated_policy: EvaluatedPolicyCheckpoint | None = None,
) -> Path:
    coordinator._checkpoint_writer.wait()
    plan = _checkpoint_plan(coordinator, evaluated_policy)
    state = _checkpoint_state(plan)
    callbacks = _CheckpointCallbacks(
        plan, coordinator.counters.journal_applied_frontier, coordinator.counters.updates
    )
    coordinator._checkpoint_writer.submit(
        _CheckpointWrite(state, plan.path, callbacks.saved, callbacks.failed)
    )
    if evaluated_policy is None:
        coordinator._last_checkpoint_transitions = plan.transitions
    if evaluated_policy is None or not evaluated_policy.suppress_events:
        _log_checkpoint_queued(plan)
    return plan.path


def _checkpoint_plan(
    coordinator: Coordinator,
    evaluated_policy: EvaluatedPolicyCheckpoint | None,
) -> _CheckpointPlan:
    if evaluated_policy is not None:
        name = (
            f"{evaluated_policy.kind.value}-policy-{evaluated_policy.version:08d}-at-update-"
            f"{coordinator.counters.updates:08d}.pt"
        )
    else:
        name = f"distributed-update-{coordinator.counters.updates:08d}.pt"
    path = coordinator.run.run_dir / "checkpoints" / name
    return _CheckpointPlan(
        coordinator, evaluated_policy, path, perf_counter(), coordinator.counters.transitions
    )


def _checkpoint_state(plan: _CheckpointPlan) -> dict[str, Any]:
    coordinator = plan.coordinator
    state = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "journal_contract_version": _JOURNAL_CONTRACT_VERSION,
        "journal_id": coordinator.journal.identity,
        "run_fingerprint": coordinator.fingerprint,
        "fingerprint_manifest": _coordinator_fingerprint_manifest(coordinator),
        "learner": snapshot_value(_learner_state(plan)),
        "replay_store": snapshot_value(state_dict(coordinator.run.replay_store)),
        "sampler": snapshot_value(state_dict(coordinator.run.sampler)),
        "distributed": _distributed_state(coordinator),
        "evaluated_policy_version": (
            plan.evaluated_policy.version if plan.evaluated_policy is not None else None
        ),
    }
    if plan.evaluated_policy is not None:
        # Evaluated weights are historical; exact resume needs the separate
        # online learner, optimizer and target state from this snapshot.
        state["training_learner"] = snapshot_value(coordinator.run.learner.state_dict())
        # Either leader can be resumed. Preserve in-flight evidence in the
        # checkpoint itself, even after journal pending records are cleared.
        state["pending_evaluation_promotions"] = coordinator.journal.pending_evaluation_promotions()
    return state


def _coordinator_fingerprint_manifest(coordinator: Coordinator) -> dict[str, Any] | None:
    if hasattr(coordinator, "_fingerprint_manifest"):
        return coordinator._fingerprint_manifest
    identity = coordinator.fingerprint
    manifest = None
    if (
        isinstance(identity, str)
        and len(identity) == 64
        and all(c in "0123456789abcdef" for c in identity)
    ):
        try:
            actual, details = fingerprint_details(coordinator.run.spec, coordinator.run.base_dir)
        except Exception as exc:
            # This information is optional. A removed source file, inaccessible
            # asset or component import must not prevent a durable checkpoint.
            logger.warning(
                "Unable to compute optional fingerprint diagnostics (%s)", type(exc).__name__
            )
        else:
            if actual == identity:
                manifest = details
    coordinator._fingerprint_manifest = manifest
    return manifest


def _learner_state(plan: _CheckpointPlan) -> Mapping[str, Any]:
    learner = plan.coordinator.run.learner
    if plan.evaluated_policy is None:
        return learner.state_dict()
    exact_state = getattr(learner, "state_dict_for_policy", None)
    if not callable(exact_state):
        raise TypeError("learner cannot build an exact evaluated-policy checkpoint")
    return cast(Mapping[str, Any], exact_state(plan.evaluated_policy.state))


def _distributed_state(coordinator: Coordinator) -> dict[str, Any]:
    counters = coordinator.counters
    return {
        "transitions": counters.transitions,
        "episodes": counters.episodes,
        "finishes": counters.finishes,
        "best_finish_time_s": counters.best_finish_time_s,
        "evaluations": counters.evaluations,
        "evaluation_finishes": counters.evaluation_finishes,
        "evaluation_bucket_finishes": dict(counters.evaluation_bucket_finishes),
        "updates": counters.updates,
        "update_credit": counters.update_credit,
        "journal_applied_frontier": counters.journal_applied_frontier,
        "policy_version": counters.policy_version,
        "actor_sequences": dict(counters.actor_sequences),
        "best_evaluation_rank": coordinator._best_evaluation,
        "fastest_evaluation_rank": coordinator._fastest_evaluation,
        "consecutive_evaluation_passes": coordinator._consecutive_evaluation_passes,
        "evaluation_stop_reason": coordinator._evaluation_stop_reason,
    }


def _log_checkpoint_queued(plan: _CheckpointPlan) -> None:
    plan.coordinator.run.logger.log(
        "train/checkpoint",
        {
            "path": str(plan.path),
            "timing/checkpoint_snapshot_s": perf_counter() - plan.started_at,
        },
        step=plan.coordinator.counters.updates,
    )
    logger.info("Checkpoint queued: %s", plan.path)


def restore_checkpoint(
    coordinator: Coordinator,
    path: Path,
    mode: ReplayRestoreMode = ReplayRestoreMode.FULL,
) -> None:
    request = _RestoreRequest(coordinator, path, mode)
    state = coordinator.run.checkpoint_codec.load(path)
    _validate_checkpoint_state(request, state)
    if request.mode is ReplayRestoreMode.LEARNER_ONLY:
        _restore_learner_only(request, state)
        coordinator._initialize_metrics()
        return
    distributed = _validated_distributed_state(request, state)
    _restore_distributed(request, state, distributed)
    coordinator._initialize_metrics()


def _validate_checkpoint_state(request: _RestoreRequest, state: Mapping[str, Any]) -> None:
    _validate_checkpoint_keys(state)
    if state["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("async runtime requires distributed checkpoint schema 2.0")
    if state["journal_contract_version"] != _JOURNAL_CONTRACT_VERSION:
        raise ValueError("distributed checkpoint requires journal contract version 2")
    if state["run_fingerprint"] != request.coordinator.fingerprint:
        current = _coordinator_fingerprint_manifest(request.coordinator)
        details = fingerprint_mismatch_details(state.get("fingerprint_manifest"), current)
        raise ValueError(f"distributed checkpoint run fingerprint mismatch: {details}")
    _validate_checkpoint_components(state)
    training = getattr(request.coordinator.run.spec, "training", None)
    if (
        request.mode is ReplayRestoreMode.FULL
        and getattr(training, "evaluation_stop_min_finish_rate", None) is not None
        and not {"consecutive_evaluation_passes", "evaluation_stop_reason"}.issubset(
            state["distributed"]
        )
    ):
        raise ValueError(
            "legacy checkpoint lacks evaluation-stop progress; exact resume requires "
            "a checkpoint that records consecutive evaluation passes"
        )
    if request.mode is ReplayRestoreMode.FULL:
        _validate_evaluation_stop_state(state["distributed"], training)
    if (
        request.mode is ReplayRestoreMode.FULL
        and state["evaluated_policy_version"] is not None
        and ("pending_evaluation_promotions" not in state or "training_learner" not in state)
    ):
        raise ValueError(
            "legacy evaluated-policy checkpoint lacks full training state or "
            "promotion recovery evidence; "
            "use a full training checkpoint for exact resume"
        )


def _validate_checkpoint_keys(state: Mapping[str, Any]) -> None:
    missing = _CHECKPOINT_KEYS - state.keys()
    unexpected = (
        state.keys()
        - _CHECKPOINT_KEYS
        - {"fingerprint_manifest", "pending_evaluation_promotions", "training_learner"}
    )
    if missing or unexpected:
        raise ValueError(
            f"distributed checkpoint keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _validate_checkpoint_components(state: Mapping[str, Any]) -> None:
    for name in ("learner", "replay_store", "sampler", "distributed"):
        if not isinstance(state[name], Mapping):
            raise TypeError(f"distributed checkpoint {name} state must be a mapping")
    if "training_learner" in state and not isinstance(state["training_learner"], Mapping):
        raise TypeError("distributed checkpoint training_learner state must be a mapping")
    if not isinstance(state["journal_id"], str) or not state["journal_id"]:
        raise TypeError("distributed checkpoint journal_id must be a non-empty string")
    version = state["evaluated_policy_version"]
    if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
        raise TypeError("evaluated policy version must be an integer or null")


def _validate_evaluation_stop_state(distributed: Mapping[str, Any], training: Any) -> None:
    passes = distributed.get("consecutive_evaluation_passes", 0)
    reason = distributed.get("evaluation_stop_reason")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 0:
        raise ValueError("consecutive evaluation passes must be a nonnegative integer")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError("evaluation stop reason must be a nonempty string or null")
    required = getattr(training, "evaluation_stop_consecutive_batches", None)
    if required is not None and passes >= required and reason is None:
        raise ValueError("evaluation stop progress reached target without a stop reason")


def _restore_learner_only(request: _RestoreRequest, state: Mapping[str, Any]) -> None:
    coordinator = request.coordinator
    if coordinator.journal.has_history():
        raise RuntimeError(
            f"cannot reset replay while {coordinator.journal.path} contains rollout data; "
            "choose a new run_id so stale journal rows cannot enter a later resume"
        )
    coordinator.run.learner.load_state_dict(state["learner"])
    coordinator.counters = _Counters()


def _validated_distributed_state(
    request: _RestoreRequest, state: Mapping[str, Any]
) -> dict[str, Any]:
    distributed = dict(state["distributed"])
    _validate_distributed_keys(distributed)
    _validate_journal(request.coordinator, state, distributed)
    return distributed


def _validate_distributed_keys(distributed: Mapping[str, Any]) -> None:
    missing = _DISTRIBUTED_KEYS - distributed.keys()
    unexpected = distributed.keys() - _DISTRIBUTED_KEYS - _DISTRIBUTED_OPTIONAL_KEYS
    if missing or unexpected:
        raise ValueError(
            f"distributed runtime keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _validate_journal(
    coordinator: Coordinator, state: Mapping[str, Any], distributed: Mapping[str, Any]
) -> None:
    frontier = int(distributed["journal_applied_frontier"])
    try:
        coordinator.journal.validate_checkpoint(state["journal_id"], frontier)
    except Exception as exc:
        coordinator._log_wal_error("checkpoint_validation", exc)
        raise


def _restore_distributed(
    request: _RestoreRequest, state: Mapping[str, Any], distributed: dict[str, Any]
) -> None:
    coordinator = request.coordinator
    learner_state = (
        state["training_learner"]
        if state["evaluated_policy_version"] is not None
        else state["learner"]
    )
    coordinator.run.learner.load_state_dict(learner_state)
    coordinator._best_evaluation = _evaluation_rank(distributed.pop("best_evaluation_rank", None))
    coordinator._fastest_evaluation = _evaluation_rank(
        distributed.pop("fastest_evaluation_rank", None)
    )
    passes = distributed.pop("consecutive_evaluation_passes", 0)
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 0:
        raise ValueError("consecutive evaluation passes must be a nonnegative integer")
    reason = distributed.pop("evaluation_stop_reason", None)
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError("evaluation stop reason must be a nonempty string or null")
    coordinator._consecutive_evaluation_passes = passes
    coordinator._evaluation_stop_reason = reason
    coordinator.counters = _Counters(**distributed)
    if not coordinator.run.spec.distributed.strict_update_budget:
        coordinator.counters.update_credit = min(
            coordinator.counters.update_credit,
            float(coordinator.run.spec.distributed.max_update_credit),
        )
    load_state_dict(coordinator.run.replay_store, state["replay_store"])
    load_state_dict(coordinator.run.sampler, state["sampler"])
    _restore_pending_promotions(coordinator, state, coordinator.counters.journal_applied_frontier)
    coordinator._recover_journal(coordinator.counters.journal_applied_frontier)


def _restore_pending_promotions(
    coordinator: Coordinator, state: Mapping[str, Any], frontier: int
) -> None:
    entries = state.get("pending_evaluation_promotions", [])
    if not isinstance(entries, list):
        raise TypeError("pending evaluation promotions must be a list")
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError("invalid pending evaluation promotion entry")
        row_id, payload = entry
        if (
            isinstance(row_id, bool)
            or not isinstance(row_id, int)
            or not 0 < row_id <= frontier
            or not isinstance(payload, bytes)
            or len(payload) > coordinator.codec.max_message_bytes
        ):
            raise ValueError("invalid pending evaluation promotion evidence")
        coordinator.journal.stage_evaluation_promotion(row_id, payload)


def _evaluation_rank(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError("evaluation leader rank must contain three numbers")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise TypeError("evaluation leader rank must contain three numbers")
    rank = tuple(float(item) for item in value)
    if not all(map(math.isfinite, rank)):
        raise ValueError("evaluation leader rank must contain three finite numbers")
    return rank[0], rank[1], rank[2]
