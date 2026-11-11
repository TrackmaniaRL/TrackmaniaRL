from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trackmaniarl.distributed.coordinator_checkpoint import (
    EvaluatedPolicyCheckpoint,
    EvaluationCheckpointKind,
)

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")

type EvaluationRank = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    finish_rate: float
    finished_trials: int
    trials: int
    best_time_s: float
    median_time_s: float
    mean_time_s: float
    policy_version: int
    max_step_race_time_ms: float = 0.0
    step_race_time_measurements_valid: bool = False
    step_race_time_measurement_count: int = 0
    step_race_time_expected_measurement_count: int = 0


@dataclass(frozen=True, slots=True)
class _LeaderPromotions:
    reliable: EvaluationRank | None
    fastest: EvaluationRank | None


@dataclass(frozen=True, slots=True)
class _LeaderReservation:
    promotions: _LeaderPromotions
    previous: tuple[EvaluationRank | None, EvaluationRank | None]


@dataclass(frozen=True, slots=True)
class _LeaderCheckpoint:
    candidate: EvaluationCandidate
    policy_state: Mapping[str, Any]
    kind: EvaluationCheckpointKind
    update: int


def candidate_from_stats(stats: Mapping[str, float]) -> EvaluationCandidate:
    return EvaluationCandidate(
        finish_rate=float(stats["finish_rate"]),
        finished_trials=int(stats["finished_trials"]),
        trials=int(stats["trials"]),
        best_time_s=float(stats["finish_time_best_s"]),
        median_time_s=float(stats["finish_time_median_s"]),
        mean_time_s=float(stats["finish_time_mean_s"]),
        policy_version=int(stats["policy_version"]),
        max_step_race_time_ms=float(stats["step_race_time_ms_max"]),
        step_race_time_measurements_valid=(
            float(stats.get("step_race_time_measurements_valid", 0.0)) == 1.0
        ),
        step_race_time_measurement_count=int(stats.get("step_race_time_measurement_count", 0.0)),
        step_race_time_expected_measurement_count=int(
            stats.get("step_race_time_expected_measurement_count", 0.0)
        ),
    )


def record_evaluation_leaders(
    coordinator: Coordinator, candidate: EvaluationCandidate, *, allow_recovery: bool = False
) -> None:
    coordinator._checkpoint_writer.wait()
    promotions = _leader_promotions(coordinator, candidate, allow_recovery=allow_recovery)
    if not _has_promotions(promotions):
        return
    policy_state = _evaluated_policy_state(coordinator, candidate.policy_version)
    for kind in _promotion_kinds(promotions):
        coordinator._checkpoint_writer.wait()
        promotion = _promotion_for_kind(promotions, kind)
        reservation = _leader_reservation(coordinator, promotion)
        _reserve_promotions(coordinator, promotion)
        request = _LeaderCheckpoint(candidate, policy_state, kind, coordinator.counters.updates)
        _save_leader(coordinator, request, reservation)


def _has_promotions(promotions: _LeaderPromotions) -> bool:
    return promotions.reliable is not None or promotions.fastest is not None


def _leader_reservation(
    coordinator: Coordinator, promotions: _LeaderPromotions
) -> _LeaderReservation:
    previous = coordinator._best_evaluation, coordinator._fastest_evaluation
    return _LeaderReservation(promotions, previous)


def _promotion_kinds(promotions: _LeaderPromotions) -> tuple[EvaluationCheckpointKind, ...]:
    kinds: list[EvaluationCheckpointKind] = []
    if promotions.reliable is not None:
        kinds.append(EvaluationCheckpointKind.RELIABLE)
    if promotions.fastest is not None:
        kinds.append(EvaluationCheckpointKind.FASTEST)
    return tuple(kinds)


def _promotion_for_kind(
    promotions: _LeaderPromotions, kind: EvaluationCheckpointKind
) -> _LeaderPromotions:
    if kind is EvaluationCheckpointKind.RELIABLE:
        return _LeaderPromotions(promotions.reliable, None)
    return _LeaderPromotions(None, promotions.fastest)


def _leader_promotions(
    coordinator: Coordinator, candidate: EvaluationCandidate, *, allow_recovery: bool = False
) -> _LeaderPromotions:
    if coordinator._recovering and not allow_recovery:
        return _LeaderPromotions(None, None)
    reliable = _reliable_rank(coordinator, candidate)
    fastest = _fastest_rank(candidate)
    promote_reliable = _reliable_qualified(coordinator, candidate) and _improves(
        reliable, coordinator._best_evaluation
    )
    promote_fastest = candidate.finished_trials > 0 and _improves(
        fastest, coordinator._fastest_evaluation
    )
    return _LeaderPromotions(
        reliable if promote_reliable else None,
        fastest if promote_fastest else None,
    )


def _required_finish_rate(coordinator: Coordinator) -> float:
    suite = coordinator.run.spec.evaluation
    return 1.0 if suite is None else suite.min_finish_rate


def _target_mean(coordinator: Coordinator) -> float | None:
    suite = coordinator.run.spec.evaluation
    return None if suite is None else getattr(suite, "target_mean_s", None)


def _maximum_step_target(coordinator: Coordinator) -> float | None:
    suite = coordinator.run.spec.evaluation
    return None if suite is None else getattr(suite, "max_step_race_time_ms", None)


def _reliable_qualified(coordinator: Coordinator, candidate: EvaluationCandidate) -> bool:
    if candidate.finished_trials < 1 or any(
        not isfinite(value) or value <= 0.0
        for value in (candidate.best_time_s, candidate.median_time_s, candidate.mean_time_s)
    ):
        return False
    if candidate.finish_rate < _required_finish_rate(coordinator):
        return False
    target = _target_mean(coordinator)
    if target is not None and candidate.mean_time_s >= target:
        return False
    maximum_step_target = _maximum_step_target(coordinator)
    return maximum_step_target is None or (
        candidate.step_race_time_measurements_valid
        and candidate.step_race_time_measurement_count
        == candidate.step_race_time_expected_measurement_count
        and candidate.step_race_time_expected_measurement_count > 0
        and 0.0 < candidate.max_step_race_time_ms <= maximum_step_target
    )


def _reliable_rank(coordinator: Coordinator, candidate: EvaluationCandidate) -> EvaluationRank:
    if _target_mean(coordinator) is not None:
        return candidate.finish_rate, -candidate.mean_time_s, -candidate.median_time_s
    return candidate.finish_rate, -candidate.median_time_s, -candidate.best_time_s


def _fastest_rank(candidate: EvaluationCandidate) -> EvaluationRank:
    return -candidate.best_time_s, candidate.finish_rate, -candidate.median_time_s


def _improves(rank: EvaluationRank, current: EvaluationRank | None) -> bool:
    return current is None or rank > current


def _evaluated_policy_state(coordinator: Coordinator, policy_version: int) -> Mapping[str, Any]:
    with coordinator._lock:
        state = coordinator._evaluation_policy_states.get(policy_version)
    if state is None:
        raise RuntimeError(
            f"cannot checkpoint evaluated policy version {policy_version}: exact snapshot missing"
        )
    return state


def _reserve_promotions(coordinator: Coordinator, promotions: _LeaderPromotions) -> None:
    with coordinator._lock:
        if promotions.reliable is not None:
            coordinator._best_evaluation = promotions.reliable
        if promotions.fastest is not None:
            coordinator._fastest_evaluation = promotions.fastest


def _rollback_callback(
    coordinator: Coordinator,
    reservation: _LeaderReservation,
    kind: EvaluationCheckpointKind,
) -> Callable[[BaseException], None]:
    def rollback(exc: BaseException) -> None:
        del exc
        with coordinator._lock:
            _rollback_reserved_rank(coordinator, reservation, kind)

    return rollback


def _rollback_reserved_rank(
    coordinator: Coordinator,
    reservation: _LeaderReservation,
    kind: EvaluationCheckpointKind,
) -> None:
    promotions = reservation.promotions
    if (
        kind is EvaluationCheckpointKind.RELIABLE
        and coordinator._best_evaluation == promotions.reliable
    ):
        coordinator._best_evaluation = reservation.previous[0]
    if (
        kind is EvaluationCheckpointKind.FASTEST
        and coordinator._fastest_evaluation == promotions.fastest
    ):
        coordinator._fastest_evaluation = reservation.previous[1]


def _save_leader(
    coordinator: Coordinator,
    request: _LeaderCheckpoint,
    reservation: _LeaderReservation,
) -> None:
    saved = _saved_callback(coordinator, request)
    rollback = _rollback_callback(coordinator, reservation, request.kind)
    evaluated = EvaluatedPolicyCheckpoint(
        request.policy_state,
        request.candidate.policy_version,
        request.kind,
        saved,
        rollback,
        suppress_events=coordinator._recovering,
    )
    try:
        coordinator._checkpoint(evaluated)
    except BaseException as exc:
        rollback(exc)
        raise


def _saved_callback(coordinator: Coordinator, request: _LeaderCheckpoint) -> Callable[[Path], None]:
    recovering = coordinator._recovering

    def saved(path: Path) -> None:
        coordinator._checkpoints.append(path)
        if not recovering:
            _log_leader_checkpoint(coordinator, request, path)

    return saved


def _candidate_metrics(candidate: EvaluationCandidate) -> dict[str, float | int]:
    return {
        "finish_rate": candidate.finish_rate,
        "finished_trials": candidate.finished_trials,
        "trials": candidate.trials,
        "finish_time_best_s": candidate.best_time_s,
        "finish_time_median_s": candidate.median_time_s,
        "finish_time_mean_s": candidate.mean_time_s,
        "policy_version": candidate.policy_version,
        "step_race_time_ms_max": candidate.max_step_race_time_ms,
        "step_race_time_measurements_valid": float(candidate.step_race_time_measurements_valid),
        "step_race_time_measurement_count": candidate.step_race_time_measurement_count,
        "step_race_time_expected_measurement_count": (
            candidate.step_race_time_expected_measurement_count
        ),
    }


def _log_leader_checkpoint(
    coordinator: Coordinator,
    request: _LeaderCheckpoint,
    path: Path,
) -> None:
    event = _leader_event(request.kind)
    payload = _leader_payload(coordinator, request, path)
    coordinator.run.logger.log(event, payload, step=request.update)
    _log_leader_info(request, path)


def _leader_event(kind: EvaluationCheckpointKind) -> str:
    if kind is EvaluationCheckpointKind.RELIABLE:
        return "eval/best_checkpoint"
    return "eval/fastest_checkpoint"


def _leader_payload(
    coordinator: Coordinator, request: _LeaderCheckpoint, path: Path
) -> dict[str, float | int | str]:
    candidate = request.candidate
    payload: dict[str, float | int | str] = {
        **_candidate_metrics(candidate),
        "exact_policy": 1.0,
        "path": str(path),
    }
    if request.kind is EvaluationCheckpointKind.RELIABLE:
        payload["release_qualified"] = 1.0
        return payload
    payload["reliable_qualified"] = float(_reliable_qualified(coordinator, candidate))
    payload["shared_with_reliable"] = 0.0
    return payload


def _log_leader_info(request: _LeaderCheckpoint, path: Path) -> None:
    candidate = request.candidate
    logger.info(
        "%s deterministic policy @update %d: best=%.2fs, %d/%d finished, checkpoint=%s",
        "Reliable" if request.kind is EvaluationCheckpointKind.RELIABLE else "Fastest",
        request.update,
        candidate.best_time_s,
        candidate.finished_trials,
        candidate.trials,
        path,
    )
