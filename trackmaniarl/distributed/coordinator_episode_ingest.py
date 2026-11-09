from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from trackmaniarl.distributed import coordinator_delivery

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator


def ingest_episode_summaries(
    coordinator: Coordinator, value: Mapping[str, Any], row_id: int
) -> None:
    for summary in value["episodes"]:
        _ingest_episode_summary(coordinator, value, summary, row_id)


def _ingest_episode_summary(  # noqa: PLR0913
    coordinator: Coordinator, value: Mapping[str, Any], summary: Mapping[str, Any], row_id: int
) -> None:
    coordinator.counters.episodes += 1
    labeled = _record_episode_outcome(coordinator, summary)
    if not coordinator._recovering:
        logged = {**summary, "replay/labeled_transitions": labeled}
        coordinator._log_episode(value, logged)
    _schedule_training_evaluation(coordinator, value, row_id)


def _record_episode_outcome(coordinator: Coordinator, summary: Mapping[str, Any]) -> int:
    if not bool(summary["finished"]):
        return 0
    coordinator.counters.finishes += 1
    finish_time_s = float(summary["finish_time_s"])
    episode_id = str(summary["episode_id"])
    labeled = _label_episode_sampling_pace(coordinator, episode_id, finish_time_s)
    best = coordinator.counters.best_finish_time_s
    if best == 0.0 or finish_time_s < best:
        coordinator.counters.best_finish_time_s = finish_time_s
    return labeled


def _schedule_training_evaluation(
    coordinator: Coordinator, value: Mapping[str, Any], row_id: int
) -> None:
    interval = coordinator.run.spec.training.evaluate_every_episodes
    if interval is None or coordinator.counters.episodes % interval:
        return
    with coordinator._lock:
        if row_id <= coordinator._evaluation_schedule_frontier:
            return
        due = coordinator._evaluation_due | {str(value["actor_id"])}
        coordinator_delivery.save(
            coordinator, due, coordinator._evaluation_assignments, schedule_frontier=row_id
        )


def _label_episode_sampling_pace(
    coordinator: Coordinator, episode_id: str, finish_time_s: float
) -> int:
    if getattr(coordinator.run.sampler, "elite_time_s", None) is None:
        return 0
    label = getattr(coordinator.run.replay_store, "label_episode_sampling_pace", None)
    if not callable(label):
        return 0
    return int(label(episode_id, finish_time_s))
