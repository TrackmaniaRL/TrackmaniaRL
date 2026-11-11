"""Durable delivery state for actor evaluation assignments."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator


def load(coordinator: Coordinator) -> None:
    delivery = coordinator.journal.evaluation_delivery()
    if delivery is not None:
        (
            coordinator._evaluation_due,
            coordinator._evaluation_assignments,
            coordinator._evaluation_schedule_frontier,
        ) = delivery
        return
    raw = coordinator.journal.evaluation_state()
    if raw is None:
        coordinator._evaluation_due = set()
        coordinator._evaluation_assignments = {}
        coordinator._evaluation_schedule_frontier = 0
        return
    state = json.loads(raw)
    if state.get("version") != 1:
        raise ValueError("unsupported evaluation delivery state")
    coordinator._evaluation_due = set(state["due"])
    coordinator._evaluation_schedule_frontier = int(state.get("schedule_frontier", 0))
    coordinator._evaluation_assignments = {
        actor: {
            **assignment,
            "snapshot": base64.b64decode(assignment["snapshot"], validate=True),
        }
        for actor, assignment in state["assignments"].items()
    }
    coordinator.journal.save_evaluation_delivery(
        coordinator._evaluation_due,
        coordinator._evaluation_assignments,
        coordinator._evaluation_schedule_frontier,
        set(),
        {},
    )


def save(  # noqa: PLR0913
    coordinator: Coordinator,
    due: set[str],
    assignments: dict[str, dict[str, Any]],
    *,
    schedule_frontier: int | None = None,
) -> None:
    frontier = (
        coordinator._evaluation_schedule_frontier
        if schedule_frontier is None
        else schedule_frontier
    )
    coordinator.journal.save_evaluation_delivery(
        due, assignments, frontier, coordinator._evaluation_due, coordinator._evaluation_assignments
    )
    coordinator._evaluation_due = due
    coordinator._evaluation_assignments = assignments
    coordinator._evaluation_schedule_frontier = frontier
