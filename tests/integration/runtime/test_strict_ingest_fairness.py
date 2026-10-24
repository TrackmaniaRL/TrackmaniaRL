"""Durable accepted rows must wait for owed updates without being lost or reordered."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

from tests.integration.runtime.distributed_runtime_support import _DISTRIBUTED_TOKEN, _resolved_run
from trackmaniarl.core.data import Transition
from trackmaniarl.core.spec import DistributedSpec
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_rpc import _policy_response
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import PROTOCOL_VERSION, transition_to_wire


def _transition(number: int) -> Transition:
    return Transition(
        observation=np.array([number], dtype=np.float32),
        action=0,
        reward=float(number),
        next_observation=np.array([number + 1], dtype=np.float32),
        terminated=True,
        truncated=False,
        episode_id=f"actor/session/{number}",
        step=0,
    )


@pytest.fixture
def coordinator(tmp_path: Path) -> Iterator[Coordinator]:
    run = _resolved_run(
        tmp_path,
        "fairness",
        {
            "total_transitions": 100,
            "batch_size": 1,
            "warmup_transitions": 1,
            "updates_per_transition": 0.25,
            "checkpoint_interval_updates": None,
        },
    )
    spec = run.spec.model_copy(
        update={
            "distributed": DistributedSpec(
                strict_update_budget=True,
                rollout_chunk_transitions=1,
                max_update_credit=2,
            )
        }
    )
    result = Coordinator(
        replace(run, spec=spec), CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, "fingerprint")
    )
    result.run.replay_store.append(_transition(0))
    result.counters.transitions = 1
    result._last_heartbeats["actor"] = monotonic()
    try:
        yield result
    finally:
        result._checkpoint_writer.close()
        result.journal.close()


def _append(coordinator: Coordinator, count: int) -> None:
    for number in range(1, count + 1):
        payload = coordinator.codec.encode(
            {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": "fingerprint",
                "actor_id": "actor",
                "session_id": "session",
                "sequence": number,
                "policy_version": 0,
                "transitions": [transition_to_wire(_transition(number))],
                "episodes": [],
                "evaluations": [],
                "evaluation_snapshot": b"",
            }
        )
        row, inserted = coordinator.journal.append("session", number, payload)
        assert inserted
        coordinator._rollouts.put((row, monotonic()))


def test_owed_updates_preserve_durable_rows_and_release_submit_wakeups(
    coordinator: Coordinator,
) -> None:
    coordinator.counters.update_credit = 2.0
    _append(coordinator, 3)
    coordinator._drain_rollouts(64)
    assert coordinator._rollouts.empty()
    assert coordinator.journal.statistics(0).pending_rows == 3
    assert coordinator.counters.journal_applied_frontier == 0
    assert not _policy_response(coordinator, 0, -1)["collect_allowed"]
    coordinator.counters.update_credit = 0.0
    coordinator._drain_rollouts(64)
    assert coordinator.counters.journal_applied_frontier == 3
    assert coordinator.journal.statistics(3).pending_rows == 0
    assert [row.reward for row in coordinator.run.replay_store.get([0, 1, 2, 3])] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    assert coordinator.counters.update_credit == 0.75
    assert _policy_response(coordinator, 0, -1)["collect_allowed"]


def test_burst_yields_as_soon_as_an_update_is_owed_and_keeps_exact_credit(
    coordinator: Coordinator,
) -> None:
    _append(coordinator, 8)
    coordinator._drain_rollouts(64)
    assert coordinator.counters.journal_applied_frontier == 4
    assert coordinator.counters.update_credit == 1.0
    assert coordinator.journal.statistics(4).pending_rows == 4
    coordinator.counters.update_credit -= 1
    coordinator.counters.updates += 1
    coordinator._drain_rollouts(64)
    assert coordinator.counters.journal_applied_frontier == 8
    coordinator.counters.update_credit -= 1
    coordinator.counters.updates += 1
    assert coordinator.counters.updates == 8 * 0.25
    assert coordinator.counters.update_credit == 0.0
    assert [row.reward for row in coordinator.run.replay_store.get(list(range(9)))] == list(
        range(9)
    )


def test_no_active_learner_still_drains_accepted_rows_for_shutdown(
    coordinator: Coordinator,
) -> None:
    _append(coordinator, 3)
    coordinator.counters.update_credit = 2.0
    coordinator._last_heartbeats.clear()
    coordinator._drain_rollouts(64)
    assert coordinator.counters.journal_applied_frontier == 3
    assert coordinator.counters.update_credit == 2.75


def test_pending_durable_rows_participate_in_collection_backpressure(
    coordinator: Coordinator,
) -> None:
    _append(coordinator, 8)
    # No data has been ingested yet, but all eight accepted rows are owed work.
    assert coordinator.counters.update_credit == 0.0
    assert not _policy_response(coordinator, 0, -1)["collect_allowed"]
    coordinator._drain_rollouts(64)
    assert coordinator.counters.update_credit == 1.0
    assert not _policy_response(coordinator, 0, -1)["collect_allowed"]
