from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.integration.runtime.distributed_evaluation_support import (
    _evaluation_coordinator,
    _finished_evaluation,
)
from trackmaniarl.distributed import coordinator_ingest, coordinator_leaders
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.journal import RolloutJournal
from trackmaniarl.distributed.protocol import PROTOCOL_VERSION


def _payload(codec: WireCodec) -> dict:
    summaries = [
        {**_finished_evaluation(40.0), "evaluation_assignment_id": "assignment"} for _ in range(2)
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": "actor",
        "session_id": "session",
        "sequence": 0,
        "policy_version": 41,
        "transitions": [],
        "episodes": [],
        "evaluations": summaries,
        "evaluation_snapshot": codec.encode({"weight": 1.0}),
    }


def test_legacy_wal_without_verdict_retained_and_rejected_for_suite(tmp_path: Path) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    coordinator.codec = WireCodec(1024 * 1024)
    coordinator.journal = RolloutJournal(tmp_path / "wal.sqlite")
    payload = _payload(coordinator.codec)
    encoded = coordinator.codec.encode(payload)
    try:
        row_id, _ = coordinator.journal.append("session", 0, encoded)
        coordinator._recovering = True
        with pytest.raises(ValueError, match="lacks a persisted admission verdict"):
            coordinator_ingest._recover_rows(coordinator, 0)
        assert list(coordinator.journal.rows_after(0)) == [(row_id, encoded)]
        assert coordinator.journal.evaluation_row_accepted(row_id) is None
        assert coordinator.counters.journal_applied_frontier == 0
        assert coordinator.counters.evaluations == 0
    finally:
        coordinator.journal.close()


@pytest.mark.parametrize("admission", [None, False])
def test_unverified_pending_promotion_is_retained_for_suite(
    tmp_path: Path, *, admission: bool | None
) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    coordinator.codec = WireCodec(1024 * 1024)
    coordinator.journal = RolloutJournal(tmp_path / "wal.sqlite")
    payload = _payload(coordinator.codec)
    encoded = coordinator.codec.encode(payload)
    metadata = None if admission is None else ("actor", "assignment", admission, 0)
    try:
        row_id, _ = coordinator.journal.append("session", 0, encoded, evaluation=metadata)
        evidence = coordinator.codec.encode(
            {"summaries": payload["evaluations"], "snapshot": payload["evaluation_snapshot"]}
        )
        coordinator.journal.stage_evaluation_promotion(row_id, evidence)
        with pytest.raises(ValueError, match="lacks a persisted accepted admission verdict"):
            coordinator_ingest._recover_pending_promotions(coordinator)
        assert coordinator.journal.pending_evaluation_promotions() == [(row_id, evidence)]
        assert coordinator.counters.evaluations == 0
    finally:
        coordinator.journal.close()


@pytest.mark.parametrize(
    ("suite", "admission", "prune"), [(True, True, True), (False, None, False)]
)
def test_pending_promotion_recovers_with_verdict_or_legacy_no_suite(  # noqa: PLR0913
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    suite: bool,
    admission: bool | None,
    prune: bool,
) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    if not suite:
        coordinator.run.spec.evaluation = None
    coordinator.codec = WireCodec(1024 * 1024)
    coordinator.journal = RolloutJournal(tmp_path / "wal.sqlite")
    promoted = Mock()
    monkeypatch.setattr(coordinator_leaders, "record_evaluation_leaders", promoted)
    payload = _payload(coordinator.codec)
    metadata = None if admission is None else ("actor", "assignment", admission, 0)
    try:
        row_id, _ = coordinator.journal.append(
            "session", 0, coordinator.codec.encode(payload), evaluation=metadata
        )
        evidence = coordinator.codec.encode(
            {"summaries": payload["evaluations"], "snapshot": payload["evaluation_snapshot"]}
        )
        coordinator.journal.stage_evaluation_promotion(row_id, evidence)
        if prune:
            coordinator.journal.prune(row_id)
            assert coordinator.journal.evaluation_row_accepted(row_id) is True
        coordinator_ingest._recover_pending_promotions(coordinator)
        promoted.assert_called_once()
        assert coordinator.journal.pending_evaluation_promotions() == []
    finally:
        coordinator.journal.close()


def test_legacy_wal_without_verdict_recovers_without_suite(tmp_path: Path) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    coordinator.run.spec.evaluation = None
    coordinator.codec = WireCodec(1024 * 1024)
    coordinator.journal = RolloutJournal(tmp_path / "wal.sqlite")
    payload = _payload(coordinator.codec)
    try:
        row_id, _ = coordinator.journal.append("session", 0, coordinator.codec.encode(payload))
        coordinator._recovering = True
        assert coordinator_ingest._recover_rows(coordinator, 0) == (1, 0)
        assert coordinator.counters.journal_applied_frontier == row_id
        assert coordinator.counters.evaluations == 2
    finally:
        coordinator.journal.close()


def test_wal_with_persisted_acceptance_recovers_for_suite(tmp_path: Path) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    coordinator.codec = WireCodec(1024 * 1024)
    coordinator.journal = RolloutJournal(tmp_path / "wal.sqlite")
    payload = _payload(coordinator.codec)
    try:
        row_id, _ = coordinator.journal.append(
            "session",
            0,
            coordinator.codec.encode(payload),
            evaluation=("actor", "assignment", True, 0),
        )
        coordinator._recovering = True
        assert coordinator_ingest._recover_rows(coordinator, 0) == (1, 0)
        assert coordinator.counters.journal_applied_frontier == row_id
        assert coordinator.counters.evaluations == 2
    finally:
        coordinator.journal.close()
