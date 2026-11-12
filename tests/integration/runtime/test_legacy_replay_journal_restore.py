"""Restore synthetic artifacts produced by the pre-refactor source revision."""

from __future__ import annotations

import sqlite3
import zlib
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from tests.integration.runtime.distributed_runtime_support import _DISTRIBUTED_TOKEN, _resolved_run
from trackmaniarl.core.data import BatchRequest
from trackmaniarl.core.replay.sequence_sampler import SequenceSampler
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "replay_legacy"
_IDENTITY = "synthetic-fixture-fingerprint"
_CASES = [
    ("journal-v1-536b42b5.bin", "checkpoint-v2-536b42b5.bin"),
    ("journal-v1-v130-eb803a21.bin", "checkpoint-v130-eb803a21.bin"),
]


def _restore_case(
    tmp_path: Path, fingerprint: str, fixture: tuple[str, str]
) -> tuple[Coordinator, Path, list[tuple]]:
    journal_file, checkpoint_file = fixture
    run = _resolved_run(
        tmp_path,
        "fixture",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    run = replace(run, sampler=SequenceSampler(run.feature_pipeline, sequence_length=2, seed=1))
    journal = run.run_dir / "distributed" / "rollouts.sqlite3"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(zlib.decompress((_FIXTURES / journal_file).read_bytes()))
    with closing(sqlite3.connect(journal)) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone() == ("1",)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not row[0].startswith("sqlite_")
        }
        assert tables == {"chunks", "actor_profiles", "receipts", "metadata"}
        receipts = connection.execute(
            "SELECT session_id, sequence, row_id, payload_sha256 FROM receipts ORDER BY sequence"
        ).fetchall()
    checkpoint = tmp_path / "legacy-checkpoint.pt"
    checkpoint.write_bytes((_FIXTURES / checkpoint_file).read_bytes())
    coordinator = Coordinator(
        run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, fingerprint)
    )
    return coordinator, checkpoint, receipts


@pytest.mark.parametrize("fixture", _CASES)
def test_legacy_checkpoint_and_v1_journal_recover_and_sample(
    tmp_path: Path, fixture: tuple[str, str]
) -> None:
    coordinator, checkpoint, legacy_receipts = _restore_case(tmp_path, _IDENTITY, fixture)
    try:
        coordinator.restore_checkpoint(checkpoint)
        assert coordinator.run.learner.value == 11
        assert coordinator.counters.transitions == 3
        assert coordinator.counters.journal_applied_frontier == 2
        assert [item.step for item in coordinator.run.replay_store.get([0, 1, 2])] == [0, 1, 2]
        original_sampler = coordinator.run.checkpoint_codec.load(checkpoint)["sampler"]
        assert coordinator.run.sampler.state_dict() == original_sampler
        batch = coordinator.run.sampler.sample(
            coordinator.run.replay_store, BatchRequest(batch_size=1, sequence_length=2)
        )
        assert tuple(batch.transition_ids) == (1, 2)
        # A repeated recovery from the persisted frontier must not duplicate a row.
        coordinator._recover_journal(coordinator.counters.journal_applied_frontier)
        assert coordinator.counters.transitions == 3
        assert len(coordinator.run.replay_store) == 3
        # Both pre-checkpoint and later durable receipts survive migration.
        with coordinator.journal._lock:
            assert coordinator.journal._connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone() == ("2",)
            receipts = coordinator.journal._connection.execute(
                "SELECT session_id, sequence, row_id, payload_sha256 FROM receipts "
                "ORDER BY sequence"
            ).fetchall()
        assert [(row[0], row[1], row[2]) for row in receipts] == [
            ("fixture-session", 0, 1),
            ("fixture-session", 1, 2),
        ]
        assert receipts == legacy_receipts
        assert coordinator.journal.has_receipt("fixture-session", 1, _journal_payload(coordinator))
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def _journal_payload(coordinator: Coordinator) -> bytes:
    return next(iter(coordinator.journal.rows_after(1)))[1]


@pytest.mark.parametrize("fixture", _CASES)
def test_legacy_checkpoint_still_rejects_a_different_run_identity(
    tmp_path: Path, fixture: tuple[str, str]
) -> None:
    coordinator, checkpoint, _ = _restore_case(tmp_path, "different-fingerprint", fixture)
    try:
        with pytest.raises(ValueError, match="run fingerprint mismatch"):
            coordinator.restore_checkpoint(checkpoint)
        assert len(coordinator.run.replay_store) == 0
        assert coordinator.run.learner.value == 0
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()
