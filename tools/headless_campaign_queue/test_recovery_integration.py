"""Exercise FULL restore from a copied journal with a durable unprocessed tail."""

import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.runtime.distributed_runtime_support import _resolved_run
from tests.integration.runtime.test_distributed_wal import (
    _append_journal_transitions,
    _close,
    _coordinator,
)
from tools.headless_campaign_queue.recovery import (
    finish_replayed_coordinator,
    inspect_journal,
    snapshot_journal,
)
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig


def test_isolated_full_restore_replays_wal_tail_once(tmp_path: Path) -> None:
    original = _coordinator(tmp_path / "original", "interrupted", total=10)
    try:
        _append_journal_transitions(original, 2)
        original._drain_rollouts(1)
        checkpoint = original._checkpoint()
        original._checkpoint_writer.wait()
        journal = original.journal.path
        assert original.counters.journal_applied_frontier == 1
    finally:
        _close(original)
    snapshot = tmp_path / "recovery" / "interrupted" / "distributed" / "rollouts.sqlite3"
    copied_checkpoint = tmp_path / "recovery" / "copied.pt"
    snapshot_journal(journal, snapshot)
    shutil.copyfile(checkpoint, copied_checkpoint)
    with sqlite3.connect(snapshot) as db:
        journal_id = dict(db.execute("SELECT key,value FROM metadata"))["journal_id"]
    assert inspect_journal(snapshot, journal_id, 1)["pending_ids"] == [2]
    resumed = _coordinator(tmp_path / "recovery", "interrupted", total=10)
    try:
        resumed.restore_checkpoint(copied_checkpoint)
        assert resumed.counters.transitions == 2
        assert resumed.counters.journal_applied_frontier == 2
        assert [row.step for row in resumed.run.replay_store.get([0, 1])] == [0, 1]
        resumed.restore_checkpoint(copied_checkpoint)
        assert resumed.counters.transitions == 2
        assert len(resumed.run.replay_store) == 2
    finally:
        _close(resumed)
    assert inspect_journal(journal, journal_id, 1)["pending_ids"] == [2]


def test_replayed_budget_drains_credit_and_saves_full_checkpoint_without_touching_source(
    tmp_path: Path,
) -> None:
    def make_coordinator(root):
        run = _resolved_run(
            root,
            "wal-final",
            {
                "total_transitions": 2,
                "warmup_transitions": 0,
                "batch_size": 1,
                "sequence_length": 1,
                "n_step": 1,
                "checkpoint_interval_updates": None,
            },
        )
        return Coordinator(
            run,
            CoordinatorConfig("127.0.0.1:0", "test-token-123456789012345678901234", "fingerprint"),
        )

    original = make_coordinator(tmp_path / "original")
    try:
        _append_journal_transitions(original, 2)
        original._drain_rollouts(1)
        checkpoint = original._checkpoint()
        original._checkpoint_writer.wait()
        journal = original.journal.path
    finally:
        _close(original)
    source_checkpoint = checkpoint.read_bytes()
    snapshot = tmp_path / "recovery" / "wal-final" / "distributed" / "rollouts.sqlite3"
    snapshot_journal(journal, snapshot)
    copied = tmp_path / "recovery" / "source.pt"
    shutil.copyfile(checkpoint, copied)
    resumed = make_coordinator(tmp_path / "recovery")
    try:
        resumed.restore_checkpoint(copied)
        assert resumed.counters.transitions == 2
        assert resumed.counters.update_credit == 2
        with pytest.raises(ValueError, match="outside accepted budget"):
            finish_replayed_coordinator(resumed, 1, 0)
        assert resumed.counters.update_credit == 2
        resumed.external_stop = SimpleNamespace(is_set=lambda: True)
        with pytest.raises(ValueError, match="stopped"):
            finish_replayed_coordinator(resumed, 2, 0)
        assert resumed.counters.update_credit == 2
        resumed.external_stop = None
        final = finish_replayed_coordinator(resumed, 2, 0)
        state = resumed.run.checkpoint_codec.load(final)
        assert state["distributed"]["transitions"] == 2
        assert state["distributed"]["updates"] == 2
        assert state["distributed"]["update_credit"] == 0
        assert state["learner"]["value"] == 2
        assert "replay_store" in state and "sampler" in state
    finally:
        _close(resumed)
    assert checkpoint.read_bytes() == source_checkpoint
    with sqlite3.connect(journal) as db:
        journal_id = dict(db.execute("SELECT key,value FROM metadata"))["journal_id"]
    assert inspect_journal(journal, journal_id, 1)["pending_ids"] == [2]
