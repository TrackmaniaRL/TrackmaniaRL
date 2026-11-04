"""Exercise FULL restore from a copied journal with a durable unprocessed tail."""

import shutil
import sqlite3
from pathlib import Path

from tests.integration.runtime.test_distributed_wal import (
    _append_journal_transitions,
    _close,
    _coordinator,
)
from tools.headless_campaign_queue.recovery import inspect_journal, snapshot_journal


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
