from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from trackmaniarl.distributed.journal import JournalPayloadConflictError, RolloutJournal


def test_independent_connections_deduplicate_concurrent_append_and_recover(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite"
    journals = [RolloutJournal(path), RolloutJournal(path)]
    barrier = Barrier(2)

    def append(journal: RolloutJournal) -> tuple[int, bool]:
        barrier.wait(timeout=5)
        return journal.append("session", 0, b"payload")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(append, journals))
        assert {row for row, _ in results} == {1}
        assert sorted(created for _, created in results) == [False, True]
    finally:
        for journal in journals:
            journal.close()

    reopened = RolloutJournal(path)
    try:
        assert list(reopened.rows_after(0)) == [(1, b"payload")]
        assert reopened.append("session", 0, b"payload") == (1, False)
        with pytest.raises(JournalPayloadConflictError):
            reopened.append("session", 0, b"different")
    finally:
        reopened.close()
