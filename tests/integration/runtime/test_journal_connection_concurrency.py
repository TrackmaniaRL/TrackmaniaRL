from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from trackmaniarl.distributed.journal import JournalPayloadConflictError, RolloutJournal


def test_independent_connections_can_initialize_the_same_new_journal(tmp_path: Path) -> None:
    path = tmp_path / "new-journal.sqlite"
    barrier = Barrier(2)

    def open_and_append(index: int) -> str:
        barrier.wait(timeout=5)
        journal = RolloutJournal(path)
        try:
            journal.append(f"session-{index}", 0, b"payload")
            return journal.identity
        finally:
            journal.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = list(pool.map(open_and_append, range(2)))
    assert identities[0] == identities[1]
    journal = RolloutJournal(path)
    try:
        assert len(list(journal.rows_after(0))) == 2
    finally:
        journal.close()


def test_independent_connections_can_migrate_the_same_legacy_journal(tmp_path: Path) -> None:
    path = tmp_path / "legacy-journal.sqlite"
    journal = RolloutJournal(path)
    identity = journal.identity
    journal.append("legacy", 0, b"payload")
    journal.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE evaluation_assignments")
        connection.execute("DROP TABLE evaluation_snapshots")
        connection.execute("DROP TABLE evaluation_due")
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")

    barrier = Barrier(2)

    def reopen(_: int) -> tuple[str, list[tuple[int, bytes]]]:
        barrier.wait(timeout=5)
        opened = RolloutJournal(path)
        try:
            return opened.identity, list(opened.rows_after(0))
        finally:
            opened.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reopen, range(2)))
    assert results == [(identity, [(1, b"payload")])] * 2


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
