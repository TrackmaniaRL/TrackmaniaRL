from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from trackmaniarl.distributed import coordinator_delivery
from trackmaniarl.distributed import journal as journal_module
from trackmaniarl.distributed.journal import JournalSchemaError, RolloutJournal


def _coordinator(journal: RolloutJournal) -> SimpleNamespace:
    result = SimpleNamespace(journal=journal)
    coordinator_delivery.load(result)
    return result


def _assignment(snapshot: bytes, sequence: int = 0) -> dict:
    return {
        "session": "s",
        "sequence": sequence,
        "assigned_row_id": 5,
        "version": 3,
        "snapshot": snapshot,
        "retired_sessions": [],
    }


def test_migrate_legacy_delivery_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    journal = RolloutJournal(path)
    journal.close()
    snapshot = b"legacy data"
    state = {
        "version": 1,
        "due": ["a"],
        "schedule_frontier": 8,
        "assignments": {
            "b": {**_assignment(snapshot), "snapshot": base64.b64encode(snapshot).decode()}
        },
    }
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE evaluation_assignments")
        connection.execute("DROP TABLE evaluation_snapshots")
        connection.execute("DROP TABLE evaluation_due")
        connection.execute("UPDATE metadata SET value = '1' WHERE key = 'schema_version'")
        connection.execute(
            "INSERT INTO metadata VALUES ('evaluation_delivery', ?)", (json.dumps(state),)
        )
    journal = RolloutJournal(path)
    coordinator = _coordinator(journal)
    assert coordinator._evaluation_due == {"a"}
    assert coordinator._evaluation_assignments == {"b": _assignment(snapshot)}
    assert coordinator._evaluation_schedule_frontier == 8
    assert journal.evaluation_state() is None
    journal.close()
    reopened = RolloutJournal(path)
    assert _coordinator(reopened)._evaluation_assignments == {"b": _assignment(snapshot)}
    reopened.close()


def test_shared_snapshots_incremental_updates_and_rollback(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    journal = RolloutJournal(path)
    coordinator = _coordinator(journal)
    snapshot = b"snapshot"
    first = {"a": _assignment(snapshot), "b": _assignment(snapshot)}
    coordinator_delivery.save(coordinator, {"c"}, first, schedule_frontier=9)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0] == 1
    updated = {"b": _assignment(snapshot, sequence=2)}
    coordinator_delivery.save(coordinator, set(), updated, schedule_frontier=10)
    broken = {**updated, "bad": {**_assignment(b"other"), "sequence": object()}}
    with pytest.raises(TypeError):
        coordinator_delivery.save(coordinator, {"d"}, broken, schedule_frontier=11)
    journal.close()
    reopened = RolloutJournal(path)
    restored = _coordinator(reopened)
    assert restored._evaluation_due == set()
    assert restored._evaluation_assignments == updated
    assert restored._evaluation_schedule_frontier == 10
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0] == 1
    reopened.close()


def test_schema_migration_rolls_back_ddl_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.db"
    journal = RolloutJournal(path)
    journal.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE evaluation_assignments")
        connection.execute("DROP TABLE evaluation_snapshots")
        connection.execute("DROP TABLE evaluation_due")
        connection.execute("UPDATE metadata SET value = '1' WHERE key = 'schema_version'")
    schema = journal_module._EVALUATION_SCHEMA
    with monkeypatch.context() as context:
        context.setattr(journal_module, "_EVALUATION_SCHEMA", (schema[0], "INVALID SQL"))
        with pytest.raises(sqlite3.OperationalError):
            RolloutJournal(path)
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "evaluation_due" not in tables
        assert (
            connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
            == "1"
        )
    reopened = RolloutJournal(path)
    assert _coordinator(reopened)._evaluation_assignments == {}
    reopened.close()


@pytest.mark.parametrize("failure_phase", ["ddl", "metadata"])
def test_new_journal_creation_rolls_back_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    path = tmp_path / "journal.db"
    with monkeypatch.context() as context:
        if failure_phase == "ddl":
            schema = journal_module._EVALUATION_SCHEMA
            context.setattr(journal_module, "_EVALUATION_SCHEMA", (schema[0], "INVALID SQL"))
        else:
            original = RolloutJournal._initialize_metadata

            def fail_after_metadata(journal: RolloutJournal) -> None:
                original(journal)
                raise OSError("simulated metadata failure")

            context.setattr(RolloutJournal, "_initialize_metadata", fail_after_metadata)
        with pytest.raises((sqlite3.OperationalError, OSError)):
            RolloutJournal(path)
    with sqlite3.connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert tables == []
    reopened = RolloutJournal(path)
    assert reopened.identity
    assert _coordinator(reopened)._evaluation_assignments == {}
    reopened.close()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_corrupt_or_missing_snapshot_is_rejected(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "journal.db"
    journal = RolloutJournal(path)
    coordinator = _coordinator(journal)
    coordinator_delivery.save(coordinator, set(), {"a": _assignment(b"good")})
    journal.close()
    with sqlite3.connect(path) as connection:
        if damage == "missing":
            connection.execute("DELETE FROM evaluation_snapshots")
        else:
            connection.execute("UPDATE evaluation_snapshots SET payload = ?", (b"bad",))
    journal = RolloutJournal(path)
    with pytest.raises(JournalSchemaError):
        _coordinator(journal)
    journal.close()
