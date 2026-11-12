"""Game-free tests of map-switch quarantine and recovery."""

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from trackmaniarl.trackmania.map_switch_journal import (
    MapSwitchIntent,
    MapSwitchJournal,
    MapSwitchJournalError,
)


def _intent(identifier: str = "one") -> MapSwitchIntent:
    return MapSwitchIntent(identifier, "epoch-1", "uid-1", "map.Map.Gbx", "a" * 64)


def test_dispatch_is_durable_before_send_and_blocks_after_reopening(tmp_path: Path) -> None:
    path = tmp_path / "switch.sqlite"
    journal = MapSwitchJournal(path, "game-1", create=True)
    intent = _intent()
    journal.prepare(intent)
    journal.begin_dispatch(intent.operation_id)

    # Simulate the first action of an external send, including a newly opened
    # observer as would be needed after actor termination.
    reopened = MapSwitchJournal(path, "game-1")
    assert reopened.pending() == (intent, "dispatching")
    with pytest.raises(MapSwitchJournalError, match="unresolved"):
        reopened.prepare(_intent("two"))
    with pytest.raises(MapSwitchJournalError):
        reopened.abandon_prepared("one")
    with pytest.raises(MapSwitchJournalError, match="epoch"):
        reopened.resolve_terminal("one", "different-epoch", "ready")
    assert reopened.pending() == (intent, "dispatching")
    reopened.resolve_terminal("one", "epoch-1", "terminal result verified")
    assert reopened.pending() is None
    reopened.prepare(_intent("two"))


def test_prepared_intent_is_recovered_and_can_only_be_abandoned_before_dispatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "switch.sqlite"
    MapSwitchJournal(path, "game", create=True).prepare(_intent())
    recovered = MapSwitchJournal(path, "game")
    assert recovered.pending() == (_intent(), "prepared")
    recovered.abandon_prepared("one")
    with pytest.raises(MapSwitchJournalError):
        recovered.begin_dispatch("one")
    with pytest.raises(MapSwitchJournalError, match="already been used"):
        recovered.prepare(_intent())


def test_instance_identity_and_empty_evidence_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "switch.sqlite"
    journal = MapSwitchJournal(path, "one", create=True)
    journal.prepare(_intent())
    journal.begin_dispatch("one")
    with pytest.raises(MapSwitchJournalError, match="another game"):
        MapSwitchJournal(path, "two")
    with pytest.raises(ValueError, match="evidence"):
        journal.resolve_terminal("one", "epoch-1", "")
    assert journal.pending() == (_intent(), "dispatching")


def test_concurrent_prepare_allows_only_one_unresolved_intent(tmp_path: Path) -> None:
    path = tmp_path / "switch.sqlite"
    MapSwitchJournal(path, "game", create=True)
    barrier = Barrier(2)

    def prepare(identifier: str) -> str:
        journal = MapSwitchJournal(path, "game")
        barrier.wait(timeout=5)
        try:
            journal.prepare(_intent(identifier))
        except MapSwitchJournalError:
            return "rejected"
        return "prepared"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(prepare, ("one", "two")))
    assert sorted(results) == ["prepared", "rejected"]
    assert MapSwitchJournal(path, "game").pending() is not None


def test_abrupt_process_exit_leaves_dispatch_quarantined(tmp_path: Path) -> None:
    path = tmp_path / "switch.sqlite"
    script = """
import os, sys
from pathlib import Path
from trackmaniarl.trackmania.map_switch_journal import MapSwitchIntent, MapSwitchJournal
journal = MapSwitchJournal(Path(sys.argv[1]), 'game', create=True)
journal.prepare(MapSwitchIntent('one', 'epoch-1', 'uid-1', 'map.Map.Gbx', 'a' * 64))
journal.begin_dispatch('one')
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], check=False, timeout=15)
    assert result.returncode == 17
    assert MapSwitchJournal(path, "game").pending() == (_intent(), "dispatching")


def test_existing_quarantine_cannot_be_recreated_or_silently_missing(tmp_path: Path) -> None:
    path = tmp_path / "switch.sqlite"
    MapSwitchJournal(path, "game", create=True)
    with pytest.raises(FileExistsError):
        MapSwitchJournal(path, "game", create=True)
    with pytest.raises(sqlite3.OperationalError):
        MapSwitchJournal(tmp_path / "missing.sqlite", "game")


def test_journal_keeps_its_path_after_working_directory_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    journal = MapSwitchJournal(Path("switch.sqlite"), "game", create=True)
    journal.prepare(_intent())
    monkeypatch.chdir(second)
    MapSwitchJournal(Path("switch.sqlite"), "other", create=True)
    assert journal.pending() == (_intent(), "prepared")
    journal.begin_dispatch("one")
    assert MapSwitchJournal(first / "switch.sqlite", "game").pending() == (_intent(), "dispatching")


def test_instance_is_rechecked_on_each_connection(tmp_path: Path) -> None:
    path = tmp_path / "switch.sqlite"
    journal = MapSwitchJournal(path, "game", create=True)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE metadata SET instance_id = 'other'")
    with pytest.raises(MapSwitchJournalError, match="another game"):
        journal.prepare(_intent())
