"""Durable, fail-closed intent records for experimental map switching.

This module does not launch maps or authenticate backend evidence. A backend
must verify operation identity, terminality and readiness before resolving a
record. The experimental scheduled backend is not validated in a real game.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class MapSwitchJournalError(RuntimeError):
    """The operation conflicts with the durable state of the game instance."""


@dataclass(frozen=True, slots=True)
class MapSwitchIntent:
    operation_id: str
    plugin_epoch: str
    target_uid: str
    target_path: str
    target_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.operation_id, self.plugin_epoch, self.target_uid, self.target_path)
        ):
            raise ValueError("operation ID, epoch, map UID and path are required")
        if len(self.target_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.target_sha256
        ):
            raise ValueError("target SHA-256 must be lowercase hexadecimal")


class MapSwitchJournal:
    """Serialize switch intents in a local SQLite file for one game instance.

    ``begin_dispatch`` must return before a launch request is sent. A crash or
    transport error after that point leaves the record unresolved, blocking all
    later operations until a backend verifies a terminal result. Do not use a
    shared database path for different instances, or a filesystem that cannot
    provide SQLite locking and durable sync.
    """

    def __init__(self, path: Path, instance_id: str, *, create: bool = False) -> None:
        if not instance_id.strip():
            raise ValueError("game instance identity is required")
        self.path = Path(path).resolve()
        self.instance_id = instance_id
        if create:
            # Never silently replace a previous quarantine, even if it is empty
            # after an interrupted creation.
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        with self._connect(verify=not create) as db:
            db.execute("BEGIN IMMEDIATE")
            if create:
                db.execute(
                    "CREATE TABLE metadata (instance_id TEXT NOT NULL, version INTEGER NOT NULL)"
                )
                db.execute("""CREATE TABLE operations (
                operation_id TEXT PRIMARY KEY, plugin_epoch TEXT NOT NULL,
                target_uid TEXT NOT NULL, target_path TEXT NOT NULL,
                target_sha256 TEXT NOT NULL, state TEXT NOT NULL
                    CHECK (state IN ('prepared', 'dispatching', 'abandoned', 'terminal')),
                terminal_evidence TEXT
            )""")
                db.execute(
                    "CREATE UNIQUE INDEX one_unresolved ON operations ((1)) "
                    "WHERE state IN ('prepared', 'dispatching')"
                )
                db.execute("INSERT INTO metadata VALUES (?, 1)", (instance_id,))
            identities = db.execute("SELECT instance_id, version FROM metadata").fetchall()
            if identities != [(instance_id, 1)]:
                raise MapSwitchJournalError("journal belongs to another game instance")
            db.commit()

    @contextmanager
    def _connect(self, *, verify: bool = True) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=rw", timeout=5.0, isolation_level=None, uri=True
        )
        try:
            # EXTRA adds a directory sync when SQLite supports it. Filesystem
            # and device durability still depend on the local host.
            db.execute("PRAGMA synchronous=EXTRA")
            mode = db.execute("PRAGMA journal_mode").fetchone()
            if mode != ("delete",):
                raise MapSwitchJournalError("map-switch journal requires DELETE journal mode")
            if verify:
                identities = db.execute("SELECT instance_id, version FROM metadata").fetchall()
                if identities != [(self.instance_id, 1)]:
                    raise MapSwitchJournalError("journal belongs to another game instance")
            yield db
        finally:
            db.close()

    def pending(self) -> tuple[MapSwitchIntent, str] | None:
        """Return a durable unresolved intent, including one from a crashed actor."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT operation_id, plugin_epoch, target_uid, target_path, target_sha256, state "
                "FROM operations WHERE state IN ('prepared', 'dispatching')"
            ).fetchall()
        if len(rows) > 1:
            raise MapSwitchJournalError("journal has multiple unresolved operations")
        if not rows:
            return None
        return MapSwitchIntent(*rows[0][:5]), rows[0][5]

    def latest(self) -> tuple[MapSwitchIntent, str] | None:
        """Return the latest durable intent, including terminal operations."""
        with self._connect() as db:
            row = db.execute(
                "SELECT operation_id, plugin_epoch, target_uid, target_path, target_sha256, state "
                "FROM operations ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        return None if row is None else (MapSwitchIntent(*row[:5]), row[5])

    def prepare(self, intent: MapSwitchIntent) -> None:
        """Reserve an operation before sending any command; existing IDs never repeat."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM operations WHERE state IN ('prepared', 'dispatching') LIMIT 1"
            ).fetchone():
                raise MapSwitchJournalError("a switch is unresolved; session is quarantined")
            try:
                db.execute(
                    "INSERT INTO operations VALUES (?, ?, ?, ?, ?, 'prepared', NULL)",
                    (
                        intent.operation_id,
                        intent.plugin_epoch,
                        intent.target_uid,
                        intent.target_path,
                        intent.target_sha256,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise MapSwitchJournalError("operation ID has already been used") from exc
            db.commit()

    def begin_dispatch(self, operation_id: str) -> None:
        """Durably mark the result ambiguous before any externally visible send."""
        self._transition(operation_id, "prepared", "dispatching")

    def abandon_prepared(self, operation_id: str) -> None:
        """Abandon only an intent whose dispatch has not begun."""
        self._transition(operation_id, "prepared", "abandoned")

    def resolve_terminal(self, operation_id: str, plugin_epoch: str, evidence: str) -> None:
        """Record a terminal result verified by the backend, not a queue ACK."""
        if not evidence.strip():
            raise ValueError("verified terminal evidence is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT plugin_epoch, state FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row != (plugin_epoch, "dispatching"):
                raise MapSwitchJournalError("operation or plugin epoch does not match")
            db.execute(
                "UPDATE operations SET state = 'terminal', terminal_evidence = ? "
                "WHERE operation_id = ?",
                (evidence, operation_id),
            )
            db.commit()

    def _transition(self, operation_id: str, old: str, new: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                "UPDATE operations SET state = ? WHERE operation_id = ? AND state = ?",
                (new, operation_id, old),
            )
            if cursor.rowcount != 1:
                raise MapSwitchJournalError("operation missing or in an unexpected state")
            db.commit()
