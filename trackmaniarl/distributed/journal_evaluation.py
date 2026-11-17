"""Evaluation delivery and evidence stored in the rollout journal.

The mixin shares the journal connection and lock: evaluation writes participate
in the same SQLite transactions as rollout receipts where required.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from _thread import RLock
from hashlib import sha256
from typing import Any

from trackmaniarl.distributed.journal_errors import JournalPayloadConflictError, JournalSchemaError


class EvaluationJournalMixin:
    _connection: sqlite3.Connection
    _lock: RLock

    def _metadata_value(self, key: str) -> str:
        raise NotImplementedError

    def evaluation_state(self) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'evaluation_delivery'"
            ).fetchone()
        return None if row is None else str(row[0])

    def save_evaluation_state(self, state: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('evaluation_delivery', ?)",
                (state,),
            )
            self._connection.commit()

    def evaluation_delivery(self) -> tuple[set[str], dict[str, dict[str, Any]], int] | None:
        with self._lock, self._connection:
            # Multiple SELECTs must observe the same database version even if
            # another journal connection commits during recovery.
            self._connection.execute("BEGIN")
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'evaluation_delivery_frontier'"
            ).fetchone()
            if row is None:
                return None
            due = {
                str(item[0])
                for item in self._connection.execute("SELECT actor_id FROM evaluation_due")
            }
            verified: dict[bytes, bytes] = {}
            # Read each referenced BLOB once, rather than transferring it for
            # every actor sharing the same policy snapshot.
            for digest_value, snapshot in self._connection.execute(
                """SELECT digest, payload FROM evaluation_snapshots
                   WHERE digest IN (SELECT snapshot_digest FROM evaluation_assignments)"""
            ):
                digest = bytes(digest_value)
                payload = bytes(snapshot)
                if sha256(payload).digest() != digest:
                    raise JournalSchemaError("evaluation snapshot checksum mismatch")
                verified[digest] = payload
            assignments = {}
            for actor, details, digest_value in self._connection.execute(
                "SELECT actor_id, details, snapshot_digest FROM evaluation_assignments"
            ):
                digest = bytes(digest_value)
                if digest not in verified:
                    raise JournalSchemaError("evaluation assignment references missing snapshot")
                assignments[str(actor)] = {**json.loads(details), "snapshot": verified[digest]}
            return due, assignments, int(row[0])

    def save_evaluation_delivery(  # noqa: PLR0913
        self,
        due: set[str],
        assignments: dict[str, dict[str, Any]],
        frontier: int,
        old_due: set[str],
        old_assignments: dict[str, dict[str, Any]],
    ) -> None:
        """Commit only changed assignments and their content-addressed snapshots."""
        with self._lock, self._connection:
            for actor in old_due - due:
                self._connection.execute("DELETE FROM evaluation_due WHERE actor_id = ?", (actor,))
            self._connection.executemany(
                "INSERT OR IGNORE INTO evaluation_due(actor_id) VALUES (?)",
                ((actor,) for actor in due - old_due),
            )
            for actor in old_assignments.keys() - assignments.keys():
                self._connection.execute(
                    "DELETE FROM evaluation_assignments WHERE actor_id = ?", (actor,)
                )
            for actor, assignment in assignments.items():
                if actor in old_assignments and assignment == old_assignments[actor]:
                    continue
                snapshot = assignment["snapshot"]
                digest = sha256(snapshot).digest()
                self._connection.execute(
                    "INSERT OR IGNORE INTO evaluation_snapshots(digest, payload) VALUES (?, ?)",
                    (digest, snapshot),
                )
                details = json.dumps(
                    {key: value for key, value in assignment.items() if key != "snapshot"},
                    separators=(",", ":"),
                )
                self._connection.execute(
                    "INSERT OR REPLACE INTO evaluation_assignments"
                    "(actor_id, details, snapshot_digest) VALUES (?, ?, ?)",
                    (actor, details, digest),
                )
            self._connection.execute(
                "DELETE FROM evaluation_snapshots WHERE digest NOT IN "
                "(SELECT snapshot_digest FROM evaluation_assignments)"
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) "
                "VALUES ('evaluation_delivery_frontier', ?)",
                (str(frontier),),
            )
            self._connection.execute("DELETE FROM metadata WHERE key = 'evaluation_delivery'")

    def is_canonical_evaluation(self, actor_id: str, assignment_id: str, row_id: int) -> bool:
        """Persist the first row while allowing replay of that row after a crash.

        Later rows for the same assignment are duplicates. Metadata survives
        journal pruning along with rollout receipts.
        """
        if not actor_id or not assignment_id or row_id <= 0:
            raise ValueError("actor, assignment and positive journal row are required")
        with self._lock, self._connection:
            return self._claim_canonical_evaluation(actor_id, assignment_id, row_id)

    def _claim_canonical_evaluation(self, actor_id: str, assignment_id: str, row_id: int) -> bool:
        identity = json.dumps([actor_id, assignment_id], ensure_ascii=True, separators=(",", ":"))
        key = "evaluation_result:" + sha256(identity.encode("utf-8")).hexdigest()
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, str(row_id))
        )
        return int(self._metadata_value(key)) == row_id

    def evaluation_row_accepted(self, row_id: int) -> bool | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (f"evaluation_row:{row_id}",)
            ).fetchone()
            return None if row is None else str(row[0]) == "1"

    def stage_evaluation_promotion(self, row_id: int, payload: bytes) -> None:
        """Keep exact policy evidence until both leader promotions are durable."""
        if row_id <= 0:
            raise ValueError("positive evaluation journal row required")
        key = f"evaluation_promotion:{row_id:020d}"
        encoded = sha256(payload).hexdigest() + ":" + base64.b64encode(payload).decode("ascii")
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, encoded)
            )
            if self._metadata_value(key) != encoded:
                raise JournalPayloadConflictError("evaluation promotion evidence changed")

    def pending_evaluation_promotions(self) -> list[tuple[int, bytes]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT key, value FROM metadata "
                "WHERE key LIKE 'evaluation_promotion:%' ORDER BY key"
            ).fetchall()
        pending = []
        for key, value in rows:
            try:
                expected, encoded = str(value).split(":", 1)
                payload = base64.b64decode(encoded, validate=True)
                if sha256(payload).hexdigest() != expected:
                    raise ValueError("checksum mismatch")
                pending.append((int(str(key).split(":", 1)[1]), payload))
            except (TypeError, ValueError) as exc:
                raise JournalSchemaError("invalid evaluation promotion evidence") from exc
        return pending

    def complete_evaluation_promotion(self, row_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM metadata WHERE key = ?", (f"evaluation_promotion:{row_id:020d}",)
            )
