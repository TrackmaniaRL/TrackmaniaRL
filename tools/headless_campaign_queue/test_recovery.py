import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.headless_campaign_queue.recovery import (
    inspect_journal,
    reject_pending_spool,
    snapshot_journal,
    validate_state,
)


class RecoveryTests(unittest.TestCase):
    def test_sqlite_backup_preserves_committed_uncheckpointed_wal_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "journal.db"
            dest = Path(temp) / "copy" / "journal.db"
            db = sqlite3.connect(source)
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA wal_autocheckpoint=0")
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                db.execute(
                    "CREATE TABLE chunks (id INTEGER PRIMARY KEY, session_id TEXT, sequence INTEGER, payload BLOB)"
                )
                db.execute(
                    "CREATE TABLE receipts (session_id TEXT, sequence INTEGER, row_id INTEGER, payload_sha256 BLOB)"
                )
                db.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    [("journal_id", "j1"), ("pruned_through", "0")],
                )
                payload = b"accepted once"
                db.execute("INSERT INTO chunks VALUES (1,?,?,?)", ("s", 0, payload))
                db.execute(
                    "INSERT INTO receipts VALUES (?,?,?,?)",
                    ("s", 0, 1, hashlib.sha256(payload).digest()),
                )
                db.commit()
                self.assertGreater(Path(str(source) + "-wal").stat().st_size, 0)
                evidence = snapshot_journal(source, dest)
                self.assertIsNotNone(evidence["source"]["-wal"])
                self.assertEqual(inspect_journal(dest, "j1", 0)["pending_ids"], [1])
                self.assertEqual(inspect_journal(dest, "j1", 1)["pending_rows"], 0)
                with self.assertRaisesRegex(ValueError, "identity"):
                    inspect_journal(dest, "wrong", 0)
            finally:
                db.close()

    def test_journal_payload_must_match_durable_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                db.execute(
                    "CREATE TABLE chunks (id INTEGER PRIMARY KEY, session_id TEXT, sequence INTEGER, payload BLOB)"
                )
                db.execute(
                    "CREATE TABLE receipts (session_id TEXT, sequence INTEGER, row_id INTEGER, payload_sha256 BLOB)"
                )
                db.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    [("journal_id", "j"), ("pruned_through", "0")],
                )
                db.execute("INSERT INTO chunks VALUES (1,?,?,?)", ("s", 0, b"payload"))
                db.execute("INSERT INTO receipts VALUES (?,?,?,?)", ("s", 0, 1, b"wrong"))
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                inspect_journal(path, "j", 0)

    def test_pending_actor_spool_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spool = root / "actors" / "a" / "spool"
            spool.mkdir(parents=True)
            (spool / "0001.rollout").write_bytes(b"pending")
            with self.assertRaisesRegex(ValueError, "unacknowledged"):
                reject_pending_spool(root)
            self.assertEqual((spool / "0001.rollout").read_bytes(), b"pending")

    def test_state_requires_matching_fingerprint_and_full_replay(self):
        state = {
            "run_fingerprint": "f",
            "learner": {
                "weights": 1,
                "optimizer": {},
                "rng": {"python": 1, "numpy": 2, "torch": 3},
            },
            "distributed": {"transitions": 99, "updates": 2},
            "replay_store": {},
            "sampler": {},
        }
        self.assertEqual(validate_state(state, "f", 100, False)["transitions"], 99)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_state(state, "other", 100, False)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_state(state, "f", 99, False)
        del state["replay_store"]
        with self.assertRaisesRegex(ValueError, "FULL replay"):
            validate_state(state, "f", 100, False)

    def test_evaluation_requires_completed_budget_and_drained_credit(self):
        state = {
            "run_fingerprint": "f",
            "learner": {"optimizer": {}, "rng": {"python": 1, "numpy": 2, "torch": 3}},
            "distributed": {"transitions": 103, "updates": 2, "update_credit": 0.5},
            "replay_store": {},
            "sampler": {},
        }
        self.assertEqual(
            validate_state(state, "f", 100, False, stage="evaluation", max_overshoot=5)[
                "transitions"
            ],
            103,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_state(state, "f", 100, False, stage="evaluation", max_overshoot=2)
        state["distributed"]["update_credit"] = 1.0
        with self.assertRaisesRegex(ValueError, "undrained"):
            validate_state(state, "f", 100, False, stage="evaluation", max_overshoot=5)


if __name__ == "__main__":
    unittest.main()
