import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from tools.headless_campaign_queue.recovery import (
    evaluate_copy,
    inspect_journal,
    reject_pending_spool,
    snapshot_journal,
    validate_state,
)


class RecoveryTests(unittest.TestCase):
    def test_cli_blocks_normal_resume_when_replayed_wal_reaches_budget(self) -> None:
        from tools.headless_campaign_queue import recovery

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            args = [
                "recovery",
                "--config",
                str(home / "run.yaml"),
                "--config-sha256",
                "x",
                "--checkpoint",
                str(home / "model.pt"),
                "--destination",
                str(home / "out"),
                "--stage",
                "training",
                "--execute",
            ]
            with (
                patch("sys.argv", args),
                patch.object(Path, "home", return_value=home),
                patch.object(recovery, "reject_live_competitors"),
                patch.object(recovery, "prepare", return_value={"stage": "wal_finalization"}),
                patch.object(recovery.subprocess, "Popen") as popen,
            ):
                assert recovery.main() == 2
                popen.assert_not_called()
            assert not (home / ".trackmaniarl-headless-campaign.lock").exists()

    def test_evaluation_freshness_checked_before_resolver_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            checkpoint = base / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            evaluation = SimpleNamespace(maps=[object()], trials_per_map=30)
            spec = SimpleNamespace(
                run_id="run", artifacts_dir=Path("artifacts"), evaluation=evaluation
            )
            spec.model_copy = lambda update: SimpleNamespace(**{**spec.__dict__, **update})
            expected = base / "artifacts" / "run-recovery-evaluation"
            run = SimpleNamespace(run_dir=expected, logger=Mock())

            def resolve(*args: Any, **kwargs: Any) -> Any:
                expected.mkdir(parents=True)
                return run

            with (
                patch("trackmaniarl.core.runtime.resolve_run", side_effect=resolve),
                patch(
                    "trackmaniarl.commands.evaluation._configure_evaluation_stop",
                    side_effect=RuntimeError("reached evaluator"),
                ),
                pytest.raises(RuntimeError, match="reached evaluator"),
            ):
                evaluate_copy(spec, base, checkpoint, base / "STOP", {})
            run.logger.close.assert_called_once()
            with patch("trackmaniarl.core.runtime.resolve_run") as resolver:
                with pytest.raises(ValueError, match="fresh"):
                    evaluate_copy(spec, base, checkpoint, base / "STOP", {})
                resolver.assert_not_called()

    def test_sqlite_backup_preserves_committed_uncheckpointed_wal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "journal.db"
            dest = Path(temp) / "copy" / "journal.db"
            db = sqlite3.connect(source)
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA wal_autocheckpoint=0")
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                db.execute(
                    "CREATE TABLE chunks (id INTEGER PRIMARY KEY, session_id"
                    " TEXT, sequence INTEGER, payload BLOB)"
                )
                db.execute(
                    "CREATE TABLE receipts (session_id TEXT, sequence INTEGE"
                    "R, row_id INTEGER, payload_sha256 BLOB)"
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
                assert Path(str(source) + "-wal").stat().st_size > 0
                evidence = snapshot_journal(source, dest)
                assert evidence["source"]["-wal"] is not None
                assert inspect_journal(dest, "j1", 0)["pending_ids"] == [1]
                assert inspect_journal(dest, "j1", 1)["pending_rows"] == 0
                with pytest.raises(ValueError, match="identity"):
                    inspect_journal(dest, "wrong", 0)
            finally:
                db.close()

    def test_journal_payload_must_match_durable_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                db.execute(
                    "CREATE TABLE chunks (id INTEGER PRIMARY KEY, session_id"
                    " TEXT, sequence INTEGER, payload BLOB)"
                )
                db.execute(
                    "CREATE TABLE receipts (session_id TEXT, sequence INTEGE"
                    "R, row_id INTEGER, payload_sha256 BLOB)"
                )
                db.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    [("journal_id", "j"), ("pruned_through", "0")],
                )
                db.execute("INSERT INTO chunks VALUES (1,?,?,?)", ("s", 0, b"payload"))
                db.execute("INSERT INTO receipts VALUES (?,?,?,?)", ("s", 0, 1, b"wrong"))
            with pytest.raises(ValueError, match="receipt mismatch"):
                inspect_journal(path, "j", 0)

    def test_pending_actor_spool_blocks_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spool = root / "actors" / "a" / "spool"
            spool.mkdir(parents=True)
            (spool / "0001.rollout").write_bytes(b"pending")
            with pytest.raises(ValueError, match="unacknowledged"):
                reject_pending_spool(root)
            assert (spool / "0001.rollout").read_bytes() == b"pending"

    def test_state_requires_matching_fingerprint_and_full_replay(self) -> None:
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
        assert validate_state(state, "f", 100, on_policy=False)["transitions"] == 99
        with pytest.raises(ValueError, match="fingerprint"):
            validate_state(state, "other", 100, on_policy=False)
        with pytest.raises(ValueError, match="incomplete"):
            validate_state(state, "f", 99, on_policy=False)
        del state["replay_store"]
        with pytest.raises(ValueError, match="FULL replay"):
            validate_state(state, "f", 100, on_policy=False)

    def test_evaluation_requires_completed_budget_and_drained_credit(self) -> None:
        state = {
            "run_fingerprint": "f",
            "learner": {"optimizer": {}, "rng": {"python": 1, "numpy": 2, "torch": 3}},
            "distributed": {"transitions": 103, "updates": 2, "update_credit": 0.5},
            "replay_store": {},
            "sampler": {},
        }
        assert (
            validate_state(state, "f", 100, on_policy=False, stage="evaluation", max_overshoot=5)[
                "transitions"
            ]
            == 103
        )
        with pytest.raises(ValueError, match="outside"):
            validate_state(state, "f", 100, on_policy=False, stage="evaluation", max_overshoot=2)
        state["distributed"]["update_credit"] = 1.0
        with pytest.raises(ValueError, match="undrained"):
            validate_state(state, "f", 100, on_policy=False, stage="evaluation", max_overshoot=5)


if __name__ == "__main__":
    unittest.main()
