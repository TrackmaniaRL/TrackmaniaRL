import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.headless_campaign_queue.verify_artifacts import (
    final_checkpoint,
    validate_checkpoint,
    validate_drained_journal,
    validate_evaluation,
)


class ArtifactTests(unittest.TestCase):
    def test_checks_complete_off_policy_state(self):
        state = {
            "run_fingerprint": "fp",
            "learner": {"x": 1},
            "distributed": {"transitions": 50, "updates": 5, "update_credit": 0.5},
            "replay_store": {},
            "sampler": {},
        }
        validate_checkpoint(state, "fp", 50, on_policy=False)
        for field, value in (("run_fingerprint", "other"), ("learner", {}), ("replay_store", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_checkpoint({**state, field: value}, "fp", 50, on_policy=False)
        with self.assertRaises(ValueError):
            validate_checkpoint(state, "fp", 51, on_policy=False)
        with self.assertRaisesRegex(ValueError, "undrained"):
            validate_checkpoint(
                {**state, "distributed": {**state["distributed"], "update_credit": 1}},
                "fp",
                50,
                on_policy=False,
            )

    def test_checks_ppo_counter(self):
        state = {
            "run_fingerprint": "fp",
            "learner": {"x": 1},
            "counters": {"transitions": 50, "updates": 1},
        }
        validate_checkpoint(state, "fp", 50, on_policy=True)
        with self.assertRaises(ValueError):
            validate_checkpoint(state, "fp", 50, on_policy=False)

    def test_acceptance_window_uses_actual_checkpoint_count(self):
        state = {
            "run_fingerprint": "fp",
            "learner": {"x": 1},
            "counters": {"transitions": 55, "updates": 1},
        }
        self.assertEqual(validate_checkpoint(state, "fp", 50, on_policy=True, max_overshoot=5), 55)
        with self.assertRaises(ValueError):
            validate_checkpoint(state, "fp", 50, on_policy=True, max_overshoot=4)
        with self.assertRaises(ValueError):
            validate_checkpoint(state, "fp", 56, on_policy=True, max_overshoot=5)

    def test_last_completed_checkpoint_and_pending_wal_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            directory = run_dir / "checkpoints"
            directory.mkdir()
            earlier, final = (
                directory / "distributed-update-00000001.pt",
                directory / "distributed-update-00000002.pt",
            )
            earlier.write_bytes(b"earlier")
            final.write_bytes(b"final")
            events = [
                {"event": "train/checkpoint_completed", "payload": {"path": str(p)}, "step": step}
                for p, step in ((earlier, 1), (final, 2))
            ]
            (run_dir / "events.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
            self.assertEqual(final_checkpoint(run_dir, directory, on_policy=False), (final, 2))
            journal = run_dir / "distributed" / "rollouts.sqlite3"
            journal.parent.mkdir()
            with sqlite3.connect(journal) as db:
                db.execute("CREATE TABLE metadata(key TEXT, value TEXT)")
                db.executemany(
                    "INSERT INTO metadata VALUES(?,?)",
                    [("journal_id", "id"), ("pruned_through", "5")],
                )
                db.execute("CREATE TABLE chunks(id INTEGER)")
            state = {"journal_id": "id", "distributed": {"journal_applied_frontier": 5}}
            validate_drained_journal(run_dir, state)
            with sqlite3.connect(journal) as db:
                db.execute("INSERT INTO chunks VALUES(6)")
            with self.assertRaisesRegex(ValueError, "not drained"):
                validate_drained_journal(run_dir, state)

    def test_evaluation_includes_failures_and_unique_indices(self):
        artifact = {
            "checkpoint": str(Path("/tmp/final.pt")),
            "trials": [
                {"map_id": "map", "map_uid": "uid", "trial_index": 0, "finished": False},
                {"map_id": "map", "map_uid": "uid", "trial_index": 1, "finished": True},
            ],
        }
        validate_evaluation(artifact, Path("/tmp/final.pt"), "map", "uid", 2)
        artifact["trials"][1]["trial_index"] = 0
        with self.assertRaises(ValueError):
            validate_evaluation(artifact, Path("/tmp/final.pt"), "map", "uid", 2)


if __name__ == "__main__":
    unittest.main()
