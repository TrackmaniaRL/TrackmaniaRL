import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.headless_campaign_queue.__main__ import main, verify_receipt


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "queue.json"
        self.out = self.root / "out"

    def job(self, name, code):
        return {"id": name, "cwd": str(self.root), "argv": [sys.executable, "-c", code]}

    def write(self, jobs):
        self.manifest.write_text(json.dumps({"schema_version": 1, "jobs": jobs}), encoding="utf-8")

    def call(self, *options):
        return main([str(self.manifest), "--output", str(self.out), *options])

    def test_default_is_read_only(self):
        marker = self.root / "marker"
        self.write([self.job("first", f"open({str(marker)!r}, 'w').write('x')")])
        self.assertEqual(self.call(), 0)
        self.assertFalse(marker.exists())
        self.assertFalse(self.out.exists())

    def test_campaign_prototype_cannot_execute(self):
        self.manifest.write_text(
            json.dumps({"status": "prototype_not_runnable", "jobs": []}), encoding="utf-8"
        )
        self.assertEqual(self.call("--execute"), 2)
        self.assertFalse(self.out.exists())

    def test_qualification_receipt_cannot_complete_campaign(self):
        receipt = self.root / "receipt.json"
        receipt.write_text(
            json.dumps({"job_id": "first", "status": "completed", "qualification_only": True})
        )
        with self.assertRaisesRegex(ValueError, "qualification-only"):
            verify_receipt({"id": "first", "receipt": str(receipt)}, "f" * 64, 123)

    def test_campaign_receipt_accepts_only_declared_window(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.write_bytes(b"data")
        sha = hashlib.sha256(b"data").hexdigest()
        receipt = self.root / "receipt.json"
        value = {
            "job_id": "first",
            "status": "completed",
            "process_pid": 123,
            "manifest_sha256": "f" * 64,
            "requested_transitions": 100,
            "transitions": 105,
            "max_overshoot": 5,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha,
            "evaluation": str(checkpoint),
            "evaluation_sha256": sha,
        }
        receipt.write_text(json.dumps(value))
        job = {"id": "first", "receipt": str(receipt), "transitions": 100, "max_overshoot": 5}
        self.assertEqual(verify_receipt(job, "f" * 64, 123)["accepted_transitions"], 105)
        value["transitions"] = 106
        receipt.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            verify_receipt(job, "f" * 64, 123)

    def test_invalid_later_job_prevents_first_launch(self):
        marker = self.root / "marker"
        self.write(
            [
                self.job("first", f"open({str(marker)!r}, 'w').write('x')"),
                {"id": "bad", "cwd": str(self.root), "argv": [str(self.root / "missing")]},
            ]
        )
        self.assertEqual(self.call("--execute"), 2)
        self.assertFalse(marker.exists())
        self.assertFalse(self.out.exists())

    def test_failure_stops_queue_and_retains_logs_and_state(self):
        marker = self.root / "marker"
        self.write(
            [
                self.job("first", "print('hello')"),
                self.job("bad", "import sys; sys.exit(7)"),
                self.job("last", f"open({str(marker)!r}, 'w').write('x')"),
            ]
        )
        self.assertEqual(self.call("--execute"), 1)
        state = json.loads((self.out / "state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual([j["status"] for j in state["jobs"]], ["exited", "exited", "pending"])
        self.assertEqual(state["jobs"][1]["returncode"], 7)
        self.assertIn("hello", (self.out / "first.stdout.log").read_text())
        self.assertFalse(marker.exists())
        self.assertEqual(self.call("--execute"), 2)

    def test_lock_prevents_second_owner(self):
        self.write([self.job("first", "pass")])
        (self.out / ".lock").mkdir(parents=True)
        self.assertEqual(self.call("--execute"), 2)
        self.assertTrue((self.out / ".lock").exists())
        self.assertFalse((self.out / "state.json").exists())

    def test_successful_queue(self):
        self.write([self.job("one", "print('one')"), self.job("two", "print('two')")])
        self.assertEqual(self.call("--execute"), 0)
        state = json.loads((self.out / "state.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertTrue(all(j["returncode"] == 0 for j in state["jobs"]))
        self.assertFalse((self.out / ".lock").exists())

    def test_global_lock_blocks_other_output(self):
        self.write([self.job("first", "pass")])
        (self.root / ".trackmaniarl-headless-campaign.lock").mkdir()
        with patch("tools.headless_campaign_queue.__main__.Path.home", return_value=self.root):
            self.assertEqual(self.call("--execute"), 2)
        self.assertFalse((self.out / "state.json").exists())

    def test_missing_receipt_stops_queue_after_zero_exit(self):
        job = self.job("first", "pass")
        job.update(receipt=str(self.root / "receipt.json"), transitions=12)
        self.manifest.write_text(json.dumps({"schema_version": 2, "jobs": [job]}))
        self.assertEqual(self.call("--execute"), 1)
        state = json.loads((self.out / "state.json").read_text())
        self.assertEqual(state["jobs"][0]["status"], "invalid_receipt")
        self.assertEqual(state["status"], "failed")

    def test_valid_receipt_verifies_checkpoint_and_budget(self):
        checkpoint, receipt = self.root / "model.pt", self.root / "receipt.json"
        payload = {
            "job_id": "first",
            "status": "completed",
            "transitions": 12,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
        }
        payload.update(evaluation=str(checkpoint), evaluation_sha256=payload["checkpoint_sha256"])
        code = (
            f"from pathlib import Path; import json, os; Path({str(checkpoint)!r}).write_bytes(b'model'); "
            f"d=json.loads({json.dumps(payload)!r}); d.update(process_pid=os.getpid(), manifest_sha256=os.environ['TRACKMANIARL_QUEUE_MANIFEST_SHA256']); "
            f"Path({str(receipt)!r}).write_text(json.dumps(d))"
        )
        job = self.job("first", code)
        job.update(receipt=str(receipt), transitions=12)
        self.manifest.write_text(json.dumps({"schema_version": 2, "jobs": [job]}))
        self.assertEqual(self.call("--execute"), 0)
        state = json.loads((self.out / "state.json").read_text())
        self.assertEqual(state["jobs"][0]["status"], "verified")

    def test_receipt_from_different_process_is_rejected(self):
        checkpoint, receipt = self.root / "model.pt", self.root / "receipt.json"
        payload = {
            "job_id": "first",
            "status": "completed",
            "transitions": 12,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
        }
        payload.update(evaluation=str(checkpoint), evaluation_sha256=payload["checkpoint_sha256"])
        code = (
            f"from pathlib import Path; import json, os; Path({str(checkpoint)!r}).write_bytes(b'model'); "
            f"d=json.loads({json.dumps(payload)!r}); d.update(process_pid=os.getpid()+1, manifest_sha256=os.environ['TRACKMANIARL_QUEUE_MANIFEST_SHA256']); "
            f"Path({str(receipt)!r}).write_text(json.dumps(d))"
        )
        job = self.job("first", code)
        job.update(receipt=str(receipt), transitions=12)
        self.manifest.write_text(json.dumps({"schema_version": 2, "jobs": [job]}))
        self.assertEqual(self.call("--execute"), 1)
        state = json.loads((self.out / "state.json").read_text())
        self.assertEqual(state["jobs"][0]["status"], "invalid_receipt")

    def test_resume_checks_checkpoint_before_starting_pending_job(self):
        receipt = self.root / "receipt.json"
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        marker = self.root / "marker"
        first = self.job("first", "pass")
        first.update(receipt=str(receipt), transitions=12)
        second = self.job("second", f"open({str(marker)!r}, 'w').write('x')")
        second.update(receipt=str(self.root / "second.json"), transitions=12)
        self.manifest.write_text(json.dumps({"schema_version": 2, "jobs": [first, second]}))
        digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        receipt.write_text(
            json.dumps(
                {
                    "job_id": "first",
                    "status": "completed",
                    "transitions": 12,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha,
                    "evaluation": str(checkpoint),
                    "evaluation_sha256": sha,
                    "manifest_sha256": digest,
                    "process_pid": 123,
                }
            )
        )
        self.out.mkdir()
        state = {
            "status": "running",
            "manifest_sha256": digest,
            "jobs": [
                {
                    "id": "first",
                    "status": "verified",
                    "pid": 123,
                    "evidence": {
                        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                        "checkpoint_sha256": sha,
                        "evaluation_sha256": sha,
                    },
                },
                {"id": "second", "status": "pending"},
            ],
        }
        (self.out / "state.json").write_text(json.dumps(state))
        checkpoint.write_bytes(b"tampered")
        self.assertEqual(self.call("--execute", "--resume"), 2)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
