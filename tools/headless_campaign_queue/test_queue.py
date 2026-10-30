import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.headless_campaign_queue.__main__ import main


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
        self.manifest.write_text(json.dumps({"status": "prototype_not_runnable", "jobs": []}), encoding="utf-8")
        self.assertEqual(self.call("--execute"), 2)
        self.assertFalse(self.out.exists())

    def test_invalid_later_job_prevents_first_launch(self):
        marker = self.root / "marker"
        self.write([self.job("first", f"open({str(marker)!r}, 'w').write('x')"),
                    {"id": "bad", "cwd": str(self.root), "argv": [str(self.root / 'missing')]}])
        self.assertEqual(self.call("--execute"), 2)
        self.assertFalse(marker.exists())
        self.assertFalse(self.out.exists())

    def test_failure_stops_queue_and_retains_logs_and_state(self):
        marker = self.root / "marker"
        self.write([self.job("first", "print('hello')"), self.job("bad", "import sys; sys.exit(7)"),
                    self.job("last", f"open({str(marker)!r}, 'w').write('x')")])
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


if __name__ == "__main__":
    unittest.main()
