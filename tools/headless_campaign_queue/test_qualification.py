import json
from pathlib import Path
import tempfile
import unittest

from tools.headless_campaign_queue.qualification_audit import ALLOWED_DIFFERENCES, differences, matching_restore
from tools.headless_campaign_queue.qualification_restore import audit, equal, sha
from tools.headless_campaign_queue.qualification_readiness import assess


class QualificationTests(unittest.TestCase):
    def test_detects_unexpected_recipe_difference(self):
        full = {"training": {"warmup_transitions": 10000, "gamma": 0.995}, "run_id": "full"}
        short = {"training": {"warmup_transitions": 256, "gamma": 0.9}, "run_id": "short"}
        diff = differences(full, short)
        self.assertEqual({item["path"] for item in diff}, {"training.warmup_transitions", "training.gamma", "run_id"})
        self.assertNotIn("training.gamma", ALLOWED_DIFFERENCES)
        self.assertIn("training.warmup_transitions", ALLOWED_DIFFERENCES)

    def test_state_equality_checks_nested_rng_and_tensors(self):
        import numpy as np
        import torch
        a = {"optimizer": {"x": torch.tensor([1, 2])}, "rng": ("state", np.array([1, 2]))}
        b = {"optimizer": {"x": torch.tensor([1, 2])}, "rng": ("state", np.array([1, 2]))}
        self.assertGreater(equal(a, b), 0)
        b["rng"][1][1] = 3
        with self.assertRaisesRegex(ValueError, "rng"):
            equal(a, b)

    def test_restore_evidence_must_match_same_checkpoint_and_receipt(self):
        report = {"kind": "headless-qualification-offline-restore-v1", "status": "PASS", "job_id": "qr", "run_fingerprint": "fp", "source_sha256": {"config": "config", "checkpoint": "checkpoint", "receipt": "receipt"}}
        records = [{"path": "/audit/report.json", "sha256": "report", "report": report}]
        self.assertIsNone(matching_restore(records, "qr", "fp", "config", "different-receipt", "checkpoint"))
        self.assertIsNone(matching_restore(records, "qr", "fp", "config", "receipt", "different-checkpoint"))
        self.assertEqual(matching_restore(records, "qr", "fp", "config", "receipt", "checkpoint"), {"path": "/audit/report.json", "sha256": "report"})

    def test_first_campaign_readiness_does_not_require_prior_full_budget(self):
        ids = ["qr", "iqn", "sac", "tqc", "ppo"]
        draft = {"jobs": [{"id": name} for name in ids]}
        inventory = {"results": [{"id": name, "status": "insufficient_evidence", "short_receipt": {"sha256": "x"}, "same_checkpoint_restore": {"sha256": "y"}, "missing": ["full-budget qualification"]} for name in ids]}
        physical = {name: {"status": "PASS"} for name in ids}
        report = assess(inventory, physical, draft)
        self.assertEqual(report["status"], "READY_FOR_FIRST_CAMPAIGN")
        self.assertFalse(report["automatic_launch_authorized"])
        self.assertTrue(any("not proof of full-budget" in item for item in report["limitations"]))

    def test_missing_same_checkpoint_restore_or_source_blocks(self):
        ids = ["qr", "iqn", "sac", "tqc", "ppo"]
        draft = {"jobs": [{"id": name} for name in ids]}
        inventory = {"results": [{"id": name, "short_receipt": {"sha256": "x"}, "same_checkpoint_restore": None if name == "qr" else {"sha256": "y"}} for name in ids]}
        physical = {name: {"status": "FAIL" if name == "sac" else "PASS"} for name in ids}
        report = assess(inventory, physical, draft)
        self.assertEqual(report["status"], "NOT_READY")
        self.assertEqual({item["id"] for item in report["missing"]}, {"qr", "sac"})

    def test_refuses_nonqualification_receipt_before_creating_output(self):
        import trackmaniarl
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config, receipt = root / "config.yaml", root / "receipt.json"
            config.write_text("config")
            receipt.write_text(json.dumps({"status": "completed", "qualification_only": False}))
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "qualification-only"):
                audit(config, sha(config), receipt, sha(receipt), Path(trackmaniarl.__file__).resolve().parent.parent, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
