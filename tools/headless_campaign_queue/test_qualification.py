import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.headless_campaign_queue.qualification_audit import (
    ALLOWED_DIFFERENCES,
    differences,
    matching_restore,
)
from tools.headless_campaign_queue.qualification_bridge import compare_maps, verify_training_change
from tools.headless_campaign_queue.qualification_readiness import assess
from tools.headless_campaign_queue.qualification_restore import audit, equal, sha


class QualificationTests(unittest.TestCase):
    def test_source_bridge_rejects_unknown_file_change(self) -> None:
        with pytest.raises(ValueError, match="unrecognized source change"):
            compare_maps({"trackmaniarl/algorithm.py": "a"}, {"trackmaniarl/algorithm.py": "b"})

    def test_shutdown_bridge_checks_actual_contents(self) -> None:
        anchor = "from trackmaniarl.trackmania.demonstrations import resolve_demonstration_paths\n"
        original = anchor + "\n_join_during_shutdown(processes.actor, timeout=10)\n"
        updated = anchor + (
            "\n# Actor teardown includes native environment close, tr"
            "ace archiving, sender\n# drain and background-worker joi"
            "ns. Its internal sender and worker grace\n# periods are "
            "each 10 s, so the launcher's bound must exceed their su"
            "m.\n_ACTOR_SHUTDOWN_GRACE_S = 60.0\n\n_join_during_shutdow"
            "n(processes.actor, timeout=_ACTOR_SHUTDOWN_GRACE_S)\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / "old.py", root / "new.py"
            old.write_text(original, encoding="utf-8")
            new.write_text(updated, encoding="utf-8")
            with (
                patch("tools.headless_campaign_queue.qualification_bridge.OLD_TRAINING", sha(old)),
                patch("tools.headless_campaign_queue.qualification_bridge.NEW_TRAINING", sha(new)),
            ):
                verify_training_change(old, new)
                altered = updated + "\nextra_change = True\n"
                new.write_text(altered, encoding="utf-8")
                with (
                    patch(
                        "tools.headless_campaign_queue.qualification_bridge.NEW_TRAINING", sha(new)
                    ),
                    pytest.raises(ValueError, match="more than"),
                ):
                    verify_training_change(old, new)

    def test_detects_unexpected_recipe_difference(self) -> None:
        full = {"training": {"warmup_transitions": 10000, "gamma": 0.995}, "run_id": "full"}
        short = {"training": {"warmup_transitions": 256, "gamma": 0.9}, "run_id": "short"}
        diff = differences(full, short)
        assert {item["path"] for item in diff} == {
            "training.warmup_transitions",
            "training.gamma",
            "run_id",
        }
        assert "training.gamma" not in ALLOWED_DIFFERENCES
        assert "training.warmup_transitions" in ALLOWED_DIFFERENCES

    def test_state_equality_checks_nested_rng_and_tensors(self) -> None:
        import numpy as np
        import torch

        a = {"optimizer": {"x": torch.tensor([1, 2])}, "rng": ("state", np.array([1, 2]))}
        b = {"optimizer": {"x": torch.tensor([1, 2])}, "rng": ("state", np.array([1, 2]))}
        assert equal(a, b) > 0
        b["rng"][1][1] = 3
        with pytest.raises(ValueError, match="rng"):
            equal(a, b)

    def test_restore_evidence_must_match_same_checkpoint_and_receipt(self) -> None:
        report = {
            "kind": "headless-qualification-offline-restore-v1",
            "status": "PASS",
            "job_id": "qr",
            "run_fingerprint": "fp",
            "source_sha256": {"config": "config", "checkpoint": "checkpoint", "receipt": "receipt"},
        }
        records = [{"path": "/audit/report.json", "sha256": "report", "report": report}]
        assert (
            matching_restore(records, "qr", "fp", "config", "different-receipt", "checkpoint")
            is None
        )
        assert (
            matching_restore(records, "qr", "fp", "config", "receipt", "different-checkpoint")
            is None
        )
        assert matching_restore(records, "qr", "fp", "config", "receipt", "checkpoint") == {
            "path": "/audit/report.json",
            "sha256": "report",
        }

    def test_first_campaign_readiness_does_not_require_prior_full_budget(self) -> None:
        ids = ["qr", "iqn", "sac", "tqc", "ppo"]
        draft = {"jobs": [{"id": name} for name in ids]}
        inventory = {
            "results": [
                {
                    "id": name,
                    "status": "insufficient_evidence",
                    "short_receipt": {"sha256": "x"},
                    "same_checkpoint_restore": {"sha256": "y"},
                    "missing": ["full-budget qualification"],
                }
                for name in ids
            ]
        }
        physical = {name: {"status": "PASS"} for name in ids}
        report = assess(inventory, physical, draft)
        assert report["status"] == "READY_FOR_FIRST_CAMPAIGN"
        assert not report["automatic_launch_authorized"]
        assert any("not proof of full-budget" in item for item in report["limitations"])

    def test_missing_same_checkpoint_restore_or_source_blocks(self) -> None:
        ids = ["qr", "iqn", "sac", "tqc", "ppo"]
        draft = {"jobs": [{"id": name} for name in ids]}
        inventory = {
            "results": [
                {
                    "id": name,
                    "short_receipt": {"sha256": "x"},
                    "same_checkpoint_restore": None if name == "qr" else {"sha256": "y"},
                }
                for name in ids
            ]
        }
        physical = {name: {"status": "FAIL" if name == "sac" else "PASS"} for name in ids}
        report = assess(inventory, physical, draft)
        assert report["status"] == "NOT_READY"
        assert {item["id"] for item in report["missing"]} == {"qr", "sac"}

    def test_refuses_nonqualification_receipt_before_creating_output(self) -> None:
        import trackmaniarl

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config, receipt = root / "config.yaml", root / "receipt.json"
            config.write_text("config")
            receipt.write_text(json.dumps({"status": "completed", "qualification_only": False}))
            output = root / "output"
            with pytest.raises(ValueError, match="qualification-only"):
                audit(
                    config,
                    sha(config),
                    receipt,
                    sha(receipt),
                    Path(trackmaniarl.__file__).resolve().parent.parent,
                    output,
                )
            assert not output.exists()


if __name__ == "__main__":
    unittest.main()
