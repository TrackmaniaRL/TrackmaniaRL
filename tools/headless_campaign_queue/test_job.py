import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never
from unittest.mock import patch

import pytest

from tools.headless_campaign_queue.job import execute, monitor_disk


class FakeSpec:
    def __init__(self, root: Any) -> None:
        self.run_id = "example"
        self.artifacts_dir = root / "artifacts"
        self.training = SimpleNamespace(
            total_transitions=100, checkpoint_milestones=(100,), save_final_checkpoint=True
        )
        self.distributed = SimpleNamespace(strict_update_budget=True)
        self.evaluation = SimpleNamespace(
            trials_per_map=30, maps=[SimpleNamespace(id="map", expected_map_uid="uid")]
        )
        self.components = SimpleNamespace(learner=SimpleNamespace(class_path="irrelevant"))

    def model_copy(self, update: Any) -> Any:
        result = FakeSpec(self.artifacts_dir.parent)
        result.run_id = update["run_id"]
        return result


class JobTests(unittest.TestCase):
    def test_disk_monitor_requests_stop_and_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            spec, stop = FakeSpec(root), root / "STOP"
            checked = threading.Event()

            def out_of_space(actual_spec: Any, base: Any) -> Never:
                assert actual_spec is spec
                assert base == root
                checked.set()
                raise OSError("insufficient free disk space")

            with patch(
                "tools.headless_campaign_queue.job.check_disk_space", side_effect=out_of_space
            ):
                with (
                    pytest.raises(RuntimeError, match="disk reserve check failed"),
                    monitor_disk(spec, root, stop, interval=0.001),
                ):
                    assert checked.wait(timeout=2)
                assert stop.exists()

    def test_insufficient_disk_prevents_training_process_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = root / "run.yaml"
            config.write_text("config")
            spec = FakeSpec(root)
            env = {
                "TRACKMANIARL_QUEUE_JOB_ID": "example",
                "TRACKMANIARL_QUEUE_MANIFEST_SHA256": "a" * 64,
            }
            with (
                patch.dict(os.environ, env),
                patch("trackmaniarl.core.spec.RunSpec.from_yaml", return_value=spec),
                patch(
                    "tools.headless_campaign_queue.job.check_disk_space",
                    side_effect=OSError("insufficient free disk space"),
                ) as disk,
                patch("tools.headless_campaign_queue.job.subprocess.call") as start,
            ):
                with pytest.raises(OSError, match="insufficient free disk space"):
                    execute(
                        config,
                        root / "receipt.json",
                        100,
                        root / "STOP",
                        max_overshoot=0,
                        expected_config_sha=hashlib.sha256(config.read_bytes()).hexdigest(),
                        expected_fingerprint="fp",
                    )
                disk.assert_called_once_with(spec, root)
                start.assert_not_called()
                assert not (root / "receipt.json").exists()

    def test_changed_configuration_fails_before_process_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = root / "run.yaml"
            config.write_text("new content")
            env = {
                "TRACKMANIARL_QUEUE_JOB_ID": "example",
                "TRACKMANIARL_QUEUE_MANIFEST_SHA256": "a" * 64,
            }
            with (
                patch.dict(os.environ, env),
                patch("tools.headless_campaign_queue.job.subprocess.call") as start,
            ):
                with pytest.raises(ValueError, match="changed since"):
                    execute(
                        config,
                        root / "receipt.json",
                        100,
                        root / "STOP",
                        max_overshoot=0,
                        expected_config_sha="wrong",
                        expected_fingerprint="fp",
                    )
                start.assert_not_called()

    def test_disk_failure_after_training_prevents_evaluation_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config, checkpoint, receipt = (
                root / "run.yaml",
                root / "checkpoint.pt",
                root / "receipt.json",
            )
            config.write_text("config")
            checkpoint.write_bytes(b"checkpoint")
            spec = FakeSpec(root)
            state = {"distributed": {"updates": 10}}
            env = {
                "TRACKMANIARL_QUEUE_JOB_ID": "example",
                "TRACKMANIARL_QUEUE_MANIFEST_SHA256": "a" * 64,
            }
            with (
                patch.dict(os.environ, env),
                patch("trackmaniarl.core.spec.RunSpec.from_yaml", return_value=spec),
                patch("trackmaniarl.core.fingerprint.run_fingerprint", return_value="fp"),
                patch("tools.headless_campaign_queue.job.subprocess.call", return_value=0) as train,
                patch(
                    "tools.headless_campaign_queue.job.final_checkpoint",
                    return_value=(checkpoint, 10),
                ),
                patch("trackmaniarl.core.builtins.TorchCheckpointCodec.load", return_value=state),
                patch("trackmaniarl.core.runtime.import_symbol", return_value=object),
                patch("tools.headless_campaign_queue.job.validate_checkpoint", return_value=100),
                patch("tools.headless_campaign_queue.job.validate_drained_journal"),
                patch(
                    "tools.headless_campaign_queue.job.check_disk_space",
                    side_effect=[[], OSError("insufficient free disk space")],
                ) as disk,
                patch("trackmaniarl.commands.evaluation._evaluate_checkpoint") as evaluate,
            ):
                with pytest.raises(OSError, match="insufficient free disk space"):
                    execute(
                        config,
                        receipt,
                        100,
                        root / "STOP",
                        max_overshoot=0,
                        expected_config_sha=hashlib.sha256(config.read_bytes()).hexdigest(),
                        expected_fingerprint="fp",
                    )
                train.assert_called_once()
                assert disk.call_count == 2
                evaluate.assert_not_called()
                assert not receipt.exists()

    def test_checkpoint_and_all_trials_are_required_before_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config, stop, receipt = root / "run.yaml", root / "STOP", root / "receipt.json"
            config.write_text("config")
            spec = FakeSpec(root)
            checkpoint = (
                spec.artifacts_dir / spec.run_id / "checkpoints" / "distributed-update-00000010.pt"
            )
            eval_dir = spec.artifacts_dir / (spec.run_id + "-final-evaluation")
            trials = [
                {
                    "map_id": "map",
                    "map_uid": "uid",
                    "trial_index": n,
                    "telemetry_error": None,
                    "controller_error": None,
                }
                for n in range(30)
            ]

            def train(_args: Any) -> int:
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
                run_dir = checkpoint.parent.parent
                (run_dir / "events.jsonl").write_text(
                    json.dumps(
                        {
                            "event": "train/checkpoint_completed",
                            "payload": {"path": str(checkpoint)},
                            "step": 10,
                        }
                    )
                    + "\n"
                )
                journal = run_dir / "distributed" / "rollouts.sqlite3"
                journal.parent.mkdir()
                with sqlite3.connect(journal) as db:
                    db.execute("CREATE TABLE metadata(key TEXT, value TEXT)")
                    db.executemany(
                        "INSERT INTO metadata VALUES(?, ?)",
                        [("journal_id", "journal"), ("pruned_through", "3")],
                    )
                    db.execute("CREATE TABLE chunks(id INTEGER)")
                return 0

            def evaluate(_run: Any, path: Any) -> Any:
                eval_dir.mkdir(parents=True)
                (eval_dir / "evaluation.json").write_text(
                    json.dumps({"checkpoint": str(path), "trials": trials})
                )
                return trials, {}, str(path)

            state = {
                "run_fingerprint": "fp",
                "journal_id": "journal",
                "learner": {"value": 1},
                "distributed": {
                    "transitions": 103,
                    "updates": 10,
                    "update_credit": 0.75,
                    "journal_applied_frontier": 3,
                },
                "replay_store": {},
                "sampler": {},
            }
            env = {
                "TRACKMANIARL_QUEUE_JOB_ID": "example",
                "TRACKMANIARL_QUEUE_MANIFEST_SHA256": "a" * 64,
            }
            with (
                patch.dict(os.environ, env),
                patch("trackmaniarl.core.spec.RunSpec.from_yaml", return_value=spec),
                patch("trackmaniarl.core.fingerprint.run_fingerprint", return_value="fp"),
                patch("trackmaniarl.core.builtins.TorchCheckpointCodec.load", return_value=state),
                patch("trackmaniarl.core.runtime.import_symbol", return_value=object),
                patch(
                    "trackmaniarl.core.runtime.resolve_run",
                    return_value=SimpleNamespace(logger=SimpleNamespace(close=lambda: None)),
                ),
                patch("trackmaniarl.commands.evaluation._configure_evaluation_stop"),
                patch(
                    "trackmaniarl.commands.evaluation._evaluate_checkpoint", side_effect=evaluate
                ),
                patch("tools.headless_campaign_queue.job.subprocess.call", side_effect=train),
                patch("tools.headless_campaign_queue.job.check_disk_space") as disk,
            ):
                execute(
                    config,
                    receipt,
                    100,
                    stop,
                    max_overshoot=5,
                    expected_config_sha=hashlib.sha256(config.read_bytes()).hexdigest(),
                    expected_fingerprint="fp",
                )
                assert disk.call_count == 2
                assert all(call.args == (spec, root) for call in disk.call_args_list)
            actual = json.loads(receipt.read_text())
            assert actual["process_pid"] == os.getpid()
            assert actual["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
            assert actual["transitions"] == 103
            assert actual["requested_transitions"] == 100


if __name__ == "__main__":
    unittest.main()
