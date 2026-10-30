import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.headless_campaign_queue.campaign import prepare


class CampaignTests(unittest.TestCase):
    def test_not_qualified_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "campaign.json"
            path.write_text(json.dumps({"status": "not_runnable", "jobs": []}))
            with self.assertRaisesRegex(ValueError, "not qualified"):
                prepare(path)

    def test_fifth_config_must_validate_before_any_queue_is_built(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            jobs = []
            for name in ("qr", "iqn", "sac", "tqc", "ppo"):
                config = root / (name + ".yaml")
                config.write_text(name)
                jobs.append({"id": name, "config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "receipt": str(root / (name + ".json")), "stop_file": str(root / (name + ".STOP")), "run_fingerprint": "fp"})
            jobs[-1]["config_sha256"] = "wrong"
            evidence = {j["id"]: {"config_sha256": j["config_sha256"], "run_fingerprint": "fp", "native_action_ack": "pass", "checkpoint_restore": "pass", "evaluation": "pass", "clean_process_shutdown": "pass"} for j in jobs}
            report = root / "qualification.json"
            report.write_text(json.dumps({"kind": "headless-five-qualification-v1", "jobs": evidence}))
            path = root / "campaign.json"
            path.write_text(json.dumps({"schema_version": 3, "status": "qualified", "seed": 17, "transitions": 100, "map_uid": "uid", "jobs": jobs, "qualification_report": str(report), "qualification_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}))
            def spec(config):
                name = Path(config).stem
                learner = {"qr": "DiscreteValueLearner", "iqn": "DiscreteValueLearner", "sac": "SoftActorCritic", "tqc": "TruncatedQuantileCritic", "ppo": "ProximalPolicyOptimization"}[name]
                prefix = "trackmaniarl.algorithms.value_based:" if name in ("qr", "iqn") else "trackmaniarl.algorithms:"
                head = "FixedQuantileHead" if name == "qr" else "DuelingImplicitQuantileHead"
                components = SimpleNamespace(learner=SimpleNamespace(class_path=prefix + learner), model_factory=SimpleNamespace(kwargs={"head": {"class_path": "example:" + head}}), environment=SimpleNamespace(class_path="trackmaniarl.trackmania.native_headless:Factory"), evaluator=object())
                return SimpleNamespace(seed=17, training=SimpleNamespace(total_transitions=100, checkpoint_milestones=(100,)), evaluation=SimpleNamespace(maps=[SimpleNamespace(expected_map_uid="uid")], trials_per_map=30), components=components, artifacts_dir=root / "artifacts", checkpoint_root=None, run_id=name)
            with patch("trackmaniarl.core.spec.RunSpec.from_yaml", side_effect=spec), patch("trackmaniarl.core.fingerprint.run_fingerprint", return_value="fp"):
                with self.assertRaisesRegex(ValueError, "ppo: configuration"):
                    prepare(path)
            self.assertFalse((root / "artifacts").exists())


if __name__ == "__main__":
    unittest.main()
