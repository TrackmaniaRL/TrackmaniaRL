import unittest
from pathlib import Path

from tools.headless_campaign_queue.verify_artifacts import validate_checkpoint, validate_evaluation


class ArtifactTests(unittest.TestCase):
    def test_checks_complete_off_policy_state(self):
        state = {"run_fingerprint": "fp", "learner": {"x": 1}, "distributed": {"transitions": 50, "updates": 5}, "replay_store": {}, "sampler": {}}
        validate_checkpoint(state, "fp", 50, on_policy=False)
        for field, value in (("run_fingerprint", "other"), ("learner", {}), ("replay_store", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_checkpoint({**state, field: value}, "fp", 50, on_policy=False)
        with self.assertRaises(ValueError):
            validate_checkpoint(state, "fp", 51, on_policy=False)

    def test_checks_ppo_counter(self):
        state = {"run_fingerprint": "fp", "learner": {"x": 1}, "counters": {"transitions": 50, "updates": 1}}
        validate_checkpoint(state, "fp", 50, on_policy=True)
        with self.assertRaises(ValueError):
            validate_checkpoint(state, "fp", 50, on_policy=False)

    def test_evaluation_includes_failures_and_unique_indices(self):
        artifact = {"checkpoint": str(Path("/tmp/final.pt")), "trials": [{"map_id": "map", "map_uid": "uid", "trial_index": 0, "finished": False}, {"map_id": "map", "map_uid": "uid", "trial_index": 1, "finished": True}]}
        validate_evaluation(artifact, Path("/tmp/final.pt"), "map", "uid", 2)
        artifact["trials"][1]["trial_index"] = 0
        with self.assertRaises(ValueError):
            validate_evaluation(artifact, Path("/tmp/final.pt"), "map", "uid", 2)


if __name__ == "__main__":
    unittest.main()
