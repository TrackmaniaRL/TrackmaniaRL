import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools.record_rl1168.learner import Learner
from tools.record_rl1168.model import build_model


class LearnerTests(unittest.TestCase):
    def test_zero_retention_is_pure_td_even_with_demonstrations(self):
        torch.set_num_threads(2)
        learner = Learner(
            {"state": build_model().state_dict(), "BC_updates": 0},
            device="cpu",
            retention_weight=0.0,
        )
        x = torch.randn(160, 47)
        y = torch.arange(160) % 68
        learner.demo = (x, y)
        for i in range(160):
            learner.add(x[i].numpy(), int(y[i]), 0.1, x[i].numpy(), i == 159)
        metrics = learner.update()
        assert learner.imitation_updates == 0
        assert metrics["auxiliary_weight"] == 0.0
        assert metrics["auxiliary_margin"] == 0.0
        assert learner.learner.update_count == 1

    def test_invalid_retention_rejected(self):
        for weight in [-1.0, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                Learner({}, device="cpu", retention_weight=weight)

    def test_joint_retention_has_only_one_adam_step(self):
        torch.set_num_threads(2)
        torch.manual_seed(1168)
        learner = Learner({"state": build_model().state_dict(), "BC_updates": 0}, device="cpu")
        x = torch.randn(160, 47)
        y = torch.arange(160) % 68
        learner.demo = (x, y)
        for i in range(160):
            learner.add(x[i].numpy(), int(y[i]), 0.1, x[i].numpy(), i == 159)
        optimizer = learner.learner.optimizer
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            learner.update()
            assert step.call_count == 1
        assert learner.imitation_updates == 1

    def test_failed_candidate_cannot_replace_proven_actor(self):
        learner = Learner({"state": build_model().state_dict(), "BC_updates": 0}, device="cpu")
        original = {k: v.clone() for k, v in learner.actor.state_dict().items()}
        changed = {k: v + 1 for k, v in original.items()}
        for result in [
            {"status": "DRIVING_FAILURE"},
            {
                "status": "NATIVE_GOAL",
                "actual_goal_ms": 18000,
                "clean_closure": True,
                "full_start": True,
                "training": False,
            },
            {
                "status": "NATIVE_GOAL",
                "actual_goal_ms": 17000,
                "clean_closure": True,
                "full_start": False,
                "training": False,
            },
        ]:
            assert not learner.consider_candidate(result, changed)
            assert all(torch.equal(v, original[k]) for k, v in learner.actor.state_dict().items())
        learner.learner.update_count = 32
        assert learner.consider_candidate(
            {
                "status": "NATIVE_GOAL",
                "actual_goal_ms": 17600,
                "clean_closure": True,
                "full_start": True,
                "training": False,
            },
            changed,
        )
        assert learner.accepted_RL_updates == 32

    def test_rewards_update_full_encoder_and_control_head(self):
        torch.set_num_threads(2)
        torch.manual_seed(1168)
        learner = Learner({"state": build_model().state_dict(), "BC_updates": 0}, device="cpu")
        random = np.random.default_rng(1168)
        learner.begin_episode()
        for i in range(160):
            state = random.normal(size=47).astype(np.float32)
            learner.add(state, i % 68, 0.1, state + 0.001, i == 159)
        before = {k: v.clone() for k, v in learner.learner.model.state_dict().items()}
        learner.update()
        assert learner.learner.update_count == 1
        assert learner.imitation_updates == 0
        changed = [
            k
            for k, v in learner.learner.model.state_dict().items()
            if not torch.equal(v, before[k])
        ]
        assert any("encoder" in k for k in changed), changed
        assert any("head" in k for k in changed), changed
        assert all(torch.isfinite(v).all() for v in learner.learner.model.parameters())
        saved = learner.checkpoint()
        assert saved["RL_updates"] == 1
        assert saved["transitions"] == 160
        assert "optimizer" in str(saved["learner"].keys()).lower()


if __name__ == "__main__":
    unittest.main()
