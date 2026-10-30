import random
import unittest
from tools.vision_qr1167.curriculum import choose_start, ProgressReward, validate_step


class CurriculumTests(unittest.TestCase):
    def test_evaluation_never_restores_or_perturbs(self):
        for seed in range(100):
            start = choose_start(['middle', 'finish'], random.Random(seed), evaluation=True)
            self.assertEqual((start.snapshot, start.perturb_ticks), (None, 0))

    def test_training_covers_full_start_and_suffixes(self):
        rng = random.Random(7)
        starts = [choose_start(['middle', 'finish'], rng) for _ in range(100)]
        self.assertEqual({s.snapshot for s in starts}, {None, 'middle', 'finish'})
        self.assertTrue(all(abs(s.perturb_steer) <= .12 and 1 <= s.perturb_ticks <= 5 for s in starts))

    def test_no_oscillation_reward(self):
        reward = ProgressReward(0)
        reward.step(1, 0, .05)
        self.assertLess(reward.step(0, 0, .05) + reward.step(1, 0, .05), 0)

    def test_no_corridor_boundary_reward_farming(self):
        reward = ProgressReward(0)
        reward.step(1, 15, .05)
        self.assertLess(reward.step(1, 0, .05), 0)

    def test_reject_restore_and_repeated_goal(self):
        reward = ProgressReward(0)
        with self.assertRaises(ValueError):
            reward.step(100, 0, .05)
        reward.step(1, 0, .05, goal=True)
        with self.assertRaises(RuntimeError):
            reward.step(1, 0, .05, goal=True)

    def test_rejected_progress_preserves_reward_state_and_allows_reset(self):
        from tools.vision_qr1167.curriculum import ProgressDiscontinuity
        reward = ProgressReward(10)
        with self.assertRaises(ProgressDiscontinuity):
            reward.step(22, 0, .05)
        self.assertEqual(reward.maximum, 10)
        self.assertFalse(reward.done)
        fresh_episode = ProgressReward(0)
        self.assertAlmostEqual(fresh_episode.step(1, 0, .05), .05)

    def test_reject_stale_image_and_wrong_action(self):
        before = dict(map_uid='map', episode_id=1, race_tick=1)
        after = dict(before, race_tick=6, paused=True, render_tick=6,
                     applied_action_id=8, requested_action_id=8)
        validate_step(before, after, 5, 'map')
        for changes in ({'render_tick': 5}, {'applied_action_id': 9},
                        {'race_tick': 7}, {'episode_id': 2}, {'paused': False}):
            with self.assertRaises(ValueError):
                validate_step(before, dict(after, **changes), 5, 'map')


if __name__ == '__main__':
    unittest.main()
