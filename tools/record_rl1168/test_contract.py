import itertools
import unittest

import numpy as np
import pytest

from tools.record_rl1168.features import RouteFeatures, shaped_reward
from tools.record_rl1168.model import ACTIONS


class ContractTests(unittest.TestCase):
    def state(self, x=0, clock=0):
        state = np.zeros(22, dtype=np.float32)
        state[0] = clock
        state[1:10] = np.eye(3).reshape(-1)
        state[10] = x / 1000
        return state

    def test_clock_is_not_a_policy_input(self):
        route = np.array([[x, 0, 0] for x in range(140)])
        a = RouteFeatures(route).observe(self.state(clock=0))[0]
        b = RouteFeatures(route).observe(self.state(clock=20))[0]
        np.testing.assert_array_equal(a, b)
        assert a.shape == (47,)

    def test_route_preview_depends_on_actual_position(self):
        route = np.array([[x, 0, 0] for x in range(140)])
        a = RouteFeatures(route).observe(self.state(5))[0]
        b = RouteFeatures(route).observe(self.state(15))[0]
        assert b[-2] > a[-2]

    def test_unavailable_filtered_steering_cannot_leak_into_policy(self):
        route = np.array([[x, 0, 0] for x in range(140)])
        a = self.state()
        b = self.state()
        b[-1] = 0.75
        np.testing.assert_array_equal(
            RouteFeatures(route).observe(a)[0], RouteFeatures(route).observe(b)[0]
        )

    def test_complete_lap_return_rewards_time_not_reference_length(self):
        for positions in ([0, 1, 2, 3], [0, 2, 1, 3]):
            rewards = [
                shaped_reward(a, b, goal=i == 2)
                for i, (a, b) in enumerate(itertools.pairwise(positions))
            ]
            self.assertAlmostEqual(sum(rewards), 100 - 0.03)

    def test_progress_jump_is_not_training_data(self):
        with pytest.raises(ValueError):
            shaped_reward(0, 100)

    def test_full_control_action_catalog(self):
        assert len(ACTIONS) == 68
        for control in ((-1, 1, 0), (1, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)):
            assert control in ACTIONS


if __name__ == "__main__":
    unittest.main()
