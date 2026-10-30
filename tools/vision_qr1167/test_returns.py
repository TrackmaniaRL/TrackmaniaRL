import unittest
from tools.vision_qr1167.returns import driving_return


def row(reward, goal=False, reset=False):
    return ('image',8,reward,'next-image',goal,reset)


class ReturnTests(unittest.TestCase):
    def test_stall_reset_has_no_imaginary_continuation_value(self):
        result=driving_return([row(-.05,reset=True)],[None],0)
        self.assertTrue(result[4])
        self.assertEqual(result[5],0)

    def test_five_step_return_uses_actual_rewards_and_matching_discount(self):
        result=driving_return([row(i) for i in range(1,7)],[1]*6,0)
        self.assertAlmostEqual(result[2],sum((.997**i)*(i+1) for i in range(5)))
        self.assertAlmostEqual(result[5],.997**5)
        self.assertEqual(result[6],5)

    def test_goal_stops_return_before_next_reset(self):
        result=driving_return([row(1),row(10,goal=True),row(999)],[1,1,2],0)
        self.assertAlmostEqual(result[2],1+.997*10)
        self.assertEqual(result[5],0)
        self.assertEqual(result[6],2)

    def test_unobserved_reset_boundary_never_splices_two_episodes(self):
        result=driving_return([row(1),row(999)],[1,2],0)
        self.assertEqual(result[2],1)
        self.assertAlmostEqual(result[5],.997)
        self.assertEqual(result[6],1)

    def test_legacy_without_episode_provenance_stays_one_step(self):
        result=driving_return([row(1),row(999)],[None,None],0)
        self.assertEqual(result[2],1)
        self.assertEqual(result[6],1)


if __name__=='__main__':
    unittest.main()
