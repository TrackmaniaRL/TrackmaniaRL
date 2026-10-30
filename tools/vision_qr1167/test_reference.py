import unittest
from tools.vision_qr1167.reference import Reference


class ReferenceTests(unittest.TestCase):
    def test_crossing_does_not_jump_forward(self):
        points = [(i, 0, 0) for i in range(40)] + [(40, 0, i) for i in range(40)]
        points += [(40-i, 0, 40-i) for i in range(41)]
        rows = [dict(raceTick=i*50, position=dict(zip(('x','y','z'), p))) for i,p in enumerate(points)]
        ref = Reference(rows)
        ref.reset(0, dict(x=0,y=0,z=0))
        progress, _ = ref.project(dict(x=1,y=0,z=0), .05)
        self.assertLess(progress, 3)
        self.assertLess(ref.index, 3)

    def test_projection_cannot_exceed_reward_admission_limit(self):
        from tools.vision_qr1167.curriculum import ProgressReward
        rows = [dict(raceTick=i*50, position=dict(x=x,y=0,z=0))
                for i,x in enumerate((0, 5, 11.5, 20))]
        ref = Reference(rows)
        distance, _ = ref.reset(0, dict(x=0,y=0,z=0))
        reward = ProgressReward(distance)
        distance, lateral = ref.project(dict(x=11.5,y=0,z=0), .05)
        self.assertLessEqual(distance, 11)
        self.assertLessEqual(reward.step(distance,lateral,.05), 1.05)

    def test_rejects_unordered_reference(self):
        rows = [dict(raceTick=t, position=dict(x=0,y=0,z=0)) for t in (0,10,5)]
        with self.assertRaises(ValueError):
            Reference(rows)


if __name__ == '__main__':
    unittest.main()
