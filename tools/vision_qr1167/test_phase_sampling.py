import unittest
import numpy as np
from tools.vision_qr1167.retention import phase_balanced_indices


class PhaseSamplingTests(unittest.TestCase):
    def test_sparse_start_phase_is_not_drowned_by_dense_later_phase(self):
        phases=[np.array([0]),np.arange(1,10),np.array([10])]
        selected=phase_balanced_indices(phases,0,6).tolist()
        self.assertEqual(selected,[0,1,10,0,2,10])

    def test_counter_resume_is_deterministic_and_cycles_phase_examples(self):
        phases=[np.array([0]),np.arange(1,4)]
        self.assertEqual(phase_balanced_indices(phases,1,4).tolist(),[0,3,0,1])
        self.assertTrue(np.array_equal(phase_balanced_indices(phases,6304,16),phase_balanced_indices(phases,6304,16)))


if __name__=='__main__':unittest.main()
