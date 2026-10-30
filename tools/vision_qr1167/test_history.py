import unittest
import numpy as np
from tools.vision_qr1167.build_demonstrations import history_window


class HistoryTests(unittest.TestCase):
    def test_initial_history_repeats_real_image_only(self):
        frames=[np.full((84,84),x,np.uint8) for x in (7,11,13,17)]
        self.assertEqual(history_window(frames,0)[:,0,0].tolist(),[7,7,7,7])
        self.assertEqual(history_window(frames,1)[:,0,0].tolist(),[7,7,7,11])
        self.assertEqual(history_window(frames,3)[:,0,0].tolist(),[7,11,13,17])


if __name__=='__main__':unittest.main()
