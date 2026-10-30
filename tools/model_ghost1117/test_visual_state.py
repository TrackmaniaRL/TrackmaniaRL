"""Reject altered ownership, clocks, and visual data beyond the native grid."""

import struct
import unittest

from tools.model_ghost1117.visual_state import float32, quantized_pose_agrees, visual_records


class RecordedVisualEvidenceTests(unittest.TestCase):
    def test_internal_half_grid_boundary_does_not_imply_pose_divergence(self) -> None:
        value = 503.8905029296875
        self.assertTrue(quantized_pose_agrees(value, float32(503890 * float32(0.001))))
        self.assertTrue(quantized_pose_agrees(value, float32(503891 * float32(0.001))))
        self.assertFalse(quantized_pose_agrees(value, float32(503892 * float32(0.001))))

    def test_rejects_nonfinite_and_non_grid_values(self) -> None:
        self.assertFalse(quantized_pose_agrees(5.0, float("nan")))
        self.assertFalse(quantized_pose_agrees(5.0, 5.0001))
        self.assertFalse(quantized_pose_agrees(5.0, 5.002))

    def test_rejects_wrong_owner_and_clock(self) -> None:
        row = bytearray(1280)
        row[:8] = b"TMVH0001"
        struct.pack_into("<4I", row, 8, 1280, 1, 461, 461)
        struct.pack_into("<I", row, 32, 2200)
        with self.assertRaisesRegex(ValueError, "owner"):
            visual_records(bytes(row), 462)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            visual_records(bytes(row), 461)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            visual_records(bytes(row[:-1]), 461)


if __name__ == "__main__":
    unittest.main()
