import importlib.util
import unittest
from pathlib import Path

import numpy as np


METRICS_PATH = Path(__file__).resolve().parents[1] / "envs" / "placement_metrics.py"
SPEC = importlib.util.spec_from_file_location("robotwin_placement_metrics", METRICS_PATH)
METRICS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(METRICS)


class TestPlacementMetrics(unittest.TestCase):
    def aligned(self, position, quaternion=(1.0, 0.0, 0.0, 0.0)):
        return METRICS.functional_pose_alignment_success(
            position,
            quaternion,
            [0.0, 0.0, 0.75],
            [1.0, 0.0, 0.0, 0.0],
            position_tolerance=(0.05, 0.03, 0.04),
            min_quaternion_alignment=0.98,
        )

    def test_aligned_pose_succeeds_without_any_gripper_state(self):
        self.assertTrue(self.aligned([0.01, 0.01, 0.76]))

    def test_pose_above_target_fails(self):
        self.assertFalse(self.aligned([0.01, 0.01, 0.80]))

    def test_wrong_xy_or_orientation_fails(self):
        self.assertFalse(self.aligned([0.06, 0.0, 0.75]))
        self.assertFalse(self.aligned([0.0, 0.0, 0.75], quaternion=[0.0, 1.0, 0.0, 0.0]))

    def test_quaternion_sign_is_equivalent(self):
        self.assertTrue(self.aligned([0.0, 0.0, 0.75], quaternion=[-1.0, 0.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()

