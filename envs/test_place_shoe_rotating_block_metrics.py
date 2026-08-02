import unittest

import numpy as np

from .place_shoe_rotating_block import place_shoe_rotating_block


class _FakePose:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=np.float64)
        self.q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class _FakeActor:
    def __init__(self, position):
        self.pose = _FakePose(position)

    def get_functional_point(self, *_args, **_kwargs):
        return self.pose


class TestPlaceShoeRotatingBlockMetrics(unittest.TestCase):
    def test_metrics_track_final_and_best_pose_errors(self):
        task = place_shoe_rotating_block.__new__(place_shoe_rotating_block)
        task.shoe = _FakeActor([0.04, 0.0, 0.0])
        task.target_block = _FakeActor([0.0, 0.0, 0.0])
        task.shoe_id = 3
        task.take_action_cnt = 37
        task._min_translation_error_norm_m = float("inf")
        task._min_rotation_error_deg = float("inf")

        self.assertTrue(task.check_success())
        task.shoe.pose.p[0] = 0.06
        self.assertFalse(task.check_success())

        metrics = task.get_evaluation_metrics()
        self.assertAlmostEqual(metrics["translation_error_norm_m"], 0.06)
        self.assertAlmostEqual(metrics["min_translation_error_norm_m"], 0.04)
        self.assertAlmostEqual(metrics["rotation_error_deg"], 0.0)
        self.assertAlmostEqual(metrics["min_rotation_error_deg"], 0.0)
        self.assertEqual(metrics["shoe_id"], 3)
        self.assertEqual(metrics["policy_steps"], 37)


if __name__ == "__main__":
    unittest.main()
