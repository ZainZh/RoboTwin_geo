import argparse
import unittest

import numpy as np


class RealZedInferenceActionTest(unittest.TestCase):
    def test_action_delta_limit_clips_arm_joints_and_grippers_separately(self):
        from script.real_zed_inference.real_dp3_inference import limit_action_delta_for_execution

        last_action = np.zeros(14, dtype=np.float32)
        action = np.ones(14, dtype=np.float32)
        args = argparse.Namespace(max_executed_joint_delta=0.12, max_executed_gripper_delta=0.2)

        clipped = limit_action_delta_for_execution(action, last_action, args)

        arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        np.testing.assert_allclose(clipped[arm_indices], np.full((12,), 0.12, dtype=np.float32))
        self.assertAlmostEqual(float(clipped[6]), 0.2)
        self.assertAlmostEqual(float(clipped[13]), 0.2)

    def test_action_delta_limit_can_be_disabled(self):
        from script.real_zed_inference.real_dp3_inference import limit_action_delta_for_execution

        last_action = np.zeros(14, dtype=np.float32)
        action = np.ones(14, dtype=np.float32)
        args = argparse.Namespace(max_executed_joint_delta=0.0, max_executed_gripper_delta=0.0)

        clipped = limit_action_delta_for_execution(action, last_action, args)

        np.testing.assert_allclose(clipped, action)


if __name__ == "__main__":
    unittest.main()
