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


class RealZedInferencePointcloudTest(unittest.TestCase):
    def test_camera_frame_to_output_pc_filters_workspace_before_output_transform(self):
        from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds
        from script.real_zed_inference.real_dp3_inference import camera_frame_to_output_pc

        depth = np.ones((1, 2), dtype=np.float32)
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        rgb[0, 1] = [0, 255, 0]
        t_output_from_cam = np.eye(4, dtype=np.float32)
        t_output_from_cam[0, 3] = 10.0

        point_cloud = camera_frame_to_output_pc(
            camera_frame={"rgb": rgb, "depth_m": depth},
            camera_matrix=np.eye(3, dtype=np.float32),
            t_workspace_from_cam=np.eye(4, dtype=np.float32),
            workspace_bounds=WorkspaceBounds(
                x_min=-0.1,
                x_max=0.5,
                y_min=-0.1,
                y_max=0.5,
                z_min=0.5,
                z_max=1.5,
            ),
            t_output_from_cam=t_output_from_cam,
            min_depth_m=0.05,
            max_depth_m=3.0,
        )

        self.assertEqual(point_cloud.shape, (1, 6))
        np.testing.assert_allclose(point_cloud[0, :3], np.array([10.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(point_cloud[0, 3:6], np.array([1.0, 0.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
