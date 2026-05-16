import unittest

import numpy as np

from script.real_zed_collection.check_dobot_fk_consistency import (
    arm_pose_error,
    offline_dobot_fk_matrix,
    transform_from_xyz_euler_deg,
)


class TestDobotFkConsistencyHelpers(unittest.TestCase):
    def test_offline_fk_returns_valid_transform(self):
        transform = offline_dobot_fk_matrix(np.zeros(6, dtype=np.float64), tool_z_m=0.197)

        self.assertEqual(transform.shape, (4, 4))
        np.testing.assert_allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9)
        np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-9)

    def test_tool_offset_changes_tcp_position(self):
        joints = np.deg2rad(np.array([12.0, -20.0, 35.0, 7.0, -15.0, 45.0], dtype=np.float64))

        flange = offline_dobot_fk_matrix(joints, tool_z_m=0.0)
        tcp = offline_dobot_fk_matrix(joints, tool_z_m=0.197)

        offset = tcp[:3, 3] - flange[:3, 3]
        self.assertAlmostEqual(float(np.linalg.norm(offset)), 0.197, places=6)

    def test_pose_error_is_zero_for_identical_pose(self):
        transform = transform_from_xyz_euler_deg([100.0, -200.0, 300.0, 10.0, -20.0, 30.0])

        trans_error, rot_error = arm_pose_error(transform, transform)

        self.assertAlmostEqual(trans_error, 0.0, places=9)
        self.assertAlmostEqual(rot_error, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
