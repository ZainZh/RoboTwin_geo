import unittest

import numpy as np

from .augment_handled_mug_camera_goal import (
    MARKER_TO_MUG_RAW_ROTATION,
    marker_frame9_to_mug_goal_frame9,
)
from .functional_action_frame import frame9_rotation


class TestHandledMugCameraGoal(unittest.TestCase):
    def test_declared_axis_mapping_and_clearance(self):
        marker = np.asarray(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
        )
        goal = marker_frame9_to_mug_goal_frame9(marker, clearance_m=0.03)
        np.testing.assert_allclose(goal[0, :3], [1.0, 2.0, 3.03])
        rotation = frame9_rotation(goal)[0]
        np.testing.assert_allclose(rotation, MARKER_TO_MUG_RAW_ROTATION)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        self.assertGreater(float(np.linalg.det(rotation)), 0.999999)

    def test_wrong_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"shape \[N,9\]"):
            marker_frame9_to_mug_goal_frame9(np.zeros((2, 8)))


if __name__ == "__main__":
    unittest.main()
