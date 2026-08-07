import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from policy.GeometryFlow.hand_object_frame import (
    active_hand_object_frame11,
    active_hand_object_frame11_from_goal_relative,
)
from policy.GeometryFlow.online_ndf_functional_frame import relative_frame9_metric


def pose9(position, rotation):
    matrix = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    return np.concatenate((position, matrix[:, :2].reshape(6))).astype(np.float32)


class HandObjectFrameTest(unittest.TestCase):
    def test_goal_relative_reconstruction_matches_direct_frame(self):
        current_position = np.asarray([0.3, -0.2, 0.8], dtype=np.float32)
        current_rotation = Rotation.from_euler("xyz", [20, -15, 35], degrees=True).as_matrix()
        goal_position = np.asarray([0.4, -0.1, 0.75], dtype=np.float32)
        goal_rotation = Rotation.from_euler("xyz", [-5, 10, 70], degrees=True).as_matrix()
        relative = relative_frame9_metric(
            current_position, current_rotation, goal_position, goal_rotation
        )
        left_rotation = Rotation.from_euler("xyz", [5, 30, -20], degrees=True).as_matrix()
        right_rotation = np.eye(3, dtype=np.float32)
        state = np.concatenate(
            (
                pose9(np.asarray([0.2, -0.22, 0.78]), left_rotation),
                [0.0],
                pose9(np.asarray([-0.2, 0.1, 0.7]), right_rotation),
                [1.0],
            )
        ).astype(np.float32)
        direct = active_hand_object_frame11(
            current_position, current_rotation, state
        )
        reconstructed = active_hand_object_frame11_from_goal_relative(
            pose9(goal_position, goal_rotation), relative, state
        )
        np.testing.assert_allclose(reconstructed, direct, atol=1e-5)
        np.testing.assert_array_equal(reconstructed[-2:], [1.0, 0.0])

    def test_selects_right_closed_gripper(self):
        identity = np.eye(3, dtype=np.float32)
        state = np.concatenate(
            (
                pose9(np.asarray([-1.0, 0.0, 0.0]), identity),
                [1.0],
                pose9(np.asarray([0.1, 0.0, 0.0]), identity),
                [0.0],
            )
        ).astype(np.float32)
        token = active_hand_object_frame11(
            np.asarray([0.3, 0.0, 0.0]), identity, state
        )
        np.testing.assert_allclose(token[:3], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_array_equal(token[-2:], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
