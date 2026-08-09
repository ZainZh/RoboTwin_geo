import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .train_interaction_flow_tokens import task_aligned_point_coordinates


def frame9(position, rotation):
    return np.concatenate((position, rotation[:, 0], rotation[:, 1])).astype(
        np.float32
    )


class TaskAlignedPointCoordinatesTest(unittest.TestCase):
    def test_recovers_source_and_target_local_coordinates(self):
        source_rotation = Rotation.from_euler("xyz", (20, -10, 35), degrees=True).as_matrix()
        goal_rotation = Rotation.from_euler("xyz", (-5, 15, 80), degrees=True).as_matrix()
        source_position = np.asarray((0.2, -0.1, 0.4))
        goal_position = np.asarray((0.5, 0.3, 0.7))
        source_local = np.asarray(((0.01, 0.02, 0.03), (-0.02, 0.01, 0.04)))
        target_local = np.asarray(((0.03, -0.01, 0.02), (0.01, 0.04, -0.02)))
        points_a = source_local @ source_rotation.T + source_position
        points_b = target_local @ goal_rotation.T + goal_position
        relative_rotation = goal_rotation @ source_rotation.T
        relative = frame9(goal_position - source_position, relative_rotation)
        predicted_a, predicted_b = task_aligned_point_coordinates(
            points_a,
            points_b,
            frame9(goal_position, goal_rotation),
            relative,
            scale_m=0.1,
        )
        np.testing.assert_allclose(predicted_a, source_local / 0.1, atol=1e-6)
        np.testing.assert_allclose(predicted_b, target_local / 0.1, atol=1e-6)

    def test_rejects_nonpositive_scale(self):
        with self.assertRaises(ValueError):
            task_aligned_point_coordinates(
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                np.zeros(9),
                np.zeros(9),
                scale_m=0.0,
            )


if __name__ == "__main__":
    unittest.main()
