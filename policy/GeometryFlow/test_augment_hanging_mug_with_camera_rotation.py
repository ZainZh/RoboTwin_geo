from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .augment_hanging_mug_with_camera_rotation import (
    compose_goal_rotation_frame,
    compose_rotation_token,
    stabilize_frame_symmetry,
    trimmed_icp,
)
from .functional_action_frame import frame9_rotation


class HangingMugCameraRotationTest(unittest.TestCase):
    def test_trimmed_icp_recovers_close_rigid_transform_with_outliers(self):
        generator = np.random.default_rng(41)
        source = generator.normal(size=(200, 3)) * np.asarray([0.08, 0.03, 0.12])
        expected_rotation = Rotation.from_euler("y", 8.0, degrees=True).as_matrix()
        expected_translation = np.asarray([0.13, -0.04, 0.02])
        target = source @ expected_rotation.T + expected_translation
        target[:20] += generator.normal(size=(20, 3)) * 0.05
        actual_rotation, actual_translation, _ = trimmed_icp(source, target)
        error = Rotation.from_matrix(actual_rotation @ expected_rotation.T).magnitude()
        self.assertLess(np.degrees(error), 1.0)
        np.testing.assert_allclose(actual_translation, expected_translation, atol=5e-3)

    def test_composition_encodes_goal_times_current_inverse_and_zero_translation(self):
        current = Rotation.from_euler("xyz", [5.0, -4.0, 20.0], degrees=True).as_matrix()
        goal = Rotation.from_euler("xyz", [-3.0, 8.0, 30.0], degrees=True).as_matrix()
        token = compose_rotation_token(current[None], goal[None])
        np.testing.assert_array_equal(token[0, :3], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(
            frame9_rotation(token)[0], goal @ current.T, atol=1e-6
        )

    def test_temporal_stabilization_removes_proper_axis_flip_without_labels(self):
        frames = Rotation.from_euler(
            "y", [0.0, 5.0, 10.0], degrees=True
        ).as_matrix()
        frames[1] = frames[1] @ np.diag([-1.0, -1.0, 1.0])
        output, confidence, variants, step, margin = stabilize_frame_symmetry(
            frames,
            np.zeros(3, dtype=np.int64),
            np.arange(3, dtype=np.int64),
        )
        expected = Rotation.from_euler("y", [0.0, 5.0, 10.0], degrees=True).as_matrix()
        np.testing.assert_allclose(output, expected, atol=1e-6)
        self.assertEqual(int(variants[1]), 3)
        np.testing.assert_array_equal(confidence, np.asarray([0.0, 1.0, 1.0]))
        self.assertLess(float(step[1]), 6.0)
        self.assertGreater(float(margin[1]), 100.0)

    def test_goal_frame_exposes_only_camera_rotation(self):
        goal = Rotation.from_euler("xyz", [5.0, 8.0, -12.0], degrees=True).as_matrix()
        frame = compose_goal_rotation_frame(goal[None])
        np.testing.assert_array_equal(frame[0, :3], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(frame9_rotation(frame)[0], goal, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
