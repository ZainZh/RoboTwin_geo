import unittest

import numpy as np

from .handled_mug_geometry import (
    CUP_FUNCTIONAL_ROTATION,
    mug_functional_matrix,
    handle_marker_functional_matrix,
    signed_axis_error_deg,
)


class TestHandledMugFunctionalFrame(unittest.TestCase):
    def test_mug_frame_is_right_handed_and_has_declared_axes(self):
        frame = mug_functional_matrix()
        np.testing.assert_allclose(frame[:3, 0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(frame[:3, 2], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(
            np.cross(frame[:3, 0], frame[:3, 1]), frame[:3, 2]
        )
        np.testing.assert_allclose(CUP_FUNCTIONAL_ROTATION.T @ CUP_FUNCTIONAL_ROTATION, np.eye(3))
        self.assertAlmostEqual(float(np.linalg.det(CUP_FUNCTIONAL_ROTATION)), 1.0)

    def test_marker_yaw_directly_controls_handle_axis(self):
        frame = handle_marker_functional_matrix(0.02, -0.01, np.pi / 2.0)
        np.testing.assert_allclose(frame[:3, 3], [0.02, -0.01, 0.0])
        np.testing.assert_allclose(frame[:3, 0], [0.0, 1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(frame[:3, 2], [0.0, 0.0, -1.0], atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(frame[:3, :3])), 1.0)

    def test_signed_handle_axis_distinguishes_180_degree_error(self):
        self.assertAlmostEqual(signed_axis_error_deg([1, 0, 0], [1, 0, 0]), 0.0)
        self.assertAlmostEqual(signed_axis_error_deg([1, 0, 0], [-1, 0, 0]), 180.0)


if __name__ == "__main__":
    unittest.main()
