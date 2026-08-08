import unittest

import numpy as np

from .analyze_handle_aware_frame import (
    frame_from_vertical_handle,
    shuffle_episode_axis,
    signed_axis_angles,
    single_axis_frame,
)


class TestHandleAwareFrameAnalysis(unittest.TestCase):
    def test_frame_from_vertical_handle_recovers_identity(self):
        frame = frame_from_vertical_handle(
            np.asarray([[0.0, 1.0, 0.0]]),
            np.asarray([[0.0, 0.0, 1.0]]),
        )
        np.testing.assert_allclose(frame, np.eye(3)[None], atol=1e-8)

    def test_single_axis_prior_is_orthogonalized(self):
        frame = single_axis_frame(
            np.asarray([[0.0, 0.0, 1.0]]),
            np.asarray([0.1, 1.0, 0.2]),
            axis_name="handle",
        )
        np.testing.assert_allclose(
            frame @ np.swapaxes(frame, -1, -2), np.eye(3)[None], atol=1e-8
        )
        np.testing.assert_allclose(np.linalg.det(frame), 1.0, atol=1e-8)

    def test_axis_metric_is_signed(self):
        actual = signed_axis_angles(
            np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(actual, [0.0, 180.0])

    def test_episode_shuffle_preserves_rows_but_changes_episode_source(self):
        axis = np.arange(18, dtype=np.float64).reshape(6, 3)
        episodes = np.asarray([0, 0, 1, 1, 1, 2])
        shuffled = shuffle_episode_axis(axis, episodes, np.arange(6))
        np.testing.assert_allclose(shuffled[:2], axis[[2, 4]])
        np.testing.assert_allclose(shuffled[2:5], axis[[5, 5, 5]])
        np.testing.assert_allclose(shuffled[5:], axis[[0]])


if __name__ == "__main__":
    unittest.main()
