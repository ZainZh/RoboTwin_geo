from __future__ import annotations

import unittest

import numpy as np

from .deploy_tagrt_dp3 import _goal_frame9, _path_list


class TagrtDP3DeployHelpersTest(unittest.TestCase):
    def test_goal_frame_columns(self):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            _goal_frame9(transform),
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

    def test_path_list(self):
        self.assertEqual(_path_list("a,b"), ["a", "b"])
        self.assertEqual(_path_list(["a", "b"]), ["a", "b"])
        self.assertEqual(_path_list(""), [])
        self.assertEqual(_path_list(None), [])


if __name__ == "__main__":
    unittest.main()
