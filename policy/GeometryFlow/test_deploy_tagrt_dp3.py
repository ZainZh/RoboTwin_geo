from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .deploy_tagrt_dp3 import _frame9_rotation_angle_deg, _goal_frame9, _path_list
from diffusion_policy_3d.policy.dp3 import _se3_frame_invariants


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

    def test_frame9_rotation_angle(self):
        identity = np.concatenate(([0.0, 0.0, 0.0], np.eye(3)[:, 0], np.eye(3)[:, 1]))
        self.assertAlmostEqual(_frame9_rotation_angle_deg(identity), 0.0)
        angle = np.deg2rad(90.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        frame = np.concatenate(([0.0, 0.0, 0.0], rotation[:, 0], rotation[:, 1]))
        self.assertAlmostEqual(_frame9_rotation_angle_deg(frame), 90.0)

    def test_se3_retention_features_discard_axis_direction(self):
        frames = []
        for axis, translation in (("x", [0.03, 0.04, 0.0]), ("y", [0.0, 0.03, 0.04])):
            rotation = Rotation.from_euler(axis, 30.0, degrees=True).as_matrix()
            frames.append(
                np.concatenate((translation, rotation[:, 0], rotation[:, 1]))
            )
        invariants = _se3_frame_invariants(
            torch.as_tensor(np.asarray(frames), dtype=torch.float32)
        ).numpy()
        np.testing.assert_allclose(invariants[0], invariants[1], atol=1.0e-6)
        self.assertAlmostEqual(float(invariants[0, 0]), 0.05, places=6)
        self.assertAlmostEqual(
            float(np.rad2deg(invariants[0, 1])), 30.0, places=3
        )


if __name__ == "__main__":
    unittest.main()
