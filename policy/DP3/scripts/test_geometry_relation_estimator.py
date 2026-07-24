#!/usr/bin/env python3

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geometry_relation_estimator import (  # noqa: E402
    GeometryRelationEstimator,
    GeometryRelationPrediction,
    OBSERVATION_RELATION_ROUTES,
)
from ndf_goal_regressor import (  # noqa: E402
    GoalRelationRegressor,
    goal_matrix_to_target,
    invariant_geometry_extras,
    rotation_6d_to_matrix,
    target_to_goal_matrix,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform(points: np.ndarray, rotation: np.ndarray, translation) -> np.ndarray:
    xyz = points[:, :3] @ rotation.T + np.asarray(translation)[None, :]
    if points.shape[1] == 3:
        return xyz.astype(np.float32)
    return np.concatenate((xyz, points[:, 3:]), axis=1).astype(np.float32)


def _cloud(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(64, 3)).astype(np.float32)
    xyz *= np.asarray([0.12, 0.04, 0.025], dtype=np.float32)
    rgb = rng.uniform(0.0, 1.0, size=(64, 3)).astype(np.float32)
    return np.concatenate((xyz, rgb), axis=1)


class TestGeometryRelationEstimator(unittest.TestCase):
    def test_observation_estimator_api_has_no_id_or_pose_argument(self):
        parameters = inspect.signature(
            GeometryRelationEstimator.estimate_goal
        ).parameters
        self.assertEqual(
            list(parameters),
            ["self", "object_pointcloud_a", "object_pointcloud_b"],
        )
        self.assertEqual(OBSERVATION_RELATION_ROUTES, ("ndf_observation_goal",))

    def test_invariant_extras_ignore_world_rigid_transform(self):
        cloud_a = _cloud(1)
        cloud_b = _cloud(2)
        expected = invariant_geometry_extras(cloud_a, cloud_b)
        rotation = _rotation_z(1.17)
        actual = invariant_geometry_extras(
            _transform(cloud_a, rotation, [0.3, -0.4, 1.2]),
            _transform(cloud_b, rotation, [-0.7, 0.2, 0.5]),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_goal_target_round_trip(self):
        goal = np.eye(4, dtype=np.float64)
        goal[:3, :3] = _rotation_z(-0.73)
        goal[:3, 3] = [0.02, -0.03, 0.04]
        recovered = target_to_goal_matrix(goal_matrix_to_target(goal))
        np.testing.assert_allclose(recovered, goal, atol=1e-6)

    def test_rotation_6d_produces_valid_so3(self):
        rotation = rotation_6d_to_matrix(
            torch.tensor([[1.0, 0.2, 0.1, -0.3, 1.0, 0.4]])
        )[0].numpy()
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)

    def test_regressor_identity_initialization(self):
        model = GoalRelationRegressor(input_dim=12, hidden_dims=(16,), dropout=0.0)
        model.eval()
        output = model(torch.zeros((2, 12)))
        rotation = rotation_6d_to_matrix(output[:, 3:9]).detach().numpy()
        np.testing.assert_allclose(rotation, np.repeat(np.eye(3)[None], 2, axis=0))

    def test_prediction_validates_goal(self):
        prediction = GeometryRelationPrediction(goal_t_a_from_b=np.eye(4))
        np.testing.assert_allclose(prediction.goal_t_a_from_b, np.eye(4))
        with self.assertRaises(ValueError):
            GeometryRelationPrediction(goal_t_a_from_b=np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
