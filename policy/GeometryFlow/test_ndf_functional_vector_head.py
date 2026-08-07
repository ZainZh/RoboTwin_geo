from __future__ import annotations

import unittest

import numpy as np
import torch

from .train_ndf_functional_vector_head import (
    axis_dataset,
    derangement,
    functional_axis_loss,
    normalize_ndf_clouds,
)


class NdfFunctionalVectorHeadTest(unittest.TestCase):
    def test_cloud_normalization(self):
        points = np.asarray(
            [[[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [1.0, 4.0, 3.0]]],
            dtype=np.float32,
        )
        normalized = normalize_ndf_clouds(points)
        np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-7)
        self.assertAlmostEqual(
            float((normalized.max(axis=1) - normalized.min(axis=1)).max()),
            1.0,
        )

    def test_derangement_has_no_fixed_points(self):
        permutation = derangement(17, 3)
        self.assertEqual(sorted(permutation.tolist()), list(range(17)))
        self.assertTrue(np.all(permutation != np.arange(17)))

    def test_axis_loss_is_signed(self):
        target = torch.tensor([[1.0, 0.0, 0.0]])
        correct = target[:, None, :].repeat(1, 5, 1)
        opposite = -correct
        self.assertAlmostEqual(float(functional_axis_loss(correct, target)), 0.0)
        self.assertAlmostEqual(float(functional_axis_loss(opposite, target)), 2.0)

    def test_task_axis_dataset_decodes_object_pose_layout(self):
        pose = np.asarray([[0, 0, 0, 1, 0, 0, 1, 0, 0]], dtype=np.float32)
        payload = {
            "points_a": np.zeros((1, 5, 3), dtype=np.float32),
            "current_object_pose9": pose,
            "shoe_id": np.asarray([4]),
            "episode_id": np.asarray([8]),
        }
        points, axis, shoe, episode = axis_dataset(payload, "current_object_z")
        np.testing.assert_array_equal(points, payload["points_a"])
        np.testing.assert_allclose(axis, [[0, 0, 1]], atol=1e-6)
        np.testing.assert_array_equal(shoe, [4])
        np.testing.assert_array_equal(episode, [8])


if __name__ == "__main__":
    unittest.main()
