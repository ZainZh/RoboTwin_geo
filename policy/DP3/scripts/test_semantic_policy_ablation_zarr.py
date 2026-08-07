from __future__ import annotations

import unittest

import numpy as np

from build_semantic_policy_ablation_zarr import (
    stable_softmax,
    transform_semantic_cloud,
)


class SemanticPolicyAblationZarrTest(unittest.TestCase):
    def test_stable_softmax(self) -> None:
        values = np.asarray([[1000.0, 1001.0], [-1000.0, -1000.0]], dtype=np.float32)
        probabilities = stable_softmax(values)
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
        self.assertGreater(float(probabilities[0, 1]), float(probabilities[0, 0]))

    def test_xyz_route_is_bitwise_identical(self) -> None:
        cloud = np.arange(2 * 4 * 7, dtype=np.float32).reshape(2, 4, 7)
        output = transform_semantic_cloud(cloud, mode="xyz")
        self.assertEqual(output.shape, (2, 4, 3))
        self.assertTrue(np.array_equal(output, cloud[..., :3]))

    def test_part_probability_route(self) -> None:
        cloud = np.asarray(
            [[[1.0, 2.0, 3.0, 1.0, 0.0], [4.0, 5.0, 6.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
        weight = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        bias = np.asarray([0.0, 0.0], dtype=np.float32)
        output = transform_semantic_cloud(
            cloud,
            mode="part_prob",
            logit_weight=weight,
            logit_bias=bias,
        )
        self.assertEqual(output.shape, (1, 2, 5))
        self.assertTrue(np.array_equal(output[..., :3], cloud[..., :3]))
        np.testing.assert_allclose(output[..., 3:].sum(axis=-1), 1.0, atol=1e-6)
        self.assertGreater(float(output[0, 0, 3]), float(output[0, 0, 4]))
        self.assertGreater(float(output[0, 1, 4]), float(output[0, 1, 3]))

    def test_part_probability_dimension_mismatch(self) -> None:
        cloud = np.zeros((2, 4, 7), dtype=np.float32)
        with self.assertRaises(ValueError):
            transform_semantic_cloud(
                cloud,
                mode="part_prob",
                logit_weight=np.zeros((2, 3), dtype=np.float32),
                logit_bias=np.zeros(2, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
