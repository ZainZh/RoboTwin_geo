import unittest

import numpy as np

from semantic_observation_perturbations import (
    replace_with_aabb_shell_outliers,
    select_normal_facing_view,
)


class SemanticObservationPerturbationsTest(unittest.TestCase):
    def setUp(self):
        grid = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
        self.points = np.stack([grid, np.zeros_like(grid), np.zeros_like(grid)], axis=1)
        self.normals = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (10, 1))
        self.colors = np.tile(np.asarray([[20.0, 40.0, 60.0]], dtype=np.float32), (10, 1))

    def test_outlier_replacement_preserves_count_and_changes_exact_fraction(self):
        points, normals, colors = replace_with_aabb_shell_outliers(
            self.points,
            self.normals,
            self.colors,
            ratio=0.2,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(self.points.shape, points.shape)
        self.assertEqual(self.normals.shape, normals.shape)
        self.assertEqual(self.colors.shape, colors.shape)
        changed = np.any(points != self.points, axis=1)
        self.assertEqual(2, int(changed.sum()))
        lower = self.points.min(axis=0)
        upper = self.points.max(axis=0)
        self.assertTrue(np.all(np.any((points[changed] < lower) | (points[changed] > upper), axis=1)))
        np.testing.assert_allclose(np.linalg.norm(normals[changed], axis=1), 1.0, atol=1e-6)

    def test_outlier_replacement_is_deterministic(self):
        first = replace_with_aabb_shell_outliers(
            self.points, self.normals, self.colors, 0.3, np.random.default_rng(11)
        )
        second = replace_with_aabb_shell_outliers(
            self.points, self.normals, self.colors, 0.3, np.random.default_rng(11)
        )
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_normal_facing_view_keeps_requested_count(self):
        normals = np.stack(
            [
                np.cos(np.linspace(0.0, np.pi, 10)),
                np.sin(np.linspace(0.0, np.pi, 10)),
                np.zeros(10),
            ],
            axis=1,
        ).astype(np.float32)
        points, kept_normals, colors = select_normal_facing_view(
            self.points,
            normals,
            self.colors,
            keep_ratio=0.5,
            rng=np.random.default_rng(13),
        )
        self.assertEqual((8, 3), points.shape)
        self.assertEqual(points.shape, kept_normals.shape)
        self.assertEqual(points.shape, colors.shape)

    def test_invalid_ratios_fail(self):
        with self.assertRaises(ValueError):
            replace_with_aabb_shell_outliers(
                self.points, self.normals, self.colors, 1.0, np.random.default_rng(1)
            )
        with self.assertRaises(ValueError):
            select_normal_facing_view(
                self.points, self.normals, self.colors, 0.0, np.random.default_rng(1)
            )


if __name__ == "__main__":
    unittest.main()
