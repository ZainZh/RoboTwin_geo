from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .ensemble_ndf_functional_frames import (
    maximum_pairwise_disagreement,
    project_rotation_mean,
    rotation_medoid,
)


class NdfFunctionalFrameEnsembleTest(unittest.TestCase):
    def test_projected_mean_is_proper_and_between_predictions(self):
        angles = np.asarray([-4.0, 0.0, 5.0])
        frames = Rotation.from_euler("z", angles, degrees=True).as_matrix()[:, None]
        mean = project_rotation_mean(frames)[0]
        angle = Rotation.from_matrix(mean).as_euler("xyz", degrees=True)[2]
        self.assertGreater(angle, -1.0)
        self.assertLess(angle, 1.0)
        self.assertAlmostEqual(float(np.linalg.det(mean)), 1.0, places=5)

    def test_pairwise_disagreement_exposes_outlier(self):
        frames = Rotation.from_euler("z", [0.0, 2.0, 100.0], degrees=True).as_matrix()
        disagreement = maximum_pairwise_disagreement(frames[:, None])
        self.assertAlmostEqual(float(disagreement[0]), 100.0, places=4)

    def test_medoid_rejects_one_flipped_member(self):
        frames = Rotation.from_euler("z", [0.0, 2.0, 100.0], degrees=True).as_matrix()
        medoid = rotation_medoid(frames[:, None])[0]
        angle = Rotation.from_matrix(medoid).as_euler("xyz", degrees=True)[2]
        self.assertAlmostEqual(float(angle), 2.0, places=4)
        self.assertAlmostEqual(float(np.linalg.det(medoid)), 1.0, places=5)

    def test_medoid_validates_shape_and_member_count(self):
        with self.assertRaisesRegex(ValueError, "models,samples"):
            rotation_medoid(np.zeros((3, 3, 3)))
        with self.assertRaisesRegex(ValueError, "at least two"):
            rotation_medoid(np.eye(3)[None, None])


if __name__ == "__main__":
    unittest.main()
