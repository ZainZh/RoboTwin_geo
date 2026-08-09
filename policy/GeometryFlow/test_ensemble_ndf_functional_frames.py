from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .ensemble_ndf_functional_frames import (
    maximum_pairwise_disagreement,
    project_rotation_mean,
    rotation_medoid,
    stabilize_rotation_against_previous,
    stabilize_rotation_sequences,
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

    def test_causal_stabilizer_corrects_and_masks_axis_sign_jump(self):
        previous = Rotation.from_euler("z", 2.0, degrees=True).as_matrix()
        flipped = Rotation.from_euler("z", 179.0, degrees=True).as_matrix()
        selected, ambiguous, corrected, jump = stabilize_rotation_against_previous(
            flipped,
            previous,
            jump_trigger_deg=120.0,
            alternative_accept_deg=45.0,
        )
        residual = Rotation.from_matrix(selected @ previous.T).magnitude()
        self.assertTrue(ambiguous)
        self.assertTrue(corrected)
        self.assertGreater(float(jump), 170.0)
        self.assertLess(float(np.rad2deg(residual)), 5.0)

    def test_sequence_stabilizer_resets_at_episode_boundary(self):
        rotations = Rotation.from_euler(
            "z", [0.0, 179.0, 179.0], degrees=True
        ).as_matrix()
        output, ambiguous, corrected = stabilize_rotation_sequences(
            rotations,
            episode_id=np.asarray([0, 0, 1]),
            frame_index=np.asarray([0, 1, 0]),
        )
        self.assertTrue(bool(ambiguous[1]))
        self.assertTrue(bool(corrected[1]))
        self.assertFalse(bool(ambiguous[2]))
        np.testing.assert_allclose(output[2], rotations[2], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
