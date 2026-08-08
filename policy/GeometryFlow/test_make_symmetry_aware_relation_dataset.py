from __future__ import annotations

import unittest

import numpy as np

from scipy.spatial.transform import Rotation

from .make_symmetry_aware_relation_dataset import symmetry_quotient_frame9


class SymmetryAwareRelationDatasetTest(unittest.TestCase):
    def test_translation_is_retained_and_twist_is_removed(self):
        translation = np.asarray([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
        pose9 = np.asarray([[0, 0, 0, 1, 0, 0, 0, 1, 0]] * 2, dtype=float)
        rotvec = np.stack(
            (
                Rotation.from_euler("y", 20, degrees=True).as_rotvec(),
                Rotation.from_euler("x", 15, degrees=True).as_rotvec(),
            )
        )
        frame = symmetry_quotient_frame9(
            translation, rotvec, pose9, np.asarray([0, 1, 0])
        )
        np.testing.assert_allclose(frame[:, :3], translation)
        np.testing.assert_allclose(frame[0, 3:], [1, 0, 0, 0, 1, 0], atol=1e-6)
        self.assertGreater(float(np.linalg.norm(frame[1, 3:] - frame[0, 3:])), 0.01)


if __name__ == "__main__":
    unittest.main()
