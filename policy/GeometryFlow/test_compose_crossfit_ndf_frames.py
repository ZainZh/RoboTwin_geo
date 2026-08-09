import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .compose_crossfit_ndf_frames import (
    compose_crossfit_frames,
    parse_object_prediction_groups,
)


class ComposeCrossfitFramesTest(unittest.TestCase):
    def test_explicit_crossfold_groups_require_every_training_object(self):
        parsed = parse_object_prediction_groups(
            [["2", "fold2_seed0.npz", "fold2_seed1.npz"],
             ["4", "fold4_seed0.npz", "fold4_seed1.npz"]],
            (2, 4),
        )
        self.assertEqual(set(parsed), {2, 4})
        self.assertEqual(parsed[2][0].name, "fold2_seed0.npz")
        with self.assertRaisesRegex(ValueError, "missing=\\[4\\]"):
            parse_object_prediction_groups(
                [["2", "a.npz", "b.npz"]],
                (2, 4),
            )

    def test_training_rows_use_heldout_and_evaluation_rows_use_full(self):
        shoe_id = np.asarray((0, 1, 2), dtype=np.int64)

        def frames(degrees):
            rotations = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
            return np.stack((rotations, rotations), axis=0).astype(np.float32)

        full = frames((1.0, 2.0, 3.0))
        heldout0 = frames((11.0, 12.0, 13.0))
        heldout1 = frames((21.0, 22.0, 23.0))
        result, confidence, disagreement, threshold = compose_crossfit_frames(
            shoe_id,
            full,
            {0: heldout0, 1: heldout1},
            (0, 1),
            confidence_percentile=99.0,
        )
        angles = Rotation.from_matrix(result).as_euler("xyz", degrees=True)[:, 2]
        np.testing.assert_allclose(angles, (11.0, 22.0, 3.0), atol=1e-4)
        np.testing.assert_allclose(disagreement, 0.0, atol=1e-4)
        self.assertEqual(confidence.shape, (3,))
        self.assertGreaterEqual(threshold, 0.0)

    def test_requires_every_training_object(self):
        members = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
        with self.assertRaises(KeyError):
            compose_crossfit_frames(
                np.asarray((0, 1)),
                members,
                {0: members},
                (0, 1),
                confidence_percentile=95.0,
            )


if __name__ == "__main__":
    unittest.main()
