from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .build_grasp_relation_flow_prediction import transfer_rigid_flow_to_anchors


class GraspRelationFlowPredictionTest(unittest.TestCase):
    def test_rigid_schedule_is_transferred_without_point_identity_mismatch(self):
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        destination = np.asarray(
            [[-0.2, 0.5, 1.0], [0.1, -0.4, 0.7], [0.8, 0.2, -0.3]],
            dtype=np.float32,
        )
        transforms = (
            (Rotation.identity().as_matrix(), np.zeros(3)),
            (Rotation.from_euler("z", 35, degrees=True).as_matrix(), np.asarray([0.3, -0.2, 0.1])),
        )
        source_flow = np.stack(
            [source @ rotation.T + translation for rotation, translation in transforms],
            axis=1,
        )
        expected = np.stack(
            [destination @ rotation.T + translation for rotation, translation in transforms],
            axis=1,
        )
        actual = transfer_rigid_flow_to_anchors(source, source_flow, destination)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(actual[:, 0], destination, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
