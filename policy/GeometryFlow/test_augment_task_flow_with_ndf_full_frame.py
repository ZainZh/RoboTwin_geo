from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .augment_task_flow_with_ndf_full_frame import augment_payload
from .functional_action_frame import frame9_rotation


class NdfFullFrameAugmentTest(unittest.TestCase):
    def test_predicted_current_frame_yields_remaining_relative_rotation(self):
        true_current = Rotation.from_euler("xyz", [10.0, -5.0, 30.0], degrees=True).as_matrix()
        prediction_error = Rotation.from_euler("z", 7.0, degrees=True).as_matrix()
        predicted_current = prediction_error @ true_current
        intended = Rotation.from_euler("xyz", [3.0, 4.0, -12.0], degrees=True)
        pose9 = np.concatenate(
            ([0.1, 0.2, 0.3], true_current[:, :2].reshape(-1))
        )[None].astype(np.float32)
        payload = {
            "current_object_pose9": pose9,
            "goal_translation_error_xyz_m": np.asarray([[0.01, -0.02, 0.03]], dtype=np.float32),
            "goal_rotation_error_rotvec": intended.as_rotvec()[None].astype(np.float32),
            "shoe_id": np.asarray([0], dtype=np.int64),
        }
        output, error = augment_payload(payload, predicted_current[None])
        actual = frame9_rotation(output["target_frame9"])[0]
        expected = intended.as_matrix() @ true_current @ predicted_current.T
        self.assertTrue(np.allclose(actual, expected, atol=1e-5))
        self.assertAlmostEqual(float(error[0]), 7.0, places=4)
        self.assertTrue(
            np.allclose(
                output["target_frame9"][0, :3],
                payload["goal_translation_error_xyz_m"][0],
            )
        )


if __name__ == "__main__":
    unittest.main()
