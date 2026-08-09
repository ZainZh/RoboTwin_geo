import unittest

import numpy as np

from .augment_handled_mug_ndf_relation import build_camera_ndf_relation


class TestHandledMugNdfRelation(unittest.TestCase):
    def test_identity_current_and_goal_produce_translation_only_relation(self):
        identity_frame9 = np.asarray(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        identity_pose9 = np.asarray(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        payload = {
            "shoe_id": np.asarray([0]),
            "points_a": np.asarray(
                [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]], dtype=np.float32
            ),
            "current_object_pose9": identity_pose9,
            "goal_frame9": identity_frame9,
            "goal_translation_error_xyz_m": np.zeros((1, 3), dtype=np.float32),
            "goal_rotation_error_rotvec": np.zeros((1, 3), dtype=np.float32),
        }
        payload["goal_frame9"][0, :3] = [0.2, -0.1, 0.3]
        payload["goal_translation_error_xyz_m"][0] = [0.2, -0.1, 0.3]
        output, diagnostics = build_camera_ndf_relation(
            payload, np.eye(3, dtype=np.float32)[None], np.asarray([True])
        )
        np.testing.assert_allclose(output["target_frame9"][0, :3], [0.2, -0.1, 0.3])
        np.testing.assert_allclose(
            output["target_frame9"][0, 3:], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            diagnostics["relative_translation_error"], np.zeros((1, 3)), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
