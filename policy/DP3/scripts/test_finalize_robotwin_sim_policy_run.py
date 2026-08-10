import unittest

from finalize_robotwin_sim_policy_run import build_expected_shape_meta


class TestFinalizeRobotwinSimPolicyRun(unittest.TestCase):
    def test_vanilla_shape_has_no_auxiliary_observation(self):
        shape_meta, obs_key = build_expected_shape_meta("none", 0)
        self.assertIsNone(obs_key)
        self.assertEqual({"point_cloud", "agent_pos"}, set(shape_meta["obs"]))
        self.assertEqual([14], shape_meta["action"]["shape"])

    def test_pointwise_shape_keeps_existing_contract(self):
        shape_meta, obs_key = build_expected_shape_meta(
            "utonia_point_cloud_A", 131
        )
        self.assertEqual("utonia_point_cloud_A", obs_key)
        self.assertEqual(
            [128, 131],
            shape_meta["obs"]["utonia_point_cloud_A"]["shape"],
        )

    def test_invalid_feature_dims_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "feature_dim=0"):
            build_expected_shape_meta("none", 3)
        with self.assertRaisesRegex(ValueError, "feature_dim >= 3"):
            build_expected_shape_meta("semantic_point_cloud_A", 2)


if __name__ == "__main__":
    unittest.main()
