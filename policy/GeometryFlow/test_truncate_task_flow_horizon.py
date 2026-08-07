from __future__ import annotations

import unittest

import numpy as np

from .truncate_task_flow_horizon import truncate_payload


class TruncateTaskFlowHorizonTest(unittest.TestCase):
    def test_truncates_action_and_future_pose_without_touching_tokens(self):
        payload = {
            "action": np.arange(2 * 6 * 14).reshape(2, 6, 14),
            "future_eef_pose7": np.arange(2 * 6 * 2 * 7).reshape(2, 6, 2, 7),
            "target_frame9": np.arange(18).reshape(2, 9),
        }
        result = truncate_payload(payload, 1)
        self.assertEqual(result["action"].shape, (2, 1, 14))
        self.assertEqual(result["future_eef_pose7"].shape, (2, 1, 2, 7))
        np.testing.assert_array_equal(result["target_frame9"], payload["target_frame9"])
        self.assertEqual(payload["action"].shape, (2, 6, 14))

    def test_rejects_horizon_larger_than_source(self):
        with self.assertRaises(ValueError):
            truncate_payload({"action": np.zeros((2, 3, 14))}, 4)


if __name__ == "__main__":
    unittest.main()
