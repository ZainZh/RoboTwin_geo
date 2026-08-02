from __future__ import annotations

import unittest

import numpy as np

from .merge_task_flow_recovery import merge_payloads


class MergeTaskFlowRecoveryTest(unittest.TestCase):
    def test_preserves_recovery_depth_and_seed(self):
        nominal = {
            "state": np.zeros((2, 1), dtype=np.float32),
            "episode_index": np.asarray([0, 0], dtype=np.int64),
            "episode_id": np.asarray([0, 0], dtype=np.int64),
            "episode_flow": np.zeros((1, 1, 2, 3), dtype=np.float32),
            "episode_pose": np.zeros((1, 2, 9), dtype=np.float32),
            "episode_shoe_id": np.asarray([2], dtype=np.int64),
        }
        recovery = {
            "state": np.zeros((3, 1), dtype=np.float32),
            "episode_index": np.asarray([0, 1, 1], dtype=np.int64),
            "episode_id": np.asarray([0, 1, 1], dtype=np.int64),
            "episode_flow": np.zeros((2, 1, 2, 3), dtype=np.float32),
            "episode_pose": np.zeros((2, 2, 9), dtype=np.float32),
            "episode_shoe_id": np.asarray([3, 4], dtype=np.int64),
        }
        manifest = {
            "records": [
                {"episode": 0, "policy_actions": 2, "seed": 11},
                {"episode": 1, "policy_actions": 4, "seed": 12},
            ]
        }
        predicted = np.zeros((2, 1, 2, 3), dtype=np.float32)
        merged = merge_payloads(nominal, [(recovery, predicted, manifest)])
        np.testing.assert_array_equal(
            merged["recovery_policy_actions"], [0, 0, 2, 4, 4]
        )
        np.testing.assert_array_equal(
            merged["recovery_source_seed"], [-1, -1, 11, 12, 12]
        )


if __name__ == "__main__":
    unittest.main()
