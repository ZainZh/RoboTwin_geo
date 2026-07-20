import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = REPO_ROOT / "policy" / "DP3"
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

import deploy_policy  # noqa: E402
from se3_relation_token_utils import RELATION_TOKEN_KEY  # noqa: E402


def make_cloud(x):
    return np.asarray([[x, 0.0, 0.7, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)


def make_observation():
    pose = np.asarray([0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return {
        "joint_action": {"vector": np.zeros((14,), dtype=np.float32)},
        "pointcloud": np.zeros((8, 6), dtype=np.float32),
        "object_pointcloud": {"{A}": make_cloud(0.1), "{B}": make_cloud(-0.1)},
        "task_state": {
            "object_pose_A": pose,
            "object_pose_B": pose,
            "goal_T_A_from_B_oracle": np.eye(4, dtype=np.float32).reshape(-1),
            "shoe_id": np.asarray([0], dtype=np.int64),
            "relation_phase": np.asarray([1.0], dtype=np.float32),
        },
    }


def make_model():
    return SimpleNamespace(
        use_object_pointcloud=True,
        use_actorseg_objpc=False,
        use_sam2_objpc=False,
        use_ndf_pointwise=False,
        use_semantic_pointwise=False,
        use_utonia_pointwise=False,
        use_se3_relation_token=True,
        se3_relation_route="oracle",
        se3_relation_goal_table=None,
        object_placeholders=["{A}", "{B}"],
        target_num_points=8,
    )


class TestSe3RelationDeploy(unittest.TestCase):
    def test_online_encoder_adds_same_low_dim_relation_token(self):
        encoded = deploy_policy.encode_obs(make_observation(), make_model())
        self.assertIn(RELATION_TOKEN_KEY, encoded)
        self.assertEqual(encoded[RELATION_TOKEN_KEY].shape, (11,))
        self.assertEqual(float(encoded[RELATION_TOKEN_KEY][-1]), 1.0)

    def test_online_encoder_rejects_observation_without_task_state(self):
        observation = make_observation()
        observation.pop("task_state")
        with self.assertRaisesRegex(RuntimeError, "task_state"):
            deploy_policy.encode_obs(observation, make_model())


if __name__ == "__main__":
    unittest.main()
