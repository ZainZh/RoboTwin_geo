import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = REPO_ROOT / "policy" / "DP3"
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

import deploy_policy
from scripts import ndf_feature_utils


class FakeNDF(torch.nn.Module):
    latent_dim = 2

    def extract_latent(self, batch):
        return torch.zeros((1, 1), dtype=torch.float32, device=batch["point_cloud"].device)

    def forward_latent(self, z, query):
        return torch.ones((*query.shape[:2], self.latent_dim), dtype=torch.float32, device=query.device)


def make_point_cloud(points):
    rgb = np.zeros((len(points), 3), dtype=np.float32)
    return np.concatenate([np.asarray(points, dtype=np.float32), rgb], axis=1)


def make_model():
    return SimpleNamespace(
        use_object_pointcloud=True,
        use_actorseg_objpc=False,
        use_ndf_pointwise=True,
        use_ndf_pointwise_hybrid=True,
        use_ndf_pointwise_interact=False,
        use_ndf_pointwise_relation=True,
        use_semantic_pointwise=False,
        use_utonia_pointwise=False,
        object_placeholders=["{A}", "{B}"],
        ndf_models={"{A}": object(), "{B}": object()},
        semantic_models={},
        utonia_models={},
        target_num_points=4,
        ndf_feat_dim=256,
        ndf_point_num_by_placeholder={"{A}": 2, "{B}": 2},
        ndf_relation_point_num_by_pair={("{A}", "{B}"): 2, ("{B}", "{A}"): 2},
        ndf_device=torch.device("cpu"),
    )


def make_observation():
    return {
        "joint_action": {"vector": np.zeros((14,), dtype=np.float32)},
        "pointcloud": np.zeros((4, 6), dtype=np.float32),
        "object_pointcloud": {
            "{A}": make_point_cloud([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            "{B}": make_point_cloud([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        },
    }


class TestNDFRelationHybrid(unittest.TestCase):
    def test_relation_cloud_uses_query_coordinates_in_support_normalized_frame(self):
        support = make_point_cloud([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        query = make_point_cloud([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        relation = ndf_feature_utils.compute_ndf_relation_pointwise_cloud(
            model=FakeNDF(),
            support_object_point_cloud=support,
            query_object_point_cloud=query,
            device=torch.device("cpu"),
            target_num_points=2,
        )

        self.assertEqual(relation.shape, (2, 5))
        np.testing.assert_allclose(relation[:, :3], np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(relation[:, 3:], np.ones((2, 2), dtype=np.float32))

    def test_deploy_policy_adds_bidirectional_relation_branches(self):
        fake_ndf_cloud = np.ones((2, 259), dtype=np.float32)
        fake_relation_cloud = np.full((2, 259), 3.0, dtype=np.float32)
        with patch.object(deploy_policy, "compute_ndf_pointwise_cloud", return_value=fake_ndf_cloud), \
                patch.object(deploy_policy, "compute_ndf_relation_pointwise_cloud", return_value=fake_relation_cloud):
            obs = deploy_policy.encode_obs(make_observation(), make_model())

        self.assertEqual(obs["ndf_relation_point_cloud_A_in_B"].shape, (2, 259))
        self.assertEqual(obs["ndf_relation_point_cloud_B_in_A"].shape, (2, 259))
        self.assertNotIn("ndf_interact_point_cloud_A_from_B", obs)

    def test_relation_hydra_and_wrappers_exist(self):
        expected = [
            POLICY_ROOT / "train_ndf_relation_hybrid.sh",
            POLICY_ROOT / "eval_ndf_relation_hybrid.sh",
            POLICY_ROOT / "process_data_ndf_relation_hybrid.sh",
            POLICY_ROOT / "scripts" / "process_data_ndf_relation_hybrid.py",
            POLICY_ROOT / "3D-Diffusion-Policy" / "diffusion_policy_3d" / "config" / "robot_dp3_ndf_relation_hybrid.yaml",
            POLICY_ROOT / "3D-Diffusion-Policy" / "diffusion_policy_3d" / "config" / "task" / "demo_task_ndf_relation_hybrid.yaml",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"missing {path}")


if __name__ == "__main__":
    unittest.main()
