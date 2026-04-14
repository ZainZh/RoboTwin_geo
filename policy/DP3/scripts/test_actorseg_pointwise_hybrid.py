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


def make_point_cloud(x_value: float) -> np.ndarray:
    return np.asarray(
        [
            [x_value, 0.0, 0.5, 0.0, 0.0, 0.0],
            [x_value, 1.0, 0.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def make_observation() -> dict:
    return {
        "joint_action": {
            "vector": np.zeros((14,), dtype=np.float32),
        },
        "pointcloud": np.zeros((4, 6), dtype=np.float32),
        "observation": {},
        "object_pointcloud": {},
    }


def fake_actorseg_extract_fn(observation, *, placeholder, **kwargs):
    x_value = 1.0 if placeholder == "{A}" else 2.0
    return make_point_cloud(x_value), {"mode": "actorseg_projected"}


class TestActorSegPointwiseHybrid(unittest.TestCase):
    def test_actorseg_ndf_pointwise_hybrid_keeps_main_raw_cloud_and_ndf_branch(self):
        model = SimpleNamespace(
            use_object_pointcloud=True,
            use_actorseg_objpc=True,
            use_sam3_objpc=False,
            use_ndf_pointwise=True,
            use_ndf_pointwise_hybrid=True,
            use_semantic_pointwise=False,
            use_semantic_pointwise_hybrid=False,
            object_placeholders=["{A}", "{B}"],
            actorseg_extract_fn=fake_actorseg_extract_fn,
            actorseg_camera_names=["head_camera", "front_camera"],
            actorseg_segmentation_key="actor_segmentation",
            actorseg_actor_ids_by_placeholder={"{A}": [1], "{B}": [2]},
            target_num_points=4,
            ndf_models={"{A}": object()},
            semantic_models={},
            ndf_feat_dim=256,
            ndf_point_num_by_placeholder={"{A}": 2},
            ndf_device=torch.device("cpu"),
        )

        fake_ndf_cloud = np.ones((2, 259), dtype=np.float32)
        with patch.object(deploy_policy, "compute_ndf_pointwise_cloud", return_value=fake_ndf_cloud):
            obs = deploy_policy.encode_obs(make_observation(), model)

        self.assertEqual(obs["point_cloud"].shape, (4, 6))
        self.assertEqual(sorted(np.unique(obs["point_cloud"][:, 0]).tolist()), [1.0, 2.0])
        self.assertEqual(obs["ndf_point_cloud_A"].shape, (2, 259))
        self.assertTrue(np.array_equal(obs["ndf_point_cloud_A"], fake_ndf_cloud))

    def test_actorseg_semantic_pointwise_hybrid_keeps_main_raw_cloud_and_semantic_branch(self):
        model = SimpleNamespace(
            use_object_pointcloud=True,
            use_actorseg_objpc=True,
            use_sam3_objpc=False,
            use_ndf_pointwise=False,
            use_ndf_pointwise_hybrid=False,
            use_semantic_pointwise=True,
            use_semantic_pointwise_hybrid=True,
            object_placeholders=["{A}", "{B}"],
            actorseg_extract_fn=fake_actorseg_extract_fn,
            actorseg_camera_names=["head_camera", "front_camera"],
            actorseg_segmentation_key="actor_segmentation",
            actorseg_actor_ids_by_placeholder={"{A}": [1], "{B}": [2]},
            target_num_points=4,
            ndf_models={},
            semantic_models={"{A}": {"sem_embedding_dim": 128}},
            semantic_point_num_by_placeholder={"{A}": 2},
            semantic_feat_dim_by_placeholder={"{A}": 128},
            semantic_feat_dim=128,
            semantic_device=torch.device("cpu"),
        )

        fake_semantic_cloud = np.ones((2, 131), dtype=np.float32)
        with patch.object(
            deploy_policy,
            "get_semantic_utils",
            return_value=(lambda **kwargs: fake_semantic_cloud, None),
        ):
            obs = deploy_policy.encode_obs(make_observation(), model)

        self.assertEqual(obs["point_cloud"].shape, (4, 6))
        self.assertEqual(sorted(np.unique(obs["point_cloud"][:, 0]).tolist()), [1.0, 2.0])
        self.assertEqual(obs["semantic_point_cloud_A"].shape, (2, 131))
        self.assertTrue(np.array_equal(obs["semantic_point_cloud_A"], fake_semantic_cloud))


if __name__ == "__main__":
    unittest.main()
