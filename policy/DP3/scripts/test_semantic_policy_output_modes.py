import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = REPO_ROOT / "policy" / "DP3"
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

import deploy_policy
import semantic_feature_utils


def make_point_cloud(x_value: float) -> np.ndarray:
    return np.asarray(
        [
            [x_value, 0.0, 0.5, 0.0, 0.0, 0.0],
            [x_value, 1.0, 0.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def make_model(output_mode: str) -> SimpleNamespace:
    feature_dim = {"embedding": 128, "xyz": 0, "part_prob": 2}[output_mode]
    return SimpleNamespace(
        use_object_pointcloud=True,
        use_actorseg_objpc=False,
        use_ndf_pointwise=False,
        use_ndf_pointwise_hybrid=False,
        use_semantic_pointwise=True,
        use_semantic_pointwise_hybrid=True,
        object_placeholders=["{A}", "{B}"],
        ndf_models={},
        semantic_models={"{A}": {"sem_embedding_dim": 128}},
        target_num_points=4,
        semantic_feat_dim=feature_dim,
        semantic_point_num_by_placeholder={"{A}": 2},
        semantic_feat_dim_by_placeholder={"{A}": feature_dim},
        semantic_policy_output_mode=output_mode,
        semantic_device=torch.device("cpu"),
    )


def make_observation() -> dict:
    return {
        "joint_action": {"vector": np.zeros((14,), dtype=np.float32)},
        "pointcloud": np.zeros((4, 6), dtype=np.float32),
        "object_pointcloud": {
            "{A}": make_point_cloud(1.0),
            "{B}": make_point_cloud(2.0),
        },
    }


class TestSemanticPolicyOutputModes(unittest.TestCase):
    def test_deploy_forwards_output_mode(self):
        for mode, channels in (("embedding", 131), ("xyz", 3), ("part_prob", 5)):
            captured = []

            def fake_semantic_cloud(**kwargs):
                captured.append(kwargs)
                return np.ones((2, channels), dtype=np.float32)

            with patch.object(
                deploy_policy,
                "get_semantic_utils",
                return_value=(fake_semantic_cloud, None),
            ):
                obs = deploy_policy.encode_obs(make_observation(), make_model(mode))

            self.assertEqual(captured[0]["output_mode"], mode)
            self.assertEqual(obs["semantic_point_cloud_A"].shape, (2, channels))

    def test_all_modes_preserve_the_same_queries(self):
        queries = np.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )
        semantic_output = {
            "embedding": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
            "logits": torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32),
        }
        artifacts = {
            "sem_embedding_dim": 4,
            "canonical_label_names": ["Handle", "Head"],
        }
        with patch.object(
            semantic_feature_utils,
            "_compute_semantic_query",
            return_value=(queries, semantic_output),
        ):
            outputs = {
                mode: semantic_feature_utils.compute_semantic_pointwise_cloud(
                    artifacts,
                    make_point_cloud(1.0),
                    target_num_points=2,
                    output_mode=mode,
                )
                for mode in ("embedding", "xyz", "part_prob")
            }

        self.assertEqual(outputs["embedding"].shape, (2, 7))
        self.assertEqual(outputs["xyz"].shape, (2, 3))
        self.assertEqual(outputs["part_prob"].shape, (2, 5))
        for output in outputs.values():
            self.assertTrue(np.array_equal(output[:, :3], queries))
        self.assertTrue(np.allclose(outputs["part_prob"][:, 3:].sum(axis=1), 1.0))

    def test_empty_cloud_uses_mode_specific_shape(self):
        artifacts = {
            "sem_embedding_dim": 4,
            "canonical_label_names": ["Handle", "Head"],
        }
        empty = np.zeros((0, 6), dtype=np.float32)
        expected_channels = {"embedding": 7, "xyz": 3, "part_prob": 5}
        for mode, channels in expected_channels.items():
            output = semantic_feature_utils.compute_semantic_pointwise_cloud(
                artifacts,
                empty,
                target_num_points=2,
                output_mode=mode,
            )
            self.assertEqual(output.shape, (2, channels))


if __name__ == "__main__":
    unittest.main()
