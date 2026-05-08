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
import process_data_utonia_pointwise
import process_data_utonia_pointwise_hybrid


def make_point_cloud(x_value: float) -> np.ndarray:
    return np.asarray(
        [
            [x_value, 0.0, 0.5, 0.0, 0.0, 0.0],
            [x_value, 1.0, 0.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def make_model(*, hybrid: bool) -> SimpleNamespace:
    return SimpleNamespace(
        use_object_pointcloud=True,
        use_actorseg_objpc=False,
        use_ndf_pointwise=False,
        use_ndf_pointwise_hybrid=False,
        use_ndf_pointwise_interact=False,
        use_semantic_pointwise=False,
        use_semantic_pointwise_hybrid=False,
        use_utonia_pointwise=True,
        use_utonia_pointwise_hybrid=hybrid,
        object_placeholders=["{A}", "{B}"],
        ndf_models={},
        semantic_models={},
        utonia_models={"{A}": {"feature_dim": 96}},
        target_num_points=4,
        utonia_feat_dim=96,
        utonia_point_num_by_placeholder={"{A}": 2},
        utonia_feat_dim_by_placeholder={"{A}": 96},
        utonia_device=torch.device("cpu"),
    )


def make_observation() -> dict:
    return {
        "joint_action": {
            "vector": np.zeros((14,), dtype=np.float32),
        },
        "pointcloud": np.zeros((4, 6), dtype=np.float32),
        "object_pointcloud": {
            "{A}": make_point_cloud(1.0),
            "{B}": make_point_cloud(2.0),
        },
    }


class TestUtoniaPointwiseHybrid(unittest.TestCase):
    def setUp(self):
        self.fake_utonia_cloud = np.ones((2, 99), dtype=np.float32)

    def test_feature_placeholder_selection_is_explicit(self):
        self.assertEqual(
            process_data_utonia_pointwise.resolve_feature_placeholders(
                ["{A}", "{B}", "{C}"],
                "{A},{B}",
            ),
            ["{A}", "{B}"],
        )
        self.assertEqual(
            process_data_utonia_pointwise.resolve_feature_placeholders(
                ["{A}", "{B}", "{C}"],
                "{B}",
            ),
            ["{B}"],
        )

    def test_utonia_pointwise_excludes_feature_placeholder_from_main_context(self):
        with patch.object(
            deploy_policy,
            "get_utonia_utils",
            return_value=(lambda **kwargs: self.fake_utonia_cloud, None),
        ):
            obs = deploy_policy.encode_obs(make_observation(), make_model(hybrid=False))

        self.assertEqual(obs["point_cloud"].shape, (4, 6))
        self.assertTrue(np.allclose(obs["point_cloud"][:, 0], 2.0))
        self.assertEqual(obs["utonia_point_cloud_A"].shape, (2, 99))

    def test_utonia_pointwise_hybrid_keeps_feature_placeholder_in_main_context(self):
        with patch.object(
            deploy_policy,
            "get_utonia_utils",
            return_value=(lambda **kwargs: self.fake_utonia_cloud, None),
        ):
            obs = deploy_policy.encode_obs(make_observation(), make_model(hybrid=True))

        self.assertEqual(obs["point_cloud"].shape, (4, 6))
        self.assertEqual(sorted(np.unique(obs["point_cloud"][:, 0]).tolist()), [1.0, 2.0])
        self.assertEqual(obs["utonia_point_cloud_A"].shape, (2, 99))

    def test_hybrid_wrapper_builds_attached_output_suffix_argument(self):
        argv = [
            "hanging_mug",
            "demo_clean_3d_object_pc",
            "50",
            "--object_placeholders",
            "{A},{B}",
            "--utonia_feature_placeholders",
            "{A}",
        ]
        forwarded = process_data_utonia_pointwise_hybrid.build_hybrid_argv(argv)
        parser = process_data_utonia_pointwise.build_parser()
        args = parser.parse_args(forwarded)

        self.assertIn("--output_suffix=-objpc-utonia-pointwise-hybrid", forwarded)
        self.assertEqual(args.output_suffix, "-objpc-utonia-pointwise-hybrid")
        self.assertTrue(args.keep_feature_placeholders_in_context)
        self.assertEqual(args.utonia_feature_placeholders, "{A}")


if __name__ == "__main__":
    unittest.main()
