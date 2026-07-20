import importlib
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
        expected_query = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float32,
            device=query.device,
        )
        torch.testing.assert_close(query, expected_query)
        return torch.ones((*query.shape[:2], self.latent_dim), dtype=torch.float32, device=query.device)


class ConstantNDF(torch.nn.Module):
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
        use_ndf_relation_token=False,
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
        ndf_relation_token_point_num_by_pair={("{A}", "{B}"): 2, ("{B}", "{A}"): 2},
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
    def test_relation_token_gates_out_of_domain_ndf_summary(self):
        support = make_point_cloud([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        near_query = make_point_cloud([[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
        far_query = make_point_cloud([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])

        near = ndf_feature_utils.compute_ndf_relation_token(
            model=ConstantNDF(),
            support_object_point_cloud=support,
            query_object_point_cloud=near_query,
            device=torch.device("cpu"),
            target_num_points=2,
        )
        far = ndf_feature_utils.compute_ndf_relation_token(
            model=ConstantNDF(),
            support_object_point_cloud=support,
            query_object_point_cloud=far_query,
            device=torch.device("cpu"),
            target_num_points=2,
        )

        self.assertEqual(near.shape, (11,))
        self.assertAlmostEqual(float(near[8]), 1.0, places=6)
        np.testing.assert_allclose(near[-2:], np.ones((2,), dtype=np.float32))
        self.assertAlmostEqual(float(far[8]), 0.0, places=6)
        np.testing.assert_allclose(far[-2:], np.zeros((2,), dtype=np.float32))
        repeated = ndf_feature_utils.compute_ndf_relation_token(
            model=ConstantNDF(),
            support_object_point_cloud=support,
            query_object_point_cloud=near_query,
            device=torch.device("cpu"),
            target_num_points=2,
        )
        np.testing.assert_allclose(repeated, near)

    def test_relation_token_v2_gates_all_geometry_and_projects_descriptor(self):
        support = make_point_cloud([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        near_query = make_point_cloud([[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
        far_query = make_point_cloud([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        kwargs = {
            "model": ConstantNDF(),
            "support_object_point_cloud": support,
            "device": torch.device("cpu"),
            "target_num_points": 2,
            "projection_dim": 1,
            "projection_seed": 7,
            "gate_geometry": True,
        }

        near = ndf_feature_utils.compute_ndf_relation_token(
            query_object_point_cloud=near_query,
            **kwargs,
        )
        far = ndf_feature_utils.compute_ndf_relation_token(
            query_object_point_cloud=far_query,
            **kwargs,
        )

        self.assertEqual(near.shape, (10,))
        self.assertAlmostEqual(float(near[8]), 1.0, places=6)
        self.assertNotEqual(float(near[9]), 0.0)
        np.testing.assert_allclose(far, np.zeros((10,), dtype=np.float32))

    def test_deploy_policy_adds_relation_token_without_dense_relation_cloud(self):
        model = make_model()
        model.use_ndf_pointwise_relation = False
        model.use_ndf_relation_token = True
        model.ndf_models = {"{A}": ConstantNDF()}
        fake_ndf_cloud = np.ones((2, 5), dtype=np.float32)
        fake_token = np.arange(11, dtype=np.float32)

        with patch.object(deploy_policy, "compute_ndf_pointwise_cloud", return_value=fake_ndf_cloud), \
                patch.object(deploy_policy, "compute_ndf_relation_token", return_value=fake_token):
            obs = deploy_policy.encode_obs(make_observation(), model)

        np.testing.assert_allclose(obs["ndf_relation_token_B_in_A"], fake_token)
        self.assertNotIn("ndf_relation_point_cloud_B_in_A", obs)

    def test_relation_cloud_preserves_world_xyz_and_queries_support_normalized_frame(self):
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
        np.testing.assert_allclose(relation[:, :3], np.asarray([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32))
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

    def test_deploy_relation_matches_offline_relation_feature(self):
        model = make_model()
        model.ndf_models = {"{A}": FakeNDF()}
        observation = make_observation()
        expected = ndf_feature_utils.compute_ndf_relation_pointwise_cloud(
            model=model.ndf_models["{A}"],
            support_object_point_cloud=observation["object_pointcloud"]["{A}"],
            query_object_point_cloud=observation["object_pointcloud"]["{B}"],
            device=torch.device("cpu"),
            target_num_points=2,
        )

        fake_ndf_cloud = np.ones((2, 5), dtype=np.float32)
        with patch.object(deploy_policy, "compute_ndf_pointwise_cloud", return_value=fake_ndf_cloud):
            encoded = deploy_policy.encode_obs(observation, model)

        np.testing.assert_allclose(encoded["ndf_relation_point_cloud_B_in_A"], expected)

    def test_relation_v2_metadata_rejects_stale_or_wrong_coordinate_schema(self):
        schema = importlib.import_module("scripts.ndf_relation_schema")
        metadata = schema.relation_v2_metadata()

        self.assertEqual(metadata["relation_schema_version"], 2)
        self.assertEqual(metadata["relation_xyz_frame"], "world")
        self.assertEqual(metadata["relation_query_frame"], "support_normalized")
        schema.validate_relation_v2_metadata(metadata)

        for key, value in (
            ("relation_schema_version", 1),
            ("relation_xyz_frame", "support_normalized"),
            ("relation_query_frame", "world"),
        ):
            invalid = dict(metadata)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                schema.validate_relation_v2_metadata(invalid)

    def test_relation_v2_wrappers_are_isolated_and_configurable(self):
        expected = [
            POLICY_ROOT / "train_ndf_relation_v2.sh",
            POLICY_ROOT / "eval_ndf_relation_v2.sh",
            POLICY_ROOT / "process_data_ndf_relation_v2.sh",
            POLICY_ROOT / "scripts" / "process_data_ndf_relation_v2.py",
            POLICY_ROOT / "scripts" / "validate_ndf_relation_v2_meta.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"missing {path}")

        train_text = expected[0].read_text(encoding="utf-8")
        eval_text = expected[1].read_text(encoding="utf-8")
        process_text = expected[2].read_text(encoding="utf-8")
        self.assertIn("-objpc-ndf-relation-v2", train_text)
        self.assertIn("validate_ndf_relation_v2_meta.py", train_text)
        self.assertIn("training.num_epochs=${num_epochs}", train_text)
        self.assertIn("--test_num ${test_num}", eval_text)
        self.assertIn("-objpc-ndf-relation-v2", process_text)

    def test_eval_policy_exposes_test_num_override(self):
        deploy_config = (POLICY_ROOT / "deploy_policy.yml").read_text(encoding="utf-8")
        eval_source = (REPO_ROOT / "script" / "eval_policy.py").read_text(encoding="utf-8")
        self.assertIn("test_num: 100", deploy_config)
        self.assertIn('test_num = int(usr_args.get("test_num", 100))', eval_source)

    def test_relation_v2_wrappers_do_not_append_escaped_placeholder_fragments(self):
        wrappers = [
            POLICY_ROOT / "process_data_ndf_relation_v2.sh",
            POLICY_ROOT / "train_ndf_relation_v2.sh",
            POLICY_ROOT / "eval_ndf_relation_v2.sh",
        ]
        for path in wrappers:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn('DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"', text)
                self.assertIn('${DEFAULT_OBJECT_PLACEHOLDERS}', text)
                self.assertNotIn(r"\\{A\\}", text)

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

    def test_relation_token_wrappers_are_isolated_and_configurable(self):
        expected = [
            POLICY_ROOT / "process_data_ndf_relation_token_v1.sh",
            POLICY_ROOT / "train_ndf_relation_token_v1.sh",
            POLICY_ROOT / "eval_ndf_relation_token_v1.sh",
            POLICY_ROOT / "scripts" / "process_data_ndf_relation_token_v1.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"missing {path}")

        train_text = expected[1].read_text(encoding="utf-8")
        eval_text = expected[2].read_text(encoding="utf-8")
        self.assertIn("-objpc-ndf-relation-token-v1", train_text)
        self.assertIn("training.num_epochs=${num_epochs}", train_text)
        self.assertIn("--use_ndf_relation_token true", eval_text)
        self.assertIn("--test_num ${test_num}", eval_text)

    def test_relation_token_v2_wrappers_are_isolated(self):
        expected = [
            POLICY_ROOT / "process_data_ndf_relation_token_v2.sh",
            POLICY_ROOT / "train_ndf_relation_token_v2.sh",
            POLICY_ROOT / "eval_ndf_relation_token_v2.sh",
            POLICY_ROOT / "scripts" / "process_data_ndf_relation_token_v2.py",
            POLICY_ROOT / "scripts" / "validate_ndf_relation_token_v2_meta.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"missing {path}")


if __name__ == "__main__":
    unittest.main()
