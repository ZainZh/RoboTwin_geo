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
    feature_dim = {
        "embedding": 128,
        "xyz": 0,
        "part_prob": 2,
        "uniform_prob": 2,
        "matched_utonia128": 128,
        "shuffled_prob": 2,
        "matched_utonia": 576,
    }[output_mode]
    return SimpleNamespace(
        use_object_pointcloud=True,
        use_actorseg_objpc=False,
        use_ndf_pointwise=False,
        use_ndf_pointwise_hybrid=False,
        use_semantic_pointwise=True,
        use_semantic_pointwise_hybrid=True,
        use_utonia_pointwise=False,
        use_utonia_pointwise_hybrid=False,
        object_placeholders=["{A}", "{B}"],
        ndf_models={},
        semantic_models={"{A}": {"sem_embedding_dim": 128}},
        utonia_models=(
            {"{A}": {"feature_dim": 576}}
            if output_mode in {"matched_utonia", "matched_utonia128"}
            else {}
        ),
        target_num_points=4,
        semantic_feat_dim=feature_dim,
        semantic_point_num_by_placeholder={"{A}": 2},
        semantic_feat_dim_by_placeholder={"{A}": feature_dim},
        utonia_feat_dim=576,
        utonia_feat_dim_by_placeholder={"{A}": feature_dim},
        semantic_policy_output_mode=output_mode,
        semantic_device=torch.device("cpu"),
        semantic_policy_output_seed=1234,
        semantic_policy_frame_index=0,
        matched_utonia_color_mode="debug_placeholder",
        matched_utonia_normal_mode="fallback",
        matched_utonia_projection=(
            {"matrix": np.eye(576, 128, dtype=np.float32)}
            if output_mode == "matched_utonia128"
            else None
        ),
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
        for mode, channels in (("embedding", 131), ("xyz", 3), ("part_prob", 5), ("uniform_prob", 5), ("shuffled_prob", 5)):
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

    def test_matched_utonia_preserves_semantic_queries_and_uses_utonia_key(self):
        queries = np.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )
        captured = {}

        def fake_semantic_cloud(**kwargs):
            captured["semantic"] = kwargs
            return queries.copy()

        def fake_matched_utonia(**kwargs):
            captured["utonia"] = kwargs
            return np.concatenate(
                [
                    kwargs["query_world_xyz"],
                    np.ones((len(kwargs["query_world_xyz"]), 576), dtype=np.float32),
                ],
                axis=1,
            )

        with (
            patch.object(
                deploy_policy,
                "get_semantic_utils",
                return_value=(fake_semantic_cloud, None),
            ),
            patch.object(
                deploy_policy,
                "get_matched_utonia_utils",
                return_value=fake_matched_utonia,
            ),
        ):
            observation = make_observation()
            obs = deploy_policy.encode_obs(
                observation, make_model("matched_utonia")
            )

        self.assertEqual("xyz", captured["semantic"]["output_mode"])
        self.assertTrue(
            np.array_equal(captured["utonia"]["query_world_xyz"], queries)
        )
        self.assertIs(
            captured["utonia"]["object_point_cloud"],
            observation["object_pointcloud"]["{A}"],
        )
        self.assertNotIn("semantic_point_cloud_A", obs)
        self.assertEqual((2, 579), obs["utonia_point_cloud_A"].shape)
        self.assertTrue(np.array_equal(obs["utonia_point_cloud_A"][:, :3], queries))

    def test_matched_utonia_rejects_missing_object_support(self):
        observation = make_observation()
        del observation["object_pointcloud"]["{A}"]
        with (
            patch.object(deploy_policy, "get_semantic_utils", return_value=(None, None)),
            patch.object(deploy_policy, "get_matched_utonia_utils", return_value=None),
            self.assertRaisesRegex(ValueError, "missing object support"),
        ):
            deploy_policy.encode_obs(observation, make_model("matched_utonia"))

    def test_matched_utonia128_projects_features_and_preserves_queries(self):
        queries = np.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )

        def fake_semantic_cloud(**kwargs):
            return queries.copy()

        def fake_matched_utonia(**kwargs):
            features = np.arange(2 * 576, dtype=np.float32).reshape(2, 576)
            return np.concatenate([kwargs["query_world_xyz"], features], axis=1)

        def fake_project(cloud, matrix):
            self.assertEqual((576, 128), matrix.shape)
            return np.concatenate([cloud[:, :3], cloud[:, 3:131]], axis=1)

        with (
            patch.object(
                deploy_policy,
                "get_semantic_utils",
                return_value=(fake_semantic_cloud, None),
            ),
            patch.object(
                deploy_policy,
                "get_matched_utonia_utils",
                return_value=fake_matched_utonia,
            ),
            patch.object(
                deploy_policy,
                "get_utonia_projection_utils",
                return_value=(None, fake_project),
            ),
        ):
            obs = deploy_policy.encode_obs(
                make_observation(), make_model("matched_utonia128")
            )

        self.assertEqual((2, 131), obs["utonia_point_cloud_A"].shape)
        self.assertTrue(np.array_equal(obs["utonia_point_cloud_A"][:, :3], queries))

    def test_matched_utonia_rejects_invalid_online_output(self):
        queries = np.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )

        def fake_semantic_cloud(**kwargs):
            return queries.copy()

        valid = np.concatenate(
            [queries, np.ones((2, 576), dtype=np.float32)], axis=1
        )
        nonfinite = valid.copy()
        nonfinite[0, 10] = np.nan
        cases = {
            "produced shape": valid[:, :-1],
            "produced NaN/Inf": nonfinite,
            "changed semantic query XYZ": np.concatenate(
                [queries + 1.0, valid[:, 3:]], axis=1
            ),
        }
        for message, output in cases.items():
            with self.subTest(message=message):
                with (
                    patch.object(
                        deploy_policy,
                        "get_semantic_utils",
                        return_value=(fake_semantic_cloud, None),
                    ),
                    patch.object(
                        deploy_policy,
                        "get_matched_utonia_utils",
                        return_value=lambda **kwargs: output,
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    deploy_policy.encode_obs(
                        make_observation(), make_model("matched_utonia")
                    )

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
                for mode in ("embedding", "xyz", "part_prob", "uniform_prob", "shuffled_prob")
            }

        self.assertEqual(outputs["embedding"].shape, (2, 7))
        self.assertEqual(outputs["xyz"].shape, (2, 3))
        self.assertEqual(outputs["part_prob"].shape, (2, 5))
        self.assertEqual(outputs["uniform_prob"].shape, (2, 5))
        self.assertEqual(outputs["shuffled_prob"].shape, (2, 5))
        self.assertTrue(np.allclose(outputs["uniform_prob"][:, 3:], 0.5))
        for output in outputs.values():
            self.assertTrue(np.array_equal(output[:, :3], queries))
        self.assertTrue(np.allclose(outputs["part_prob"][:, 3:].sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(outputs["shuffled_prob"][:, 3:], outputs["part_prob"][::-1, 3:]))
        self.assertFalse(np.array_equal(outputs["shuffled_prob"][:, 3:], outputs["part_prob"][:, 3:]))

    def test_empty_cloud_uses_mode_specific_shape(self):
        artifacts = {
            "sem_embedding_dim": 4,
            "canonical_label_names": ["Handle", "Head"],
        }
        empty = np.zeros((0, 6), dtype=np.float32)
        expected_channels = {"embedding": 7, "xyz": 3, "part_prob": 5, "uniform_prob": 5, "shuffled_prob": 5}
        for mode, channels in expected_channels.items():
            output = semantic_feature_utils.compute_semantic_pointwise_cloud(
                artifacts,
                empty,
                target_num_points=2,
                output_mode=mode,
            )
            self.assertEqual(output.shape, (2, channels))


    def test_shuffled_mode_is_seeded_by_episode_frame(self):
        queries = np.arange(12, dtype=np.float32).reshape(4, 3)
        logits = torch.tensor(
            [[[8.0, 0.0], [6.0, 1.0], [1.0, 6.0], [0.0, 8.0]]],
            dtype=torch.float32,
        )
        artifacts = {"sem_embedding_dim": 4, "canonical_label_names": ["Handle", "Head"]}
        with patch.object(
            semantic_feature_utils,
            "_compute_semantic_query",
            return_value=(queries, {"embedding": torch.zeros(1, 4, 4), "logits": logits}),
        ):
            def run(frame):
                return semantic_feature_utils.compute_semantic_pointwise_cloud(
                    artifacts,
                    np.ones((4, 6), dtype=np.float32),
                    target_num_points=4,
                    output_mode="shuffled_prob",
                    output_seed=77,
                    output_frame_index=frame,
                )

            first = run(0)
            repeated = run(0)
            second_frame = run(2)

        np.testing.assert_array_equal(first[:, :3], queries)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first[:, 3:], second_frame[:, 3:]))

    def test_encode_advances_and_reset_clears_semantic_frame_index(self):
        model = make_model("shuffled_prob")
        model.env_runner = SimpleNamespace(reset_obs=lambda: None)
        captured = []
        with patch.object(
            deploy_policy,
            "get_semantic_utils",
            return_value=(lambda **kwargs: captured.append(kwargs) or np.ones((2, 5), dtype=np.float32), None),
        ):
            deploy_policy.encode_obs(make_observation(), model)
            deploy_policy.encode_obs(make_observation(), model)
        self.assertEqual([0, 1], [item["output_frame_index"] for item in captured])
        self.assertEqual([1234, 1234], [item["output_seed"] for item in captured])
        self.assertEqual(2, model.semantic_policy_frame_index)
        deploy_policy.reset_model(model)
        self.assertEqual(0, model.semantic_policy_frame_index)

if __name__ == "__main__":
    unittest.main()
