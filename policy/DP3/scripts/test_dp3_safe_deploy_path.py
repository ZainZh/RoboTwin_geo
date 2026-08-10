import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from types import SimpleNamespace
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = REPO_ROOT / "policy" / "DP3"
DP3_CODE_ROOT = POLICY_ROOT / "3D-Diffusion-Policy"
for path in (POLICY_ROOT, DP3_CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import deploy_policy
import dp3_policy


SHAPE_META = {
    "obs": {
        "point_cloud": {"shape": [1024, 6], "type": "point_cloud"},
        "agent_pos": {"shape": [14], "type": "low_dim"},
        "semantic_point_cloud_A": {"shape": [128, 5], "type": "point_cloud"},
    },
    "action": {"shape": [14]},
}


def safe_payload(**metadata_overrides):
    metadata = {
        "task_name": "beat_block_hammer-paper-sim-partprob-joint14-e300-50",
        "use_ema": True,
        "shape_meta": SHAPE_META,
    }
    metadata.update(metadata_overrides)
    return {
        "format": "dp3_weights_only_v1",
        "metadata": metadata,
        "model": {},
        "ema_model": {},
    }


class TestSafeDeployPath(unittest.TestCase):
    def test_explicit_path_is_resolved_and_missing_path_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "policy.weights.pt"
            checkpoint.touch()
            resolved = deploy_policy.resolve_safe_checkpoint_path(
                {"safe_checkpoint_path": str(checkpoint)}
            )
            self.assertEqual(checkpoint.resolve(), resolved)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                deploy_policy.resolve_safe_checkpoint_path(
                    {"safe_checkpoint_path": str(checkpoint.with_name("missing.pt"))}
                )
        self.assertIsNone(deploy_policy.resolve_safe_checkpoint_path({}))

    def test_metadata_is_read_only_through_safe_loader_and_validated(self):
        checkpoint = Path("/tmp/does-not-need-to-exist.weights.pt")
        payload = safe_payload()
        with patch.object(
            deploy_policy, "load_weights_only_checkpoint", return_value=payload
        ) as loader:
            metadata = deploy_policy.inspect_safe_policy_metadata(
                checkpoint,
                expected_task_name=payload["metadata"]["task_name"],
            )
        loader.assert_called_once_with(checkpoint, mmap=True)
        self.assertTrue(metadata["use_ema"])

        for broken, message in (
            (safe_payload(task_name="wrong-task"), "task metadata mismatch"),
            (safe_payload(use_ema="yes"), "must be a bool"),
            ({**safe_payload(), "metadata": {"task_name": "x"}}, "metadata is missing"),
            ({**safe_payload(), "ema_model": None}, "ema_model is absent"),
        ):
            with patch.object(
                deploy_policy, "load_weights_only_checkpoint", return_value=broken
            ):
                with self.assertRaisesRegex(ValueError, message):
                    deploy_policy.inspect_safe_policy_metadata(
                        checkpoint,
                        expected_task_name=payload["metadata"]["task_name"],
                    )

    def test_shape_meta_mismatch_is_rejected(self):
        cfg = OmegaConf.create({"task": {"shape_meta": SHAPE_META}})
        deploy_policy.validate_safe_policy_shape_meta(safe_payload()["metadata"], cfg)
        wrong = safe_payload(shape_meta={"action": {"shape": [20]}})["metadata"]
        with self.assertRaisesRegex(ValueError, "shape_meta mismatch"):
            deploy_policy.validate_safe_policy_shape_meta(wrong, cfg)

    def test_dp3_wrapper_preserves_explicit_safe_path_for_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.weights.pt"
            args = {"safe_checkpoint_path": str(checkpoint)}
            workspace = Mock()
            workspace.get_policy_and_runner.return_value = ("policy", "runner")
            with patch.object(dp3_policy, "TrainDP3Workspace", return_value=workspace):
                wrapped = dp3_policy.DP3("cfg", args)

        self.assertEqual(str(checkpoint.resolve()), wrapped.safe_checkpoint_path)
        workspace.get_policy_and_runner.assert_called_once_with("cfg", args)


    def test_get_model_safe_path_never_calls_legacy_checkpoint_inspection(self):
        task_name = "beat_block_hammer"
        setting = "paper-sim-partprob-joint14-e300"
        payload = safe_payload(
            task_name=f"{task_name}-{setting}-50",
            use_ema=True,
            pointcloud_fusion_mode="state_cross_attention",
            semantic_gate_trainable=True,
            semantic_attention_heads=8,
            part_pool_temperature=0.5,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "policy.weights.pt"
            checkpoint.touch()
            args = {
                "safe_checkpoint_path": str(checkpoint),
                "policy_name": "DP3",
                "task_name": task_name,
                "task_config": "demo_clean_3d_object_pc",
                "ckpt_setting": setting,
                "expert_data_num": "50",
                "seed": "20260806",
                "config_name": "robot_dp3_semantic_pointwise_hybrid",
                "checkpoint_num": "300",
                "use_rgb": False,
                "object_placeholders": "{A},{B}",
                "point_cloud_num": "1024",
                "semantic_ckpt_A": "mock-semantic.pt",
                "semantic_ckpt_B": "none",
                "semantic_point_num": "128",
                "semantic_policy_output_mode": "part_prob",
            }
            model = SimpleNamespace()
            old_cwd = Path.cwd()
            try:
                os.chdir(POLICY_ROOT)
                with (
                    patch.object(deploy_policy, "load_weights_only_checkpoint", return_value=payload),
                    patch.object(
                        deploy_policy,
                        "infer_checkpoint_use_ema",
                        side_effect=AssertionError("legacy checkpoint inspection must not run"),
                    ),
                    patch.object(
                        deploy_policy,
                        "get_semantic_utils",
                        return_value=(None, lambda **kwargs: {"canonical_label_names": ["Handle", "Head"]}),
                    ),
                    patch.object(deploy_policy, "DP3", return_value=model) as dp3_constructor,
                ):
                    result = deploy_policy.get_model(args)
            finally:
                os.chdir(old_cwd)

        self.assertIs(model, result)
        cfg = dp3_constructor.call_args.args[0]
        self.assertTrue(cfg.training.use_ema)
        self.assertEqual([128, 5], list(cfg.task.shape_meta.obs.semantic_point_cloud_A.shape))
        self.assertEqual("state_cross_attention", cfg.policy.pointcloud_fusion_mode)
        self.assertTrue(cfg.policy.semantic_gate_trainable)
        self.assertEqual(8, cfg.policy.semantic_attention_heads)
        self.assertEqual(0.5, cfg.policy.part_pool_temperature)

    def test_matched_utonia_builds_strict_safe_shape_meta_and_loads_both_models(self):
        task_name = "beat_block_hammer"
        setting = "paper-sim-utonia-joint14-e300"
        matched_shape_meta = {
            "obs": {
                "point_cloud": {"shape": [1024, 6], "type": "point_cloud"},
                "agent_pos": {"shape": [14], "type": "low_dim"},
                "utonia_point_cloud_A": {
                    "shape": [128, 579],
                    "type": "point_cloud",
                },
            },
            "action": {"shape": [14]},
        }
        payload = safe_payload(
            task_name=f"{task_name}-{setting}-50",
            shape_meta=matched_shape_meta,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "policy.weights.pt"
            checkpoint.touch()
            args = {
                "safe_checkpoint_path": str(checkpoint),
                "policy_name": "DP3",
                "task_name": task_name,
                "task_config": "demo_clean_3d_object_pc",
                "ckpt_setting": setting,
                "expert_data_num": "50",
                "seed": "20260806",
                "config_name": "robot_dp3_semantic_pointwise_hybrid",
                "checkpoint_num": "300",
                "use_rgb": False,
                "object_placeholders": "{A},{B}",
                "point_cloud_num": "1024",
                "semantic_ckpt_A": "mock-semantic.pt",
                "semantic_ckpt_B": "none",
                "semantic_point_num": "128",
                "semantic_policy_output_mode": "matched_utonia",
                "utonia_checkpoint": "mock-utonia.pt",
                "utonia_feature_placeholders": "{A}",
            }
            model = SimpleNamespace()
            semantic_loader = Mock(
                return_value={
                    "sem_embedding_dim": 128,
                    "canonical_label_names": ["Handle", "Head"],
                }
            )
            utonia_loader = Mock(return_value={"feature_dim": 576})
            old_cwd = Path.cwd()
            try:
                os.chdir(POLICY_ROOT)
                with (
                    patch.object(
                        deploy_policy,
                        "load_weights_only_checkpoint",
                        return_value=payload,
                    ),
                    patch.object(
                        deploy_policy,
                        "get_semantic_utils",
                        return_value=(None, semantic_loader),
                    ),
                    patch.object(
                        deploy_policy,
                        "get_utonia_utils",
                        return_value=(None, utonia_loader),
                    ),
                    patch.object(deploy_policy, "DP3", return_value=model) as constructor,
                ):
                    result = deploy_policy.get_model(args)
            finally:
                os.chdir(old_cwd)

        self.assertIs(model, result)
        cfg = constructor.call_args.args[0]
        self.assertEqual(
            [128, 579], list(cfg.task.shape_meta.obs.utonia_point_cloud_A.shape)
        )
        self.assertNotIn("semantic_point_cloud_A", cfg.task.shape_meta.obs)
        semantic_loader.assert_called_once()
        utonia_loader.assert_called_once()
        self.assertEqual({"{A}"}, set(result.semantic_models))
        self.assertEqual({"{A}"}, set(result.utonia_models))

if __name__ == "__main__":
    unittest.main()
