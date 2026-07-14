import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestSemanticHybridTrainingResourceOverrides(unittest.TestCase):
    def assert_ordered_snippets(self, text: str, snippets):
        cursor = -1
        for snippet in snippets:
            next_cursor = text.find(snippet, cursor + 1)
            self.assertNotEqual(next_cursor, -1, f"missing or out-of-order snippet: {snippet}")
            cursor = next_cursor

    def assert_resource_overrides(self, train_script: str):
        self.assertIn("dataloader.num_workers=${dataloader_num_workers}", train_script)
        self.assertIn("val_dataloader.num_workers=${val_dataloader_num_workers}", train_script)
        self.assertIn("dataloader.pin_memory=${pin_memory}", train_script)
        self.assertIn("val_dataloader.pin_memory=${val_pin_memory}", train_script)
        self.assertIn("training.max_val_steps=${max_val_steps}", train_script)

    def assert_point_cloud_train_overrides(self, train_script: str):
        self.assertTrue("point_cloud_num=" in train_script or "${point_cloud_num}" in train_script)
        self.assertIn("point_cloud_suffix", train_script)
        self.assertIn("task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6]", train_script)

    def test_semantic_hybrid_train_script_exposes_dataloader_resource_overrides(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_semantic_pointwise_hybrid.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("dataloader_num_workers=${17:-4}", train_script)
        self.assertIn("val_dataloader_num_workers=${18:-2}", train_script)
        self.assertIn("pin_memory=${19:-true}", train_script)
        self.assertIn("val_pin_memory=${20:-false}", train_script)
        self.assertIn("max_val_steps=${21:-2}", train_script)
        self.assert_resource_overrides(train_script)
        self.assert_point_cloud_train_overrides(train_script)

    def test_semantic_hybrid_train_script_uses_meta_feature_placeholders(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_semantic_pointwise_hybrid.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('semantic_feature_placeholders="${semantic_meta[2]:-}"', train_script)
        self.assertIn('has_semantic_feature_placeholder "{A}"', train_script)
        self.assertIn('has_semantic_feature_placeholder "{B}"', train_script)

    def test_ndf_hybrid_train_script_matches_semantic_style_argument_layout(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_ndf_pointwise_hybrid.sh").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("normalize_ndf_train_args", train_script)
        self.assert_ordered_snippets(
            train_script,
            [
                'task_name=${1}',
                'task_config=${2:-"demo_real_zed_sam2_objpc"}',
                'expert_data_num=${3:-50}',
                'seed=${4:-0}',
                'gpu_id=${5:-0}',
                'ndf_ckpt_A=${6:-none}',
                'ndf_ckpt_B=${7:-none}',
                'ndf_device=${8:-cuda:0}',
                r'object_placeholders=${9:-\{A\},\{B\}}',
                'ndf_point_num=${10:-128}',
                'ndf_feat_dim=${11:-256}',
                'batch_size=${12:-256}',
                'val_batch_size=${13:-${batch_size}}',
                'use_ema=${14:-true}',
                'gradient_accumulate_every=${15:-1}',
                'encoder_output_dim=${16:-128}',
                'dataloader_num_workers=${17:-4}',
                'val_dataloader_num_workers=${18:-2}',
                'pin_memory=${19:-true}',
                'val_pin_memory=${20:-false}',
                'max_val_steps=${21:-2}',
                'point_cloud_num=${22:-1024}',
                'ndf_dgcnn_placeholders=${23:-}',
            ],
        )
        self.assertIn('print(int(m["ndf_num_points"]))', train_script)
        self.assertIn('print(int(m.get("ndf_feat_dim", 256)))', train_script)
        self.assertIn("+task.shape_meta.obs.ndf_point_cloud_A.shape=[${ndf_point_num},$((3 + ndf_feat_dim))]", train_script)
        self.assertIn("+task.shape_meta.obs.ndf_point_cloud_B.shape=[${ndf_point_num},$((3 + ndf_feat_dim))]", train_script)
        self.assertIn("training.use_ema=${use_ema}", train_script)
        self.assertIn("policy.encoder_output_dim=${encoder_output_dim}", train_script)

    def test_semantic_hybrid_train_and_eval_use_debug_reference_suffix(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_semantic_pointwise_hybrid.sh").read_text(
            encoding="utf-8"
        )
        eval_script = (REPO_ROOT / "policy" / "DP3" / "eval_semantic_pointwise_hybrid.sh").read_text(
            encoding="utf-8"
        )

        for script in (train_script, eval_script):
            self.assertIn("semantic_input_color_mode", script)
            self.assertIn("semantic_forward_mode", script)
            self.assertIn("semantic_feature_suffix=\"-semdebugref\"", script)
            self.assertIn("semantic_input_color_mode", script)
            self.assertIn("semantic_forward_mode", script)

    def test_direct_dp3_train_scripts_forward_resource_overrides(self):
        script_names = [
            "train_objpc.sh",
            "train_objpc_5000.sh",
            "train_objpc_actorseg.sh",
            "train_ndf.sh",
            "train_ndf_pointwise.sh",
            "train_ndf_pointwise_hybrid.sh",
            "train_ndf_pointwise_hybrid_feat5000.sh",
            "train_ndf_pointwise_hybrid_interact.sh",
            "train_ndf_pointwise_actorseg_hybrid.sh",
            "train_semantic_pointwise.sh",
            "train_semantic_pointwise_hybrid.sh",
            "train_semantic_pointwise_hybrid_feat5000.sh",
            "train_semantic_pointwise_actorseg_hybrid.sh",
            "train_utonia_pointwise_hybrid.sh",
        ]
        for name in script_names:
            with self.subTest(script=name):
                train_script = (REPO_ROOT / "policy" / "DP3" / name).read_text(encoding="utf-8")
                self.assert_resource_overrides(train_script)
                self.assert_point_cloud_train_overrides(train_script)

    def test_base_dp3_train_wrappers_pass_resource_args_to_policy_helpers(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train.sh").read_text(encoding="utf-8")
        train_rgb_script = (REPO_ROOT / "policy" / "DP3" / "train_rgb.sh").read_text(encoding="utf-8")
        helper_script = (REPO_ROOT / "policy" / "DP3" / "scripts" / "train_policy.sh").read_text(encoding="utf-8")
        helper_rgb_script = (REPO_ROOT / "policy" / "DP3" / "scripts" / "train_policy_rgb.sh").read_text(
            encoding="utf-8"
        )

        for script in (train_script, train_rgb_script):
            self.assertIn("dataloader_num_workers=${6:-4}", script)
            self.assertIn("val_dataloader_num_workers=${7:-2}", script)
            self.assertIn("pin_memory=${8:-true}", script)
            self.assertIn("val_pin_memory=${9:-false}", script)
            self.assertIn("max_val_steps=${10:-2}", script)
            self.assertIn("point_cloud_num=${11:-1024}", script)
            self.assertIn("point_cloud_suffix", script)

        for script in (helper_script, helper_rgb_script):
            self.assertIn("dataloader_num_workers=${8:-4}", script)
            self.assertIn("val_dataloader_num_workers=${9:-2}", script)
            self.assertIn("pin_memory=${10:-true}", script)
            self.assertIn("val_pin_memory=${11:-false}", script)
            self.assertIn("max_val_steps=${12:-2}", script)
            self.assert_resource_overrides(script)
            self.assertIn("point_cloud_num=${13:-1024}", script)
            self.assertIn("task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6]", script)

    def test_ndf_arg_utils_parses_resource_defaults(self):
        helper = (REPO_ROOT / "policy" / "DP3" / "ndf_pointwise_arg_utils.sh").read_text(encoding="utf-8")

        self.assertIn('dataloader_num_workers="${args[14]:-4}"', helper)
        self.assertIn('val_dataloader_num_workers="${args[15]:-2}"', helper)
        self.assertIn('pin_memory="${args[16]:-true}"', helper)
        self.assertIn('val_pin_memory="${args[17]:-false}"', helper)
        self.assertIn('max_val_steps="${args[18]:-2}"', helper)
        self.assertIn('point_cloud_num="${args[19]:-1024}"', helper)

    def test_preprocess_wrappers_forward_main_point_cloud_count_and_suffix(self):
        script_names = [
            "process_data.sh",
            "process_data_objpc.sh",
            "process_data_objpc_actorseg.sh",
            "process_data_ndf.sh",
            "process_data_ndf_pointwise.sh",
            "process_data_ndf_pointwise_hybrid.sh",
            "process_data_ndf_pointwise_hybrid_feat5000.sh",
            "process_data_ndf_pointwise_hybrid_interact.sh",
            "process_data_ndf_pointwise_actorseg_hybrid.sh",
            "process_data_semantic_pointwise.sh",
            "process_data_semantic_pointwise_hybrid.sh",
            "process_data_semantic_pointwise_hybrid_feat5000.sh",
            "process_data_semantic_pointwise_actorseg_hybrid.sh",
            "process_data_utonia_pointwise_hybrid.sh",
        ]
        for name in script_names:
            with self.subTest(script=name):
                process_script = (REPO_ROOT / "policy" / "DP3" / name).read_text(encoding="utf-8")
                self.assertIn("target_num_points", process_script)
                self.assertIn("output_suffix", process_script)
                self.assertIn("--target_num_points", process_script)
                self.assertIn("--output_suffix", process_script)
                self.assertIn('--output_suffix="${output_suffix}"', process_script)
                self.assertNotIn('--output_suffix "${output_suffix}"', process_script)

    def test_deploy_policy_applies_eval_point_cloud_num_override(self):
        deploy_policy = (REPO_ROOT / "policy" / "DP3" / "deploy_policy.py").read_text(encoding="utf-8")

        self.assertIn('usr_args.get("point_cloud_num"', deploy_policy)
        self.assertIn("cfg.task.shape_meta.obs.point_cloud.shape = [point_cloud_num, point_cloud_dim]", deploy_policy)
        self.assertIn("target_num_points = point_cloud_num", deploy_policy)

    def test_objpc_train_fails_fast_when_preprocess_fails(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_objpc.sh").read_text(encoding="utf-8")

        self.assertIn("set -euo pipefail", train_script)


if __name__ == "__main__":
    unittest.main()
