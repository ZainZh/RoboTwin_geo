import unittest
import sys
import importlib.util
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import process_data_semantic_pointwise_eef_absolute6d


def load_module_from_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEefPointwiseWrappers(unittest.TestCase):
    def test_semantic_hybrid_eef_wrapper_adds_suffix_and_eef_action_mode(self):
        argv = [
            "grasp_mug",
            "demo_real_zed_sam2_objpc",
            "32",
            "--semantic_model",
            "{A}=/tmp/mug.pt",
        ]

        forwarded = process_data_semantic_pointwise_eef_absolute6d.build_eef_argv(argv, hybrid=True)

        self.assertIn("--output_suffix=-objpc-semantic-pointwise-hybrid-semdebugref-eef-absolute6d-rightbase", forwarded)
        self.assertIn("--keep_feature_placeholders_in_context", forwarded)
        self.assertIn("--action_mode=eef_absolute6d", forwarded)

    def test_real_semantic_eef_wrapper_defaults_pointcloud_and_action_frame_to_right_base(self):
        wrapper = (SCRIPT_ROOT.parent / "real_infer_semantic_pointwise_hybrid_eef_absolute6d.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("output_frame=${12:-right_base}", wrapper)
        self.assertIn("eef_frame_mode=${13:-${output_frame}}", wrapper)
        self.assertIn('--eef_frame_mode "${eef_frame_mode}"', wrapper)

    def test_semantic_hybrid_eef_global_wrapper_uses_global_suffix(self):
        module = load_module_from_script(
            SCRIPT_ROOT / "process_data_semantic_pointwise_eef_absolute6d_global.py",
            "process_data_semantic_pointwise_eef_absolute6d_global",
        )

        forwarded = module.build_eef_argv(
            [
                "grasp_mug",
                "demo_real_zed_sam2_objpc_global",
                "32",
                "--semantic_model",
                "{A}=/tmp/mug.pt",
            ],
            hybrid=True,
        )

        self.assertIn("--output_suffix=-objpc-semantic-pointwise-hybrid-semdebugref-eef-absolute6d-global", forwarded)
        self.assertIn("--action_mode=eef_absolute6d", forwarded)
        self.assertIn("--keep_feature_placeholders_in_context", forwarded)

    def test_global_eef_shell_wrappers_use_source_and_reference_camera(self):
        semantic_process = (SCRIPT_ROOT.parent / "process_data_semantic_pointwise_hybrid_eef_absolute6d_global.sh").read_text(
            encoding="utf-8"
        )
        semantic_train = (SCRIPT_ROOT.parent / "train_semantic_pointwise_hybrid_eef_absolute6d_global.sh").read_text(
            encoding="utf-8"
        )
        semantic_infer = (SCRIPT_ROOT.parent / "real_infer_semantic_pointwise_hybrid_eef_absolute6d_global.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("eef_frame_mode=${12:-reference_camera}", semantic_process)
        self.assertIn('process_data_semantic_pointwise_hybrid_eef_absolute6d_global.py', semantic_process)
        self.assertIn('output_suffix="-objpc-semantic-pointwise-hybrid${semantic_feature_suffix}-eef-absolute6d-global${point_cloud_suffix}"', semantic_train)
        self.assertIn("task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6]", semantic_train)
        self.assertIn("ckpt_setting=\"${task_config}-objpc-semantic-pointwise-hybrid${semantic_feature_suffix}-eef-absolute6d-global\"", semantic_infer)
        self.assertIn("output_frame=${12:-source}", semantic_infer)
        self.assertIn("eef_frame_mode=${13:-reference_camera}", semantic_infer)

    def test_train_policy_accepts_explicit_zarr_path_override(self):
        train_policy = (SCRIPT_ROOT.parent / "scripts" / "train_policy.sh").read_text(encoding="utf-8")

        self.assertIn("zarr_path=${14:-}", train_policy)
        self.assertIn('dataset_overrides=()', train_policy)
        self.assertIn('task.dataset.zarr_path="${zarr_path}"', train_policy)
        self.assertIn('"${dataset_overrides[@]}"', train_policy)

    def test_objpc_eef_train_wrappers_pass_explicit_zarr_path(self):
        for script_name, suffix in [
            ("train_objpc_eef_absolute6d.sh", "-objpc-eef-absolute6d-rightbase"),
            ("train_objpc_eef_absolute6d_global.sh", "-objpc-eef-absolute6d-global"),
        ]:
            with self.subTest(script_name=script_name):
                wrapper = (SCRIPT_ROOT.parent / script_name).read_text(encoding="utf-8")

                self.assertIn(
                    f'output_suffix="{suffix}${{point_cloud_suffix}}"',
                    wrapper,
                )
                self.assertIn(
                    'zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"',
                    wrapper,
                )
                self.assertIn('"${zarr_path}"', wrapper)


if __name__ == "__main__":
    unittest.main()
