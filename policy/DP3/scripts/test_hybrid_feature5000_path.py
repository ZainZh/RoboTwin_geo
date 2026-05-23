import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = REPO_ROOT / "policy" / "DP3"


class TestHybridFeature5000Path(unittest.TestCase):
    def test_ndf_feature5000_task_shape_keeps_raw_point_cloud_at_1024(self):
        task_cfg = (
            POLICY_ROOT
            / "3D-Diffusion-Policy"
            / "diffusion_policy_3d"
            / "config"
            / "task"
            / "demo_task_ndf_pointwise_hybrid_feat5000.yaml"
        ).read_text(encoding="utf-8")
        robot_cfg = (
            POLICY_ROOT
            / "3D-Diffusion-Policy"
            / "diffusion_policy_3d"
            / "config"
            / "robot_dp3_ndf_pointwise_hybrid_feat5000.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("shape: [1024, 6]", task_cfg)
        self.assertIn("task: demo_task_ndf_pointwise_hybrid_feat5000", robot_cfg)

    def test_semantic_feature5000_task_shape_keeps_raw_point_cloud_at_1024(self):
        task_cfg = (
            POLICY_ROOT
            / "3D-Diffusion-Policy"
            / "diffusion_policy_3d"
            / "config"
            / "task"
            / "demo_task_semantic_pointwise_hybrid_feat5000.yaml"
        ).read_text(encoding="utf-8")
        robot_cfg = (
            POLICY_ROOT
            / "3D-Diffusion-Policy"
            / "diffusion_policy_3d"
            / "config"
            / "robot_dp3_semantic_pointwise_hybrid_feat5000.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("shape: [1024, 6]", task_cfg)
        self.assertIn("task: demo_task_semantic_pointwise_hybrid_feat5000", robot_cfg)

    def test_scripts_default_feature_branch_to_5000_and_use_dynamic_suffixes(self):
        train_ndf = (POLICY_ROOT / "train_ndf_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")
        eval_ndf = (POLICY_ROOT / "eval_ndf_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")
        train_sem = (POLICY_ROOT / "train_semantic_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")
        eval_sem = (POLICY_ROOT / "eval_semantic_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")

        self.assertIn('ndf_point_num=${11:-5000}', train_ndf)
        self.assertIn('ndf_point_num=${13:-5000}', eval_ndf)
        self.assertIn('semantic_point_num=${10:-5000}', train_sem)
        self.assertIn('semantic_point_num=${12:-5000}', eval_sem)

        self.assertIn('output_suffix="-objpc-ndf-pointwise-hybrid-feat${ndf_point_num}"', train_ndf)
        self.assertIn('output_suffix="-objpc-ndf-pointwise-hybrid-feat${ndf_point_num}"', eval_ndf)
        self.assertIn('output_suffix="-objpc-semantic-pointwise-hybrid-semdebugref-feat${semantic_point_num}"', train_sem)
        self.assertIn('output_suffix="-objpc-semantic-pointwise-hybrid${semantic_feature_suffix}-feat${semantic_point_num}"', eval_sem)

    def test_wrapper_scripts_build_dynamic_output_suffix_from_point_count(self):
        ndf_spec = importlib.util.spec_from_file_location(
            "process_data_ndf_pointwise_hybrid_feat5000",
            POLICY_ROOT / "scripts" / "process_data_ndf_pointwise_hybrid_feat5000.py",
        )
        ndf_module = importlib.util.module_from_spec(ndf_spec)
        assert ndf_spec.loader is not None
        ndf_spec.loader.exec_module(ndf_module)

        sem_spec = importlib.util.spec_from_file_location(
            "process_data_semantic_pointwise_hybrid_feat5000",
            POLICY_ROOT / "scripts" / "process_data_semantic_pointwise_hybrid_feat5000.py",
        )
        sem_module = importlib.util.module_from_spec(sem_spec)
        assert sem_spec.loader is not None
        sem_spec.loader.exec_module(sem_module)

        ndf_argv = ndf_module.build_hybrid_argv(["task", "cfg", "50", "--ndf_num_points", "512"])
        sem_argv = sem_module.build_hybrid_argv(["task", "cfg", "50", "--semantic_num_points", "768"])

        self.assertIn("--output_suffix=-objpc-ndf-pointwise-hybrid-feat512", ndf_argv)
        self.assertIn("--output_suffix=-objpc-semantic-pointwise-hybrid-semdebugref-feat768", sem_argv)


if __name__ == "__main__":
    unittest.main()
