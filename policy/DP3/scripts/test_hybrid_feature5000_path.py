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

    def test_scripts_default_feature_branch_to_5000_and_use_distinct_names(self):
        train_ndf = (POLICY_ROOT / "train_ndf_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")
        eval_ndf = (POLICY_ROOT / "eval_ndf_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")
        train_sem = (POLICY_ROOT / "train_semantic_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")
        eval_sem = (POLICY_ROOT / "eval_semantic_pointwise_hybrid_feat5000.sh").read_text(encoding="utf-8")

        self.assertIn('ndf_point_num=${ndf_point_num:-5000}', train_ndf)
        self.assertIn('ndf_point_num=${13:-5000}', eval_ndf)
        self.assertIn('semantic_point_num=${10:-5000}', train_sem)
        self.assertIn('semantic_point_num=${12:-5000}', eval_sem)

        self.assertIn("objpc-ndf-pointwise-hybrid-feat5000", train_ndf)
        self.assertIn("objpc-ndf-pointwise-hybrid-feat5000", eval_ndf)
        self.assertIn("objpc-semantic-pointwise-hybrid-feat5000", train_sem)
        self.assertIn("objpc-semantic-pointwise-hybrid-feat5000", eval_sem)


if __name__ == "__main__":
    unittest.main()
