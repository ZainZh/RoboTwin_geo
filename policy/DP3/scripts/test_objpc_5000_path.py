import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestObjPC5000Path(unittest.TestCase):
    def test_collection_configs_use_5000_points(self):
        clean_cfg = (REPO_ROOT / "task_config" / "demo_clean_3d_object_pc_5000.yml").read_text(encoding="utf-8")
        rand_cfg = (REPO_ROOT / "task_config" / "demo_randomized_3d_object_pc_5000.yml").read_text(encoding="utf-8")

        for text in (clean_cfg, rand_cfg):
            self.assertIn("point_num: 5000", text)
            self.assertIn("pcd_down_sample_num: 5000", text)

    def test_dp3_task_shape_uses_5000_points(self):
        task_cfg = (
            REPO_ROOT
            / "policy"
            / "DP3"
            / "3D-Diffusion-Policy"
            / "diffusion_policy_3d"
            / "config"
            / "task"
            / "demo_task_objpc_5000.yaml"
        ).read_text(encoding="utf-8")
        robot_cfg = (
            REPO_ROOT
            / "policy"
            / "DP3"
            / "3D-Diffusion-Policy"
            / "diffusion_policy_3d"
            / "config"
            / "robot_dp3_objpc_5000.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("shape: [5000, 6]", task_cfg)
        self.assertIn("task: demo_task_objpc_5000", robot_cfg)

    def test_scripts_use_5000_specific_preprocess_and_model_config(self):
        process_script = (REPO_ROOT / "policy" / "DP3" / "process_data_objpc.sh").read_text(encoding="utf-8")
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_objpc_5000.sh").read_text(encoding="utf-8")
        eval_script = (REPO_ROOT / "policy" / "DP3" / "eval_objpc_5000.sh").read_text(encoding="utf-8")

        self.assertIn("--target_num_points", process_script)
        self.assertIn("5000", train_script)
        self.assertIn("robot_dp3_objpc_5000.yaml", train_script)
        self.assertIn("robot_dp3_objpc_5000", eval_script)


if __name__ == "__main__":
    unittest.main()
