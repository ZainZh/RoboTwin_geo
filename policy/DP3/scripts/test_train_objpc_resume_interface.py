import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestTrainObjpcResumeInterface(unittest.TestCase):
    def test_train_objpc_exposes_resume_flag(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_objpc.sh").read_text(encoding="utf-8")

        self.assertIn("resume=${6:-true}", train_script)
        self.assertIn("object_placeholders=${7:-\\{A\\},\\{B\\}}", train_script)
        self.assertIn("training.resume=${resume}", train_script)

    def test_shoe_comparison_exposes_explicit_checkpoint_resume(self):
        wrapper = (
            REPO_ROOT / "policy" / "DP3" / "train_shoe_se3_placement_comparison.sh"
        ).read_text(encoding="utf-8")
        trainer = (
            REPO_ROOT / "policy" / "DP3" / "3D-Diffusion-Policy" / "train_dp3.py"
        ).read_text(encoding="utf-8")
        self.assertIn("resume_checkpoint=${23:-}", wrapper)
        self.assertIn("experiment_tag=${24:-}", wrapper)
        self.assertIn("policy_down_dims=${25:-}", wrapper)
        self.assertIn('architecture_overrides+=(policy.down_dims="${policy_down_dims}")', wrapper)
        self.assertIn('train_setting="${task_config}${output_suffix}${experiment_suffix}"', wrapper)
        self.assertIn("+training.resume_checkpoint=", wrapper)
        self.assertIn("training.resume_checkpoint", trainer)
        self.assertIn("Resuming from explicit checkpoint", trainer)
        self.assertIn("self.epoch += 1", trainer)
        self.assertIn("self.global_step += 1", trainer)
        self.assertIn("ema.optimization_step = int(self.global_step)", trainer)
        self.assertIn(
            "remaining_epochs = max(0, int(cfg.training.num_epochs) - int(self.epoch))",
            trainer,
        )
        self.assertIn('log_file.write(json.dumps(persistent_log, sort_keys=True)', trainer)

    def test_shoe_eval_exposes_matched_architecture_override(self):
        wrapper = (
            REPO_ROOT / "policy" / "DP3" / "eval_shoe_se3_placement_comparison.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("experiment_tag=${17:-}", wrapper)
        self.assertIn("policy_down_dims=${18:-}", wrapper)
        self.assertIn('route_overrides+=(--policy_down_dims "${policy_down_dims}")', wrapper)

    def test_partial_training_starts_from_deployed_ema(self):
        trainer = (
            REPO_ROOT / "policy" / "DP3" / "3D-Diffusion-Policy" / "train_dp3.py"
        ).read_text(encoding="utf-8")
        config = (
            REPO_ROOT
            / "policy"
            / "DP3"
            / "3D-Diffusion-Policy"
            / "diffusion_policy_3d"
            / "config"
            / "robot_dp3_objpc.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("partial_training_start_from_ema", trainer)
        self.assertIn(
            "self.model.load_state_dict(self.ema_model.state_dict(), strict=True)",
            trainer,
        )
        self.assertIn("Preserved deployed EMA normalizer for partial training", trainer)
        self.assertIn("partial_training_start_from_ema: true", config)


if __name__ == "__main__":
    unittest.main()
