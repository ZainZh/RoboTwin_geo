import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestSemanticHybridTrainingResourceOverrides(unittest.TestCase):
    def assert_resource_overrides(self, train_script: str):
        self.assertIn("dataloader.num_workers=${dataloader_num_workers}", train_script)
        self.assertIn("val_dataloader.num_workers=${val_dataloader_num_workers}", train_script)
        self.assertIn("dataloader.pin_memory=${pin_memory}", train_script)
        self.assertIn("val_dataloader.pin_memory=${val_pin_memory}", train_script)
        self.assertIn("training.max_val_steps=${max_val_steps}", train_script)

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

        for script in (helper_script, helper_rgb_script):
            self.assertIn("dataloader_num_workers=${8:-4}", script)
            self.assertIn("val_dataloader_num_workers=${9:-2}", script)
            self.assertIn("pin_memory=${10:-true}", script)
            self.assertIn("val_pin_memory=${11:-false}", script)
            self.assertIn("max_val_steps=${12:-2}", script)
            self.assert_resource_overrides(script)

    def test_ndf_arg_utils_parses_resource_defaults(self):
        helper = (REPO_ROOT / "policy" / "DP3" / "ndf_pointwise_arg_utils.sh").read_text(encoding="utf-8")

        self.assertIn('dataloader_num_workers="${args[14]:-4}"', helper)
        self.assertIn('val_dataloader_num_workers="${args[15]:-2}"', helper)
        self.assertIn('pin_memory="${args[16]:-true}"', helper)
        self.assertIn('val_pin_memory="${args[17]:-false}"', helper)
        self.assertIn('max_val_steps="${args[18]:-2}"', helper)


if __name__ == "__main__":
    unittest.main()
