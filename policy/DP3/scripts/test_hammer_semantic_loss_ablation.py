from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from train_semantic_field_loss_ablation_safe import (
    FORMAT_NAME,
    _reject_resume,
    save_weights_only_checkpoint,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER = SCRIPT_DIR / "run_hammer_semantic_loss_ablation_matrix.sh"


class _FakeFieldModel:
    def __init__(self) -> None:
        for name in (
            "semantic_adapter",
            "pose_adapter",
            "semantic_local_fusion",
            "pose_local_fusion",
            "occupancy_local_fusion",
            "semantic_decoder",
            "pose_decoder",
            "occupancy_decoder",
        ):
            setattr(self, name, torch.nn.Linear(2, 2))
        self.feature_extractor = SimpleNamespace(freeze_encoder=True)


class TestHammerSemanticLossAblation(unittest.TestCase):
    def test_launcher_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_dry_run_changes_exactly_one_loss_per_ablation(self):
        environment = os.environ.copy()
        environment.update(
            {
                "DRY_RUN": "1",
                "SEEDS": "7",
                "EPOCHS": "2",
                "PYTHON_BIN": "python-for-test",
                "DATASET_ROOT": "/dataset-for-test",
                "OUTPUT_ROOT": "/output-for-test",
                "UTONIA_CHECKPOINT": "/utonia-for-test.pth",
                "ALIAS_CONFIG": "/hammer-for-test.json",
            }
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        commands = completed.stdout.splitlines()
        self.assertEqual(4, len(commands))
        by_variant = {variant: next(line for line in commands if f"hammer_{variant}_" in line) for variant in ("full_fixed", "no_ce", "no_supcon", "no_consistency")}

        expected = {
            "full_fixed": ("1.0", "0.2", "0.1"),
            "no_ce": ("0.0", "0.2", "0.1"),
            "no_supcon": ("1.0", "0.0", "0.1"),
            "no_consistency": ("1.0", "0.2", "0.0"),
        }
        for variant, (ce, supcon, consistency) in expected.items():
            command = by_variant[variant]
            self.assertIn(f"--sem-ce-weight {ce}", command)
            self.assertIn(f"--sem-contrastive-weight {supcon}", command)
            self.assertIn(f"--sem-consistency-weight {consistency}", command)
            self.assertIn("--split-seed 42", command)
            self.assertIn("--seed 7", command)
            self.assertIn("--save-every 2", command)

    def test_checkpoint_is_weights_only_and_omits_optimizer(self):
        model = _FakeFieldModel()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "safe.pt"
            save_weights_only_checkpoint(
                path,
                model=model,
                optimizer=object(),
                scaler=object(),
                epoch=3,
                global_step=9,
                best_sem_score=0.8,
                best_pose_score=-1.0,
                best_joint_score=0.7,
                args=argparse.Namespace(seed=7, dataset_root=Path("/data")),
                canonical_label_names=["Handle", "Head"],
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(FORMAT_NAME, payload["format"])
        self.assertFalse(payload["training_resume_supported"])
        self.assertNotIn("optimizer_state_dict", payload)
        self.assertNotIn("scaler_state_dict", payload)
        self.assertEqual("/data", payload["args"]["dataset_root"])
    def test_dry_run_supports_explicit_conda_environment(self):
        environment = os.environ.copy()
        environment.update(
            {
                "CONDA_ENV": "RoboTwin",
                "DRY_RUN": "1",
                "SEEDS": "7",
                "VARIANTS": "full_fixed",
                "EPOCHS": "1",
            }
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertTrue(completed.stdout.startswith(
            "conda run --no-capture-output -n RoboTwin python "
        ))



    def test_resume_is_rejected(self):
        with self.assertRaises(SystemExit):
            _reject_resume(["--resume", "checkpoint.pt"])
        with self.assertRaises(SystemExit):
            _reject_resume(["--resume=checkpoint.pt"])


if __name__ == "__main__":
    unittest.main()
