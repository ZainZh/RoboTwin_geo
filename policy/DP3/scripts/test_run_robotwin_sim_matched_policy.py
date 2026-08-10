import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from safe_dp3_checkpoint import save_weights_only_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER = SCRIPT_DIR / "run_robotwin_sim_matched_policy.sh"


class TestRunRobotwinSimMatchedPolicy(unittest.TestCase):
    def _environment(self, output_root: Path, **overrides: str) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "OUTPUT_ROOT": str(output_root),
                "DRY_RUN": "1",
                "RUN_ROLE": "remote",
                "EPOCHS": "1",
                "SEED": "20260805",
            }
        )
        environment.update(overrides)
        return environment

    def _run(
        self,
        zarr_path: Path,
        output_root: Path,
        *,
        route: str = "utonia",
        feature_dim: int = 579,
        **overrides: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                route,
                "0",
                "beat_block_hammer",
                str(zarr_path),
                str(feature_dim),
            ],
            env=self._environment(output_root, **overrides),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_remote_dry_run_defaults_to_bf16_tf32_workers0_and_custom_obs_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zarr_path = root / "matched.zarr"
            zarr_path.mkdir()
            result = self._run(
                zarr_path,
                root / "outputs",
                OBS_KEY="utonia_point_cloud_A",
            )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("precision=bf16", result.stdout)
        self.assertIn("tf32=true", result.stdout)
        self.assertIn("training.precision=bf16", result.stdout)
        self.assertIn("training.allow_tf32=true", result.stdout)
        self.assertIn("dataloader.num_workers=0", result.stdout)
        self.assertIn("val_dataloader.num_workers=0", result.stdout)
        self.assertIn("utonia_point_cloud_A", result.stdout)

    def test_vanilla_uses_objpc_config_without_auxiliary_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zarr_path = root / "matched.zarr"
            zarr_path.mkdir()
            result = self._run(
                zarr_path,
                root / "outputs",
                route="vanilla",
                feature_dim=0,
            )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--config-name=robot_dp3_objpc.yaml", result.stdout)
        self.assertNotIn("extra_obs_keys", result.stdout)
        self.assertNotIn("shape_meta.obs.semantic_point_cloud_A", result.stdout)

    def test_fusion_mode_has_independent_identity_and_hydra_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zarr_path = root / "matched.zarr"
            zarr_path.mkdir()
            result = self._run(
                zarr_path,
                root / "outputs",
                route="field",
                feature_dim=131,
                POINTCLOUD_FUSION_MODE="state_cross_attention",
                SEMANTIC_GATE_INIT="0.0",
                SEMANTIC_GATE_TRAINABLE="true",
                SEMANTIC_ATTENTION_HEADS="8",
                PART_POOL_TEMPERATURE="0.5",
            )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("field-fusion-state_cross_attention", result.stdout)
        self.assertIn("policy.pointcloud_fusion_mode=state_cross_attention", result.stdout)
        self.assertIn("policy.semantic_gate_trainable=true", result.stdout)
        self.assertIn("policy.semantic_attention_heads=8", result.stdout)
        self.assertIn("policy.part_pool_temperature=0.5", result.stdout)
        self.assertIn("shape_meta.obs.semantic_point_cloud_A.shape", result.stdout)

    def test_frozen_zero_gate_control_is_explicit_and_vanilla_rejects_fusion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zarr_path = root / "matched.zarr"
            zarr_path.mkdir()
            frozen = self._run(
                zarr_path,
                root / "outputs",
                route="field",
                feature_dim=131,
                POINTCLOUD_FUSION_MODE="residual_pointnet",
                SEMANTIC_GATE_TRAINABLE="false",
            )
            vanilla = self._run(
                zarr_path,
                root / "outputs",
                route="vanilla",
                feature_dim=0,
                POINTCLOUD_FUSION_MODE="residual_pointnet",
            )
        self.assertEqual(0, frozen.returncode, frozen.stderr)
        self.assertIn("gtrainfalse", frozen.stdout)
        self.assertIn("policy.semantic_gate_init=0.0", frozen.stdout)
        self.assertIn("policy.semantic_gate_trainable=false", frozen.stdout)
        self.assertEqual(2, vanilla.returncode)
        self.assertIn("Strict Vanilla", vanilla.stderr)

    def test_valid_existing_checkpoint_is_verified_before_skip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zarr_path = root / "matched.zarr"
            zarr_path.mkdir()
            output_root = root / "outputs"
            run_name = "beat_block_hammer_utonia_joint14_e1_seed20260805"
            weights_path = output_root / run_name / f"{run_name}.weights.pt"
            save_weights_only_checkpoint(
                weights_path,
                model=torch.nn.Linear(2, 1),
                ema_model=None,
                metadata={
                    "epoch": 1,
                    "seed": 20260805,
                    "task_name": "beat_block_hammer-paper-sim-utonia-joint14-e1-50",
                    "zarr_path": str(zarr_path),
                    "precision": "bf16",
                    "allow_tf32": True,
                },
            )
            result = self._run(zarr_path, output_root)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("validated completed weights", result.stdout)
        self.assertNotIn("train_dp3.py", result.stdout)

    def test_corrupt_existing_checkpoint_does_not_skip_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zarr_path = root / "matched.zarr"
            zarr_path.mkdir()
            output_root = root / "outputs"
            run_name = "beat_block_hammer_utonia_joint14_e1_seed20260805"
            weights_path = output_root / run_name / f"{run_name}.weights.pt"
            weights_path.parent.mkdir(parents=True)
            weights_path.write_bytes(b"not-a-checkpoint")
            result = self._run(zarr_path, output_root)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("failed validation", result.stderr)
        self.assertIn("train_dp3.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
