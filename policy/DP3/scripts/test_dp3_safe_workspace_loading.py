import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = REPO_ROOT / "policy" / "DP3"
DP3_CODE_ROOT = POLICY_ROOT / "3D-Diffusion-Policy"
for path in (POLICY_ROOT, DP3_CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_dp3
from safe_dp3_checkpoint import save_weights_only_checkpoint


SHAPE_META = {
    "obs": {
        "point_cloud": {"shape": [4, 3], "type": "point_cloud"},
        "agent_pos": {"shape": [2], "type": "low_dim"},
    },
    "action": {"shape": [2]},
}


class CpuPolicy(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.cuda_called = False

    def cuda(self, *args, **kwargs):
        self.cuda_called = True
        return self


def make_cfg(*, use_ema: bool, task_name: str = "beat-paper-field-50"):
    return OmegaConf.create(
        {
            "n_obs_steps": 3,
            "n_action_steps": 6,
            "policy": {"use_pc_color": False},
            "training": {"use_ema": use_ema},
            "task": {"name": task_name, "shape_meta": SHAPE_META},
        }
    )


def make_workspace(cfg, *, with_ema: bool):
    workspace = train_dp3.TrainDP3Workspace.__new__(train_dp3.TrainDP3Workspace)
    workspace.cfg = cfg
    workspace.model = CpuPolicy()
    workspace.ema_model = CpuPolicy() if with_ema else None
    workspace.load_checkpoint = Mock(
        side_effect=AssertionError("legacy dill loader must not run for a safe checkpoint")
    )
    return workspace


class TestSafeWorkspaceLoading(unittest.TestCase):
    def test_safe_checkpoint_strictly_loads_raw_and_ema_weights(self):
        cfg = make_cfg(use_ema=True)
        source_model = CpuPolicy()
        source_ema = CpuPolicy()
        with torch.no_grad():
            source_model.weight.fill_(0.125)
            source_model.bias.fill_(0.25)
            source_ema.weight.fill_(0.75)
            source_ema.bias.fill_(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policy.weights.pt"
            save_weights_only_checkpoint(
                path,
                model=source_model,
                ema_model=source_ema,
                metadata={
                    "task_name": str(cfg.task.name),
                    "use_ema": True,
                    "shape_meta": SHAPE_META,
                },
            )
            workspace = make_workspace(cfg, with_ema=True)
            with patch.object(train_dp3, "RobotRunner", return_value="runner"):
                policy, runner = workspace.get_policy_and_runner(
                    cfg,
                    {"safe_checkpoint_path": str(path)},
                )

        torch.testing.assert_close(workspace.model.weight, source_model.weight)
        torch.testing.assert_close(workspace.ema_model.weight, source_ema.weight)
        self.assertIs(policy, workspace.ema_model)
        self.assertEqual("runner", runner)
        self.assertTrue(policy.cuda_called)
        workspace.load_checkpoint.assert_not_called()

    def test_metadata_mismatch_fails_without_legacy_fallback(self):
        cfg = make_cfg(use_ema=False)
        source_model = CpuPolicy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "wrong-task.weights.pt"
            save_weights_only_checkpoint(
                path,
                model=source_model,
                ema_model=None,
                metadata={
                    "task_name": "different-task",
                    "use_ema": False,
                    "shape_meta": SHAPE_META,
                },
            )
            workspace = make_workspace(cfg, with_ema=False)
            with (
                patch.object(train_dp3, "RobotRunner", return_value="runner"),
                self.assertRaisesRegex(ValueError, "task_name"),
            ):
                workspace.get_policy_and_runner(
                    cfg,
                    {"safe_checkpoint_path": str(path)},
                )
        workspace.load_checkpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
