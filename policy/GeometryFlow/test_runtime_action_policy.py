from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .runtime_action_policy import (
    FlowActionRuntime,
    RecoveryAwareFlowRuntime,
    load_flow_action_runtime,
)
from .train_recovery_aware_flow import RecoveryAwareFlowPolicy
from .train_task_flow_benchmark import ActionPredictor


class FlowActionRuntimeTest(unittest.TestCase):
    def test_checkpoint_load_and_prediction_shape(self):
        torch.manual_seed(3)
        model = ActionPredictor("flow", flow_steps=4, horizon=2)
        normalization = {
            "state_mean": np.zeros(20, dtype=np.float32),
            "state_std": np.ones(20, dtype=np.float32),
            "action_mean": np.zeros(14, dtype=np.float32),
            "action_std": np.ones(14, dtype=np.float32),
            "xyz_mean": np.zeros(3, dtype=np.float32),
            "xyz_std": np.ones(3, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "condition": "flow",
                    "flow_steps": 4,
                    "horizon": 2,
                    "normalization": normalization,
                },
                path,
            )
            runtime = FlowActionRuntime(path, observation_points=8)
            generator = np.random.default_rng(7)
            anchors = generator.normal(size=(6, 3)).astype(np.float32)
            flow = np.stack(
                [anchors + (0.01 * step, 0.0, 0.0) for step in range(4)], axis=1
            )
            action = runtime.predict(
                operated_point_cloud=generator.normal(size=(12, 3)),
                target_point_cloud=generator.normal(size=(14, 3)),
                eef_state20=np.zeros(20, dtype=np.float32),
                current_anchors=anchors,
                predicted_episode_flow=flow,
            )
            self.assertEqual(action.shape, (2, 14))
            self.assertTrue(np.all(np.isfinite(action)))

    def test_recovery_runtime_routes_active_arm_and_phase(self):
        model = RecoveryAwareFlowPolicy("flow", flow_steps=4, horizon=2, hidden_dim=32)
        with torch.no_grad():
            model.phase_head[-1].weight.zero_()
            model.phase_head[-1].bias.copy_(torch.tensor([0.0, 0.0, 10.0]))
        normalization = {
            "state_mean": np.zeros(20, dtype=np.float32),
            "state_std": np.ones(20, dtype=np.float32),
            "action_mean": np.zeros(14, dtype=np.float32),
            "action_std": np.ones(14, dtype=np.float32),
            "xyz_mean": np.zeros(3, dtype=np.float32),
            "xyz_std": np.ones(3, dtype=np.float32),
            "point_feature_mean": None,
            "point_feature_std": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "recovery.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "condition": "flow",
                        "flow_steps": 4,
                        "horizon": 2,
                        "hidden_dim": 32,
                        "explicit_flow_progress": True,
                        "action_frame": "world",
                        "flow_context_mode": "full",
                    },
                    "normalization": normalization,
                    "active_normalization": {
                        "mean": np.zeros(7, dtype=np.float32),
                        "std": np.ones(7, dtype=np.float32),
                        "steps_log_mean": 0.0,
                        "steps_log_std": 1.0,
                    },
                },
                checkpoint,
            )
            runtime = load_flow_action_runtime(checkpoint)
            self.assertIsInstance(runtime, RecoveryAwareFlowRuntime)
            state = np.zeros(20, dtype=np.float32)
            state[9] = 1.0
            state[19] = 0.0
            anchors = np.zeros((5, 3), dtype=np.float32)
            flow = np.zeros((5, 4, 3), dtype=np.float32)
            cloud = np.zeros((64, 3), dtype=np.float32)
            cloud[:, 0] = np.linspace(0.01, 0.1, 64)
            action = runtime.predict(
                operated_point_cloud=cloud,
                target_point_cloud=cloud,
                eef_state20=state,
                current_anchors=anchors,
                predicted_episode_flow=flow,
            )
            self.assertEqual(action.shape, (2, 14))
            np.testing.assert_allclose(action[:, :7], 0.0)
            np.testing.assert_allclose(action[:, 13], 1.0)
            self.assertEqual(runtime.last_stop_phase, "release")


if __name__ == "__main__":
    unittest.main()
