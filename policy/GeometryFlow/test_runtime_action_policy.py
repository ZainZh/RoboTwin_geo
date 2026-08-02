from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .runtime_action_policy import FlowActionRuntime
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


if __name__ == "__main__":
    unittest.main()
