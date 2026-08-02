from __future__ import annotations

import unittest

import numpy as np
import torch

from .train_recovery_aware_flow import (
    RecoveryAwareFlowPolicy,
    active_action7,
    loss_terms,
    reconstruct_action14,
)


class RecoveryAwareFlowTest(unittest.TestCase):
    def test_active_action_roundtrip(self):
        action = np.zeros((3, 2, 14), dtype=np.float32)
        action[0, :, :7] = 1.0
        action[1, :, 7:] = 2.0
        action[2, :, :7] = 3.0
        payload = {
            "action": action,
            "active_arm_right": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
        active = active_action7(payload, np.arange(3))
        rebuilt = reconstruct_action14(active, payload["active_arm_right"])
        np.testing.assert_allclose(rebuilt, action)

    def test_forward_shapes(self):
        model = RecoveryAwareFlowPolicy("flow", flow_steps=4, horizon=2, hidden_dim=32)
        batch = {
            "state": torch.zeros(3, 20),
            "flow_progress": torch.zeros(3, 1),
            "flow": torch.zeros(3, 5, 15),
            "flow_se3": torch.zeros(3, 4, 9),
        }
        output = model(batch)
        self.assertEqual(output["active_action"].shape, (3, 2, 7))
        self.assertEqual(output["stop_logits"].shape, (3, 3))
        self.assertEqual(output["steps_to_release_log"].shape, (3,))

    def test_loss_prefers_matching_output(self):
        batch = {
            "active_action": torch.zeros(2, 2, 7),
            "stop_phase": torch.tensor([0, 2]),
            "steps_to_release_log": torch.zeros(2),
            "is_recovery_sample": torch.tensor([0.0, 1.0]),
        }
        matching = {
            "active_action": torch.zeros(2, 2, 7),
            "stop_logits": torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
            "steps_to_release_log": torch.zeros(2),
        }
        wrong = {
            "active_action": torch.ones(2, 2, 7),
            "stop_logits": torch.zeros(2, 3),
            "steps_to_release_log": torch.ones(2),
        }
        kwargs = {
            "recovery_loss_weight": 2.0,
            "phase_class_weight": torch.ones(3),
            "phase_loss_weight": 0.2,
            "steps_loss_weight": 0.05,
        }
        self.assertLess(loss_terms(matching, batch, **kwargs)[0], loss_terms(wrong, batch, **kwargs)[0])


if __name__ == "__main__":
    unittest.main()
