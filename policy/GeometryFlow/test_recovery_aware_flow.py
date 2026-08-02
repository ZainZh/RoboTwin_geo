from __future__ import annotations

import unittest

import numpy as np
import torch

from .train_recovery_aware_flow import (
    RecoveryAwareFlowPolicy,
    active_action7,
    deployment_flow_batch,
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

    def test_factorized_decoder_shapes(self):
        model = RecoveryAwareFlowPolicy(
            "flow",
            flow_steps=4,
            horizon=2,
            hidden_dim=32,
            action_decoder="factorized",
        )
        batch = {
            "state": torch.zeros(3, 20),
            "flow_progress": torch.zeros(3, 1),
            "flow": torch.zeros(3, 5, 15),
            "flow_se3": torch.zeros(3, 4, 9),
        }
        output = model(batch)
        self.assertEqual(output["active_action"].shape, (3, 2, 7))
        self.assertEqual(output["stop_logits"].shape, (3, 3))

    def test_goal_aligned_rotation_has_positive_flow_projection(self):
        model = RecoveryAwareFlowPolicy(
            "flow",
            flow_steps=4,
            horizon=2,
            hidden_dim=32,
            action_decoder="factorized",
            rotation_parameterization="goal_aligned",
            active_action_mean=np.zeros(7, dtype=np.float32),
            active_action_std=np.ones(7, dtype=np.float32),
        ).eval()
        goal = torch.tensor([[0.2, -0.1, 0.3], [-0.2, 0.1, 0.15]])
        batch = {
            "state": torch.zeros(2, 20),
            "flow_progress": torch.zeros(2, 1),
            "flow": torch.zeros(2, 5, 15),
            "flow_se3": torch.zeros(2, 4, 9),
            "flow_goal_rotvec": goal,
        }
        rotation = model(batch)["active_action"][..., 3:6]
        projection = torch.sum(rotation * goal[:, None], dim=-1)
        self.assertTrue(torch.all(projection > 0.0))

    def test_factorized_translation_is_invariant_to_rotation_flow(self):
        torch.manual_seed(17)
        model = RecoveryAwareFlowPolicy(
            "flow",
            flow_steps=4,
            horizon=2,
            hidden_dim=32,
            action_decoder="factorized",
        ).eval()
        batch = {
            "state": torch.randn(2, 20),
            "flow_progress": torch.rand(2, 1),
            "flow": torch.randn(2, 6, 15),
            "flow_se3": torch.randn(2, 4, 9),
        }
        changed = {key: value.clone() for key, value in batch.items()}
        displacement = changed["flow"][..., 3:].reshape(2, 6, 4, 3)
        rotation_only_change = torch.randn_like(displacement)
        rotation_only_change -= rotation_only_change.mean(dim=1, keepdim=True)
        displacement += rotation_only_change
        changed["flow"][..., 3:] = displacement.flatten(start_dim=-2)
        changed["flow_se3"][..., 3:] = torch.randn_like(
            changed["flow_se3"][..., 3:]
        )
        changed["flow_progress"] = torch.ones_like(changed["flow_progress"])

        original_action = model(batch)["active_action"]
        changed_action = model(changed)["active_action"]
        torch.testing.assert_close(
            original_action[..., :3], changed_action[..., :3], atol=1e-6, rtol=1e-6
        )
        self.assertFalse(
            torch.allclose(original_action[..., 3:6], changed_action[..., 3:6])
        )

    def test_deployment_flow_batch_replaces_only_flow_inputs(self):
        batch = {
            "state": torch.randn(2, 20),
            "flow": torch.zeros(2, 5, 15),
            "flow_se3": torch.zeros(2, 4, 9),
            "flow_progress": torch.zeros(2, 1),
            "flow_goal_rotvec": torch.zeros(2, 3),
            "deployment_flow": torch.ones(2, 5, 15),
            "deployment_flow_se3": torch.ones(2, 4, 9) * 2.0,
            "deployment_flow_progress": torch.ones(2, 1) * 3.0,
            "deployment_flow_goal_rotvec": torch.ones(2, 3) * 4.0,
            "active_action": torch.randn(2, 2, 7),
        }
        routed = deployment_flow_batch(batch)
        self.assertIs(routed["state"], batch["state"])
        self.assertIs(routed["active_action"], batch["active_action"])
        self.assertIs(routed["flow"], batch["deployment_flow"])
        self.assertIs(routed["flow_se3"], batch["deployment_flow_se3"])
        self.assertIs(routed["flow_progress"], batch["deployment_flow_progress"])
        self.assertIs(
            routed["flow_goal_rotvec"], batch["deployment_flow_goal_rotvec"]
        )

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
