from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .deploy_tagrt_dp3 import (
    _apply_rotation_recovery_cooldown,
    _apply_rotation_recovery_persistence,
    _frame9_rotation_angle_deg,
    _frame9_rotation_rotvec,
    _goal_frame9,
    _path_list,
    _policy_uses_anchor_flow,
)
from diffusion_policy_3d.policy.dp3 import _se3_frame_invariants


class TagrtDP3DeployHelpersTest(unittest.TestCase):
    def test_anchor_flow_is_emitted_for_continuous_or_diagnostic_consumer(self):
        continuous = SimpleNamespace(
            geometry_key="tagrt_anchor_flow",
            rotation_action_translation_anchor_flow_key=None,
        )
        diagnostic = SimpleNamespace(
            geometry_key=None,
            rotation_action_translation_anchor_flow_key="tagrt_anchor_flow",
        )
        baseline = SimpleNamespace(
            geometry_key="tagrt_local",
            rotation_action_translation_anchor_flow_key=None,
        )
        self.assertTrue(_policy_uses_anchor_flow(continuous))
        self.assertTrue(_policy_uses_anchor_flow(diagnostic))
        self.assertFalse(_policy_uses_anchor_flow(baseline))

    def test_rotation_recovery_cooldown_restores_nominal_motion(self):
        recovery = torch.full((1, 2, 14), 3.0)
        nominal = torch.full((1, 2, 14), 2.0)
        prediction = {
            "action": recovery,
            "action_before_rotation_recovery": nominal,
            "rotation_action_head_active": torch.tensor([[False, True]]),
        }
        remaining, requested, suppressed = _apply_rotation_recovery_cooldown(
            prediction, cooldown_remaining=2, cooldown_chunks=2
        )
        self.assertEqual(remaining, 1)
        np.testing.assert_array_equal(requested, [False, True])
        self.assertTrue(suppressed)
        torch.testing.assert_close(prediction["action"], nominal)
        self.assertFalse(bool(prediction["rotation_action_head_active"].any()))

    def test_recovery_cooldown_uses_combined_translation_request(self):
        nominal = torch.full((1, 2, 14), 2.0)
        prediction = {
            "action": torch.full((1, 2, 14), 3.0),
            "action_before_rotation_recovery": nominal,
            "rotation_action_head_active": torch.tensor([[False, False]]),
            "translation_recovery_head_active": torch.tensor([[True, False]]),
            "recovery_action_head_active": torch.tensor([[True, False]]),
        }
        remaining, requested, suppressed = _apply_rotation_recovery_cooldown(
            prediction, cooldown_remaining=1, cooldown_chunks=2
        )
        self.assertEqual(remaining, 0)
        np.testing.assert_array_equal(requested, [True, False])
        self.assertTrue(suppressed)
        torch.testing.assert_close(prediction["action"], nominal)
        self.assertFalse(bool(prediction["recovery_action_head_active"].any()))

    def test_rotation_recovery_cooldown_arms_after_execution(self):
        recovery = torch.full((1, 2, 14), 3.0)
        prediction = {
            "action": recovery,
            "action_before_rotation_recovery": torch.full((1, 2, 14), 2.0),
            "rotation_action_head_active": torch.tensor([[True, False]]),
        }
        remaining, requested, suppressed = _apply_rotation_recovery_cooldown(
            prediction, cooldown_remaining=0, cooldown_chunks=3
        )
        self.assertEqual(remaining, 3)
        np.testing.assert_array_equal(requested, [True, False])
        self.assertFalse(suppressed)
        torch.testing.assert_close(prediction["action"], recovery)

    def test_rotation_recovery_persistence_requires_consecutive_requests(self):
        recovery = torch.full((1, 2, 14), 3.0)
        nominal = torch.full((1, 2, 14), 2.0)
        prediction = {
            "action": recovery,
            "action_before_rotation_recovery": nominal,
            "rotation_action_head_active": torch.tensor([[False, True]]),
        }
        streak, requested, suppressed = _apply_rotation_recovery_persistence(
            prediction, requested_streak=0, min_consecutive_chunks=2
        )
        self.assertEqual(streak, 1)
        np.testing.assert_array_equal(requested, [False, True])
        self.assertTrue(suppressed)
        torch.testing.assert_close(prediction["action"], nominal)

        prediction["action"] = recovery
        prediction["rotation_action_head_active"] = torch.tensor([[False, True]])
        streak, _requested, suppressed = _apply_rotation_recovery_persistence(
            prediction, requested_streak=streak, min_consecutive_chunks=2
        )
        self.assertEqual(streak, 2)
        self.assertFalse(suppressed)
        torch.testing.assert_close(prediction["action"], recovery)

    def test_goal_frame_columns(self):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            _goal_frame9(transform),
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

    def test_path_list(self):
        self.assertEqual(_path_list("a,b"), ["a", "b"])
        self.assertEqual(_path_list(["a", "b"]), ["a", "b"])
        self.assertEqual(_path_list(""), [])
        self.assertEqual(_path_list(None), [])

    def test_frame9_rotation_angle(self):
        identity = np.concatenate(([0.0, 0.0, 0.0], np.eye(3)[:, 0], np.eye(3)[:, 1]))
        self.assertAlmostEqual(_frame9_rotation_angle_deg(identity), 0.0)
        angle = np.deg2rad(90.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        frame = np.concatenate(([0.0, 0.0, 0.0], rotation[:, 0], rotation[:, 1]))
        self.assertAlmostEqual(_frame9_rotation_angle_deg(frame), 90.0)
        np.testing.assert_allclose(
            _frame9_rotation_rotvec(frame), [0.0, 0.0, angle], atol=1.0e-7
        )

    def test_frame9_rotation_rotvec_preserves_sign(self):
        positive = Rotation.from_euler("y", 35.0, degrees=True).as_matrix()
        negative = Rotation.from_euler("y", -35.0, degrees=True).as_matrix()
        positive_frame = np.concatenate(
            ([0.0, 0.0, 0.0], positive[:, 0], positive[:, 1])
        )
        negative_frame = np.concatenate(
            ([0.0, 0.0, 0.0], negative[:, 0], negative[:, 1])
        )
        self.assertGreater(_frame9_rotation_rotvec(positive_frame)[1], 0.0)
        self.assertLess(_frame9_rotation_rotvec(negative_frame)[1], 0.0)

    def test_se3_retention_features_discard_axis_direction(self):
        frames = []
        for axis, translation in (("x", [0.03, 0.04, 0.0]), ("y", [0.0, 0.03, 0.04])):
            rotation = Rotation.from_euler(axis, 30.0, degrees=True).as_matrix()
            frames.append(
                np.concatenate((translation, rotation[:, 0], rotation[:, 1]))
            )
        invariants = _se3_frame_invariants(
            torch.as_tensor(np.asarray(frames), dtype=torch.float32)
        ).numpy()
        np.testing.assert_allclose(invariants[0], invariants[1], atol=1.0e-6)
        self.assertAlmostEqual(float(invariants[0, 0]), 0.05, places=6)
        self.assertAlmostEqual(
            float(np.rad2deg(invariants[0, 1])), 30.0, places=3
        )


if __name__ == "__main__":
    unittest.main()
