from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from .deploy_policy import (
    _execute_relation_plan,
    eval as deploy_eval,
    gate_active_gripper_by_flow_progress,
    hold_active_gripper_closed,
    oracle_endpoint_correct_flow,
    reset_model,
    route_action_to_closed_gripper,
    temporal_ensemble_action,
)


class DeployPolicyTest(unittest.TestCase):
    def test_initialized_relation_plan_needs_only_proprioception(self):
        class Estimator:
            def desired_eef_pose7(
                self,
                target_frame,
                *,
                source_episode,
                clearance_m,
                gripper_symmetry_variant,
            ):
                del (
                    target_frame,
                    source_episode,
                    clearance_m,
                    gripper_symmetry_variant,
                )
                return np.asarray([0, 0, 0.8, 1, 0, 0, 0], dtype=np.float32)

        class Task:
            def __init__(self):
                self.take_action_cnt = 0
                self.step_lim = 20
                self.actions = []

            def take_action(self, action, *, action_type):
                self.actions.append((np.asarray(action), action_type))
                self.take_action_cnt += 1

        model = SimpleNamespace(
            action_decoder="relation_planner",
            predicted_flow=np.ones((4, 5, 3), dtype=np.float32),
            relation_target_frame=np.eye(4),
            relation_source_episode=3,
            relation_symmetry_variant=0,
            relation_plan_stage=0,
            relation_pre_clearance_m=0.1,
            relation_final_clearance_m=0.0,
            relation_closed_dwell_physics_steps=0,
            relation_release_steps=1,
            relation_post_release_retract_m=0.0,
            functional_flow_estimator=Estimator(),
            routed_arm="left",
            executed_actions=0,
            routed_action_count=0,
            progress_gate_closed_actions=0,
            progress_gate_released_actions=0,
            max_policy_steps=20,
        )
        observation = {
            "endpose": {
                "left_endpose": np.asarray([0, 0, 0.8, 1, 0, 0, 0]),
                "right_endpose": np.asarray([0, 0, 0.8, 1, 0, 0, 0]),
                "left_gripper": 0.0,
                "right_gripper": 1.0,
            }
        }
        task = Task()
        deploy_eval(task, model, observation)
        self.assertEqual(len(task.actions), 1)
        self.assertEqual(model.relation_plan_stage, 1)

    def test_relation_plan_approaches_places_then_releases(self):
        class Estimator:
            def desired_eef_pose7(
                self,
                target_frame,
                *,
                source_episode,
                clearance_m,
                gripper_symmetry_variant,
            ):
                del target_frame, source_episode, gripper_symmetry_variant
                return np.asarray(
                    [clearance_m, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                )

        class Task:
            def __init__(self):
                self.actions = []

            def take_action(self, action, *, action_type):
                self.actions.append((np.asarray(action), action_type))

        model = SimpleNamespace(
            relation_target_frame=np.eye(4),
            relation_source_episode=3,
            relation_symmetry_variant=0,
            relation_plan_stage=0,
            relation_pre_clearance_m=0.1,
            relation_final_clearance_m=0.012,
            relation_closed_dwell_physics_steps=0,
            relation_release_steps=1,
            relation_post_release_retract_m=0.0,
            functional_flow_estimator=Estimator(),
            routed_arm="right",
            executed_actions=0,
            routed_action_count=0,
            progress_gate_closed_actions=0,
            progress_gate_released_actions=0,
        )
        observation = {
            "endpose": {
                "left_endpose": np.asarray([0, 0, 0.8, 1, 0, 0, 0]),
                "right_endpose": np.asarray([0, 0, 0.8, 1, 0, 0, 0]),
                "left_gripper": 1.0,
                "right_gripper": 0.0,
            }
        }
        task = Task()
        _execute_relation_plan(task, model, observation)
        _execute_relation_plan(task, model, observation)
        _execute_relation_plan(task, model, observation)
        self.assertEqual([float(row[0][15]) for row in task.actions], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(float(task.actions[0][0][8]), 0.1)
        self.assertAlmostEqual(float(task.actions[1][0][8]), 0.012)
        self.assertEqual(model.relation_plan_stage, 3)
        self.assertEqual(model.progress_gate_closed_actions, 2)
        self.assertEqual(model.progress_gate_released_actions, 1)

    def test_relation_plan_supports_dwell_gradual_release_and_retract(self):
        class Estimator:
            def desired_eef_pose7(
                self,
                target_frame,
                *,
                source_episode,
                clearance_m,
                gripper_symmetry_variant,
            ):
                del target_frame, source_episode, gripper_symmetry_variant
                return np.asarray(
                    [clearance_m, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                )

        class Scene:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

        class Task:
            def __init__(self):
                self.actions = []
                self.take_action_cnt = 0
                self.scene = Scene()
                self.success_checks = 0

            def take_action(self, action, *, action_type):
                self.actions.append((np.asarray(action), action_type))

            def _update_render(self):
                pass

            def check_success(self):
                self.success_checks += 1
                return False

        model = SimpleNamespace(
            relation_target_frame=np.eye(4),
            relation_source_episode=3,
            relation_symmetry_variant=0,
            relation_plan_stage=0,
            relation_pre_clearance_m=0.1,
            relation_final_clearance_m=0.0,
            relation_closed_dwell_physics_steps=4,
            relation_release_steps=3,
            relation_post_release_retract_m=0.02,
            relation_settle_physics_steps=2,
            functional_flow_estimator=Estimator(),
            routed_arm="right",
            executed_actions=0,
            routed_action_count=0,
            progress_gate_closed_actions=0,
            progress_gate_released_actions=0,
        )
        observation = {
            "endpose": {
                "left_endpose": np.asarray([0, 0, 0.8, 1, 0, 0, 0]),
                "right_endpose": np.asarray([0, 0, 0.8, 1, 0, 0, 0]),
                "left_gripper": 1.0,
                "right_gripper": 0.0,
            }
        }
        task = Task()
        for _ in range(7):
            _execute_relation_plan(task, model, observation)
        self.assertEqual(task.scene.steps, 4)
        self.assertEqual(task.success_checks, 4)
        self.assertEqual(
            [round(float(row[0][15]), 4) for row in task.actions],
            [0.0, 0.0, 0.3333, 0.6667, 1.0, 1.0],
        )
        self.assertAlmostEqual(float(task.actions[-1][0][8]), 0.02)
        self.assertEqual(model.relation_plan_stage, 7)
        self.assertEqual(model.progress_gate_closed_actions, 5)
        self.assertEqual(model.progress_gate_released_actions, 2)

    def test_oracle_endpoint_correction_hits_privileged_rigid_goal(self):
        generator = np.random.default_rng(5)
        anchors = generator.normal(size=(12, 3))
        flow = np.stack(
            [anchors + np.array([0.02 * step, 0.0, 0.0]) for step in range(5)],
            axis=1,
        )
        angle = np.deg2rad(35.0)
        goal = np.eye(4)
        goal[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        goal[:3, 3] = [0.04, -0.03, 0.02]
        corrected = oracle_endpoint_correct_flow(anchors, flow, goal)
        expected = anchors @ goal[:3, :3].T + goal[:3, 3]
        np.testing.assert_allclose(corrected[:, 0], anchors, atol=1e-6)
        np.testing.assert_allclose(corrected[:, -1], expected, atol=1e-6)

    def test_flow_progress_gate_holds_then_releases_active_gripper(self):
        action = np.full(14, 0.25, dtype=np.float32)
        held, state = gate_active_gripper_by_flow_progress(action, "right", 0.80, 0.85)
        self.assertEqual(state, "closed")
        self.assertEqual(float(held[13]), 0.0)
        np.testing.assert_allclose(held[:13], action[:13])

        released, state = gate_active_gripper_by_flow_progress(
            action, "right", 0.85, 0.85
        )
        self.assertEqual(state, "released")
        self.assertEqual(float(released[13]), 1.0)

        disabled, state = gate_active_gripper_by_flow_progress(
            action, "right", 0.0, -1.0
        )
        self.assertIsNone(state)
        np.testing.assert_allclose(disabled, action)

    def test_reset_clears_episode_state(self):
        model = SimpleNamespace(
            initial_operated_xyz=np.ones((3, 3)),
            predicted_flow=np.ones((4, 5, 3)),
            initial_anchors=np.ones((4, 3)),
            flow_diagnostics={"source_episode": 2},
            action_chunks=4,
            pending_action_chunks=[(0, np.ones((2, 14)))],
            temporal_ensemble_candidates=9,
            routed_action_count=8,
            routed_arm="right",
            executed_actions=7,
            predicted_translation_norm_sum=1.0,
            predicted_translation_norm_max=0.2,
            predicted_rotation_norm_sum=2.0,
            predicted_rotation_norm_max=0.3,
        )
        reset_model(model)
        self.assertIsNone(model.initial_operated_xyz)
        self.assertIsNone(model.predicted_flow)
        self.assertIsNone(model.initial_anchors)
        self.assertEqual(model.flow_diagnostics, {})
        self.assertEqual(model.action_chunks, 0)
        self.assertEqual(model.pending_action_chunks, [])
        self.assertEqual(model.temporal_ensemble_candidates, 0)
        self.assertEqual(model.routed_action_count, 0)
        self.assertIsNone(model.routed_arm)
        self.assertEqual(model.executed_actions, 0)
        self.assertEqual(model.predicted_translation_norm_sum, 0.0)
        self.assertEqual(model.predicted_translation_norm_max, 0.0)
        self.assertEqual(model.predicted_rotation_norm_sum, 0.0)
        self.assertEqual(model.predicted_rotation_norm_max, 0.0)

    def test_temporal_ensemble_uses_overlapping_predictions(self):
        old = np.zeros((3, 14), dtype=np.float32)
        old[1] = 2.0
        recent = np.zeros((3, 14), dtype=np.float32)
        recent[0] = 4.0
        action, count = temporal_ensemble_action([(0, old), (1, recent)], 1, decay=0.0)
        np.testing.assert_allclose(action, 3.0)
        self.assertEqual(count, 2)

    def test_temporal_ensemble_prefers_recent_prediction(self):
        old = np.zeros((3, 14), dtype=np.float32)
        old[1] = 0.0
        recent = np.zeros((3, 14), dtype=np.float32)
        recent[0] = 1.0
        action, _ = temporal_ensemble_action([(0, old), (1, recent)], 1, decay=1.0)
        self.assertGreater(float(action[0]), 0.7)

    def test_route_action_keeps_only_closed_arm_motion(self):
        action = np.arange(14, dtype=np.float32)
        routed, arm = route_action_to_closed_gripper(
            action, {"left_gripper": 1.0, "right_gripper": 0.0}
        )
        np.testing.assert_allclose(routed[:6], 0.0)
        self.assertEqual(float(routed[6]), 1.0)
        np.testing.assert_allclose(routed[7:], action[7:])
        self.assertEqual(arm, "right")

    def test_route_action_is_noop_when_grippers_are_ambiguous(self):
        action = np.arange(14, dtype=np.float32)
        routed, arm = route_action_to_closed_gripper(
            action, {"left_gripper": 1.0, "right_gripper": 1.0}
        )
        np.testing.assert_allclose(routed, action)
        self.assertIsNone(arm)

    def test_route_action_does_not_switch_a_latched_arm(self):
        action = np.arange(14, dtype=np.float32)
        routed, arm = route_action_to_closed_gripper(
            action,
            {"left_gripper": 0.0, "right_gripper": 1.0},
            latched_arm="right",
        )
        np.testing.assert_allclose(routed[:6], 0.0)
        np.testing.assert_allclose(routed[7:], action[7:])
        self.assertEqual(float(routed[6]), 0.0)
        self.assertEqual(arm, "right")

    def test_route_action_rejects_invalid_latch(self):
        with self.assertRaisesRegex(ValueError, "invalid latched active arm"):
            route_action_to_closed_gripper(
                np.zeros(14, dtype=np.float32),
                {"left_gripper": 0.0, "right_gripper": 1.0},
                latched_arm="middle",
            )

    def test_hold_active_gripper_closed_preserves_motion(self):
        action = np.arange(14, dtype=np.float32)
        held = hold_active_gripper_closed(action, "right")
        np.testing.assert_allclose(held[:13], action[:13])
        self.assertEqual(float(held[13]), 0.0)


if __name__ == "__main__":
    unittest.main()
