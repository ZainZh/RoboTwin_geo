from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .build_task_flow_dataset import (
    collapse_static_active_eef_frames,
    deterministic_sample,
    eef_delta_action14,
    object_goal_error_labels,
    pose7_wxyz_to_matrix,
    resample_indices,
    transform_points,
)
from .augment_task_flow_recovery import (
    augment_recovery_sample,
    centered_transform,
    functional_frame9_transform,
    intended_goal_object_pose9,
    pose9_transform,
    sample_failure_matched_rotation,
    sample_uniform_magnitude_vector,
    translation_for_remaining_functional_relation,
)
from .train_task_flow_benchmark import (
    active_arm_routing_loss,
    build_action_predictor,
    fit_normalization,
    geometric_flow_progress,
    noisy_rigid_flow,
    remaining_flow_from_current,
    rigid_flow_se3_tokens,
    weighted_action_loss,
)
from .target_frame import estimate_cached_geometry_marker_frame


def pose(position, rotvec):
    x, y, z, w = Rotation.from_rotvec(rotvec).as_quat()
    return np.asarray([*position, w, x, y, z], dtype=np.float64)


class TaskFlowTest(unittest.TestCase):
    def test_remaining_relation_translation_inverse(self):
        current_position = np.asarray([0.12, -0.04, 0.31])
        goal_position = np.asarray([0.17, 0.02, 0.34])
        identity = Rotation.identity()
        current_frame = np.concatenate(
            (current_position, identity.as_matrix()[:, 0], identity.as_matrix()[:, 1])
        )
        sample = {
            "current_object_pose9": current_frame,
            "target_frame9": np.concatenate(
                (
                    goal_position - current_position,
                    identity.as_matrix()[:, 0],
                    identity.as_matrix()[:, 1],
                )
            ),
            "goal_frame9": np.concatenate(
                (goal_position, identity.as_matrix()[:, 0], identity.as_matrix()[:, 1])
            ),
        }
        desired = np.asarray([-0.03, 0.04, 0.02])
        perturbation = translation_for_remaining_functional_relation(
            sample, identity, desired
        )
        np.testing.assert_allclose(
            goal_position - (current_position + perturbation), desired, atol=1e-7
        )

    def test_uniform_magnitude_vector_respects_requested_shell(self):
        generator = np.random.default_rng(53)
        norms = [
            np.linalg.norm(
                sample_uniform_magnitude_vector(generator, 0.03, 0.08)
            )
            for _ in range(100)
        ]
        self.assertGreaterEqual(min(norms), 0.03)
        self.assertLessEqual(max(norms), 0.08)

    def test_failure_matched_rotation_inverts_remaining_axis(self):
        mode = {
            "signed_world_axis": [-1.0, 0.0, 0.0],
            "rotation_magnitude_deg_p25": 35.0,
            "rotation_magnitude_deg_p75": 35.0,
        }
        perturbation, remaining_axis, magnitude_deg = (
            sample_failure_matched_rotation(
                np.random.default_rng(7), mode, axis_jitter_deg=0.0
            )
        )
        np.testing.assert_allclose(remaining_axis, [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            perturbation.as_rotvec(), [np.deg2rad(35.0), 0.0, 0.0]
        )
        self.assertAlmostEqual(magnitude_deg, 35.0)

    def test_static_eef_collapse_removes_duplicate_prefix_without_losing_motion(self):
        left = np.stack(
            (
                pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                pose([0.01, 0.0, 0.0], [0.0, 0.0, 0.1]),
                pose([0.02, 0.0, 0.0], [0.0, 0.0, 0.2]),
            )
        )
        right = np.repeat(pose([0.5, 0.0, 0.0], [0.0, 0.0, 0.0])[None], 4, axis=0)
        closed = np.zeros(4)
        opened = np.ones(4)
        indices = collapse_static_active_eef_frames(
            left, closed, right, opened, "left"
        )
        np.testing.assert_array_equal(indices, [0, 2, 3])

    def test_cached_marker_frame_fuses_sparse_early_observations(self):
        generator = np.random.default_rng(41)
        frames = []
        for _ in range(3):
            cloud = np.zeros((64, 6), dtype=np.float32)
            cloud[:, :3] = generator.normal(size=(64, 3)) * [0.04, 0.02, 0.002]
            cloud[:, 3:6] = [0.1, 0.2, 0.8]
            cloud[:6, :3] = np.asarray(
                [[x, 0.0, 0.006] for x in np.linspace(-0.04, 0.04, 6)]
            )
            cloud[:6, 3:6] = [0.9, 0.8, 0.0]
            frames.append(cloud)
        transform, diagnostic = estimate_cached_geometry_marker_frame(
            np.asarray(frames)
        )
        self.assertEqual(diagnostic["fused_frames"], 3)
        self.assertEqual(diagnostic["marker_points"], 18)
        self.assertFalse(diagnostic["sparse_fallback"])
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
        )

    def test_cached_marker_frame_reports_zero_confidence_fallback(self):
        generator = np.random.default_rng(43)
        cloud = np.zeros((2, 64, 6), dtype=np.float32)
        cloud[..., :3] = (
            generator.normal(size=(2, 64, 3)) * [0.04, 0.02, 0.002]
        )
        cloud[..., 3:6] = [0.1, 0.2, 0.8]
        transform, diagnostic = estimate_cached_geometry_marker_frame(cloud)
        self.assertEqual(diagnostic["marker_points"], 0)
        self.assertTrue(diagnostic["sparse_fallback"])
        self.assertTrue(np.isfinite(transform).all())
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
        )

    def test_object_goal_error_labels_follow_task_transform_convention(self):
        object_a = pose([0.25, -0.1, 0.72], [0.0, 0.0, 0.3])
        object_b = pose([0.1, -0.2, 0.70], [0.0, 0.0, -0.2])
        desired_a = pose([0.2, 0.0, 0.73], [0.0, 0.0, 0.1])
        world_from_b = pose7_wxyz_to_matrix(object_b)
        desired_world_from_a = pose7_wxyz_to_matrix(desired_a)
        goal = np.linalg.inv(desired_world_from_a) @ world_from_b
        translation, rotation, aligned = object_goal_error_labels(
            object_a, object_b, goal
        )
        np.testing.assert_allclose(
            translation, desired_a[:3] - object_a[:3], atol=1e-7
        )
        np.testing.assert_allclose(rotation, [0.0, 0.0, -0.2], atol=1e-7)
        self.assertEqual(aligned, 0.0)
    def test_active_arm_routing_loss_prefers_correct_arm(self):
        import torch

        correct = torch.zeros((2, 3, 14))
        correct[0, :, 0] = 2.0
        correct[1, :, 7] = 2.0
        wrong = correct.roll(7, dims=-1)
        active_right = torch.tensor([0.0, 1.0])
        self.assertLess(
            float(active_arm_routing_loss(correct, active_right)),
            float(active_arm_routing_loss(wrong, active_right)),
        )

    def test_fit_normalization_returns_statistics(self):
        generator = np.random.default_rng(1)
        payload = {
            "state": generator.normal(size=(3, 20)).astype(np.float32),
            "action": generator.normal(size=(3, 2, 14)).astype(np.float32),
            "points_a": generator.normal(size=(3, 5, 3)).astype(np.float32),
            "points_b": generator.normal(size=(3, 5, 3)).astype(np.float32),
        }
        normalization = fit_normalization(payload, np.arange(3))
        self.assertEqual(normalization.state_mean.shape, (20,))
        self.assertEqual(normalization.action_std.shape, (14,))
        self.assertEqual(normalization.xyz_mean.shape, (3,))

    def test_priority_sampling_retains_sparse_target_marker(self):
        generator = np.random.default_rng(2)
        cloud = generator.normal(size=(128, 6)).astype(np.float32)
        cloud[:, 3:6] = [0.2, 0.3, 0.8]
        cloud[:12, 3:6] = [0.95, 0.8, 0.05]
        sampled = deterministic_sample(
            cloud,
            count=32,
            seed=7,
            point_channels=6,
            priority_color=True,
        )
        marker = sampled[:, 3] + sampled[:, 4] - 2.0 * sampled[:, 5] > 0.65
        self.assertEqual(int(marker.sum()), 12)

    def test_recovery_augmentation_preserves_grasp_rigidity_and_rejoins_future(self):
        horizon = 4
        object_pose = pose([0.2, -0.1, 0.7], [0.0, 0.0, 0.2])
        active_pose = pose([0.25, -0.05, 0.8], [0.1, 0.0, 0.2])
        inactive_pose = pose([-0.3, 0.0, 0.8], [0.0, 0.0, 0.0])
        points = np.asarray([[0.2, -0.1, 0.7], [0.25, -0.1, 0.7]], dtype=np.float32)
        future = []
        for step in range(horizon):
            future.append(
                np.stack(
                    (
                        inactive_pose,
                        pose([0.26 + 0.01 * step, -0.04, 0.81], [0.1, 0.0, 0.2]),
                    )
                )
            )
        future = np.asarray(future, dtype=np.float32)
        state = np.zeros(20, dtype=np.float32)
        state[9] = 1.0
        state[19] = 0.0
        object_matrix = pose7_wxyz_to_matrix(object_pose)
        sample = {
            "points_a": points,
            "points_b": points.copy(),
            "state": state,
            "action": np.zeros((horizon, 14), dtype=np.float32),
            "current_anchors": points.copy(),
            "current_object_pose9": np.concatenate(
                (object_pose[:3], object_matrix[:3, :3][:, :2].reshape(6))
            ).astype(np.float32),
            "current_eef_pose7": np.stack((inactive_pose, active_pose)).astype(np.float32),
            "future_eef_pose7": future,
            "active_arm_right": np.asarray(1.0, dtype=np.float32),
            "relation_phase": np.asarray(1.0, dtype=np.float32),
        }
        translation = np.asarray([0.01, -0.02, 0.015])
        rotation = Rotation.from_euler("z", 12, degrees=True)
        augmented = augment_recovery_sample(
            sample, translation=translation, rotation=rotation
        )
        before = active_pose[:3] - object_pose[:3]
        after = (
            augmented["current_eef_pose7"][1, :3]
            - augmented["current_object_pose9"][:3]
        )
        np.testing.assert_allclose(after, rotation.apply(before), atol=2e-6)
        np.testing.assert_allclose(
            augmented["future_eef_pose7"][-1, 1], future[-1, 1], atol=2e-6
        )
        self.assertGreater(np.linalg.norm(augmented["action"][0, 7:10]), 0.0)

    def test_recovery_augmentation_preserves_non_xyz_point_features(self):
        center = np.asarray([0.2, -0.1, 0.7])
        xyz = np.asarray([[0.2, -0.1, 0.7], [0.25, -0.1, 0.7]])
        rgb = np.asarray([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]])
        points = np.concatenate((xyz, rgb), axis=-1)
        transformed = centered_transform(
            points,
            center,
            Rotation.from_euler("z", 20.0, degrees=True),
            np.asarray([0.01, -0.02, 0.005]),
        )
        np.testing.assert_allclose(transformed[:, 3:], rgb, atol=1e-7)
        self.assertFalse(np.allclose(transformed[:, :3], xyz))

    def test_geometry_goal_recovery_moves_eef_with_object_transform(self):
        horizon = 4
        object_pose = pose([0.2, -0.1, 0.7], [0.0, 0.0, 0.2])
        active_pose = pose([0.25, -0.05, 0.8], [0.1, 0.0, 0.2])
        inactive_pose = pose([-0.3, 0.0, 0.8], [0.0, 0.0, 0.0])
        points = np.asarray([[0.2, -0.1, 0.7], [0.25, -0.1, 0.7]], dtype=np.float32)
        state = np.zeros(20, dtype=np.float32)
        state[9] = 1.0
        state[19] = 0.0
        object_matrix = pose7_wxyz_to_matrix(object_pose)
        goal_pose = pose([0.3, 0.0, 0.75], [0.0, 0.0, 0.6])
        goal_matrix = pose7_wxyz_to_matrix(goal_pose)
        sample = {
            "points_a": points,
            "points_b": points.copy(),
            "state": state,
            "action": np.zeros((horizon, 14), dtype=np.float32),
            "current_anchors": points.copy(),
            "current_object_pose9": np.concatenate(
                (object_pose[:3], object_matrix[:3, :3][:, :2].reshape(6))
            ).astype(np.float32),
            "current_eef_pose7": np.stack((inactive_pose, active_pose)).astype(np.float32),
            "future_eef_pose7": np.repeat(
                np.stack((inactive_pose, active_pose))[None], horizon, axis=0
            ).astype(np.float32),
            "active_arm_right": np.asarray(1.0, dtype=np.float32),
            "relation_phase": np.asarray(1.0, dtype=np.float32),
        }
        augmented = augment_recovery_sample(
            sample,
            translation=np.zeros(3),
            rotation=Rotation.identity(),
            goal_object_pose9=np.concatenate(
                (goal_pose[:3], goal_matrix[:3, :3][:, :2].reshape(6))
            ),
        )
        delta = goal_matrix @ np.linalg.inv(object_matrix)
        expected_eef = delta @ pose7_wxyz_to_matrix(active_pose)
        actual_eef = pose7_wxyz_to_matrix(augmented["future_eef_pose7"][-1, 1])
        np.testing.assert_allclose(actual_eef, expected_eef, atol=2e-6)
        self.assertTrue(np.allclose(augmented["action"][:, 13], 0.0))

    def test_geometry_goal_recovery_can_release_only_at_endpoint(self):
        horizon = 4
        object_pose = pose([0.2, -0.1, 0.7], [0.0, 0.0, 0.2])
        active_pose = pose([0.25, -0.05, 0.8], [0.1, 0.0, 0.2])
        inactive_pose = pose([-0.3, 0.0, 0.8], [0.0, 0.0, 0.0])
        object_matrix = pose7_wxyz_to_matrix(object_pose)
        goal_pose = pose([0.3, 0.0, 0.75], [0.0, 0.0, 0.6])
        goal_matrix = pose7_wxyz_to_matrix(goal_pose)
        state = np.zeros(20, dtype=np.float32)
        state[9] = 1.0
        state[19] = 0.0
        points = np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32)
        sample = {
            "points_a": points,
            "points_b": points.copy(),
            "state": state,
            "action": np.zeros((horizon, 14), dtype=np.float32),
            "current_anchors": points.copy(),
            "current_object_pose9": np.concatenate(
                (object_pose[:3], object_matrix[:3, :3][:, :2].reshape(6))
            ).astype(np.float32),
            "current_eef_pose7": np.stack((inactive_pose, active_pose)).astype(np.float32),
            "future_eef_pose7": np.repeat(
                np.stack((inactive_pose, active_pose))[None], horizon, axis=0
            ).astype(np.float32),
            "active_arm_right": np.asarray(1.0, dtype=np.float32),
            "relation_phase": np.asarray(1.0, dtype=np.float32),
        }
        augmented = augment_recovery_sample(
            sample,
            translation=np.zeros(3),
            rotation=Rotation.identity(),
            goal_object_pose9=np.concatenate(
                (goal_pose[:3], goal_matrix[:3, :3][:, :2].reshape(6))
            ),
            release_at_goal=True,
        )

        np.testing.assert_allclose(augmented["action"][:-1, 13], 0.0)
        self.assertEqual(float(augmented["action"][-1, 13]), 1.0)
        np.testing.assert_allclose(augmented["action"][:, 6], 1.0)

    def test_terminal_recovery_can_label_immediate_release(self):
        horizon = 3
        object_pose = pose([0.2, -0.1, 0.7], [0.0, 0.0, 0.2])
        active_pose = pose([0.25, -0.05, 0.8], [0.1, 0.0, 0.2])
        inactive_pose = pose([-0.3, 0.0, 0.8], [0.0, 0.0, 0.0])
        object_matrix = pose7_wxyz_to_matrix(object_pose)
        state = np.zeros(20, dtype=np.float32)
        state[9] = 1.0
        state[19] = 0.0
        points = np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32)
        sample = {
            "points_a": points,
            "points_b": points.copy(),
            "state": state,
            "action": np.zeros((horizon, 14), dtype=np.float32),
            "current_anchors": points.copy(),
            "current_object_pose9": np.concatenate(
                (object_pose[:3], object_matrix[:3, :3][:, :2].reshape(6))
            ).astype(np.float32),
            "current_eef_pose7": np.stack((inactive_pose, active_pose)).astype(np.float32),
            "future_eef_pose7": np.repeat(
                np.stack((inactive_pose, active_pose))[None], horizon, axis=0
            ).astype(np.float32),
            "active_arm_right": np.asarray(1.0, dtype=np.float32),
            "relation_phase": np.asarray(1.0, dtype=np.float32),
        }
        augmented = augment_recovery_sample(
            sample,
            translation=np.zeros(3),
            rotation=Rotation.identity(),
            goal_object_pose9=sample["current_object_pose9"],
            release_immediately=True,
        )

        np.testing.assert_allclose(augmented["action"][:, 13], 1.0)
        np.testing.assert_allclose(augmented["action"][:, 6], 1.0)

    def test_recovery_updates_camera_relative_frame_and_intended_goal(self):
        horizon = 3
        object_pose = pose([0.2, -0.1, 0.7], [0.0, 0.0, 0.2])
        active_pose = pose([0.25, -0.05, 0.8], [0.1, 0.0, 0.2])
        inactive_pose = pose([-0.3, 0.0, 0.8], [0.0, 0.0, 0.0])
        object_matrix = pose7_wxyz_to_matrix(object_pose)
        current_object9 = np.concatenate(
            (object_pose[:3], object_matrix[:3, :3][:, :2].reshape(6))
        ).astype(np.float32)
        goal_position = np.asarray([0.3, 0.0, 0.75])
        goal_rotation = Rotation.from_rotvec([0.0, 0.0, 0.6])
        current_functional_position = np.asarray([0.21, -0.09, 0.71])
        current_functional_rotation = Rotation.from_rotvec([0.0, 0.0, 0.25])
        relative_rotation = goal_rotation * current_functional_rotation.inv()
        state = np.zeros(20, dtype=np.float32)
        sample = {
            "points_a": np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32),
            "points_b": np.asarray([[0.3, 0.0, 0.75]], dtype=np.float32),
            "state": state,
            "action": np.zeros((horizon, 14), dtype=np.float32),
            "current_anchors": np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32),
            "current_object_pose9": current_object9,
            "current_eef_pose7": np.stack((inactive_pose, active_pose)).astype(np.float32),
            "future_eef_pose7": np.repeat(
                np.stack((inactive_pose, active_pose))[None], horizon, axis=0
            ).astype(np.float32),
            "active_arm_right": np.asarray(1.0, dtype=np.float32),
            "relation_phase": np.asarray(1.0, dtype=np.float32),
            "goal_translation_error_xyz_m": (goal_position - object_pose[:3]).astype(np.float32),
            "goal_rotation_error_rotvec": (
                goal_rotation * Rotation.from_rotvec([0.0, 0.0, 0.2]).inv()
            ).as_rotvec().astype(np.float32),
            "goal_frame9": np.concatenate(
                (
                    goal_position,
                    goal_rotation.as_matrix()[:, 0],
                    goal_rotation.as_matrix()[:, 1],
                )
            ).astype(np.float32),
            "target_frame9": np.concatenate(
                (
                    goal_position - current_functional_position,
                    relative_rotation.as_matrix()[:, 0],
                    relative_rotation.as_matrix()[:, 1],
                )
            ).astype(np.float32),
            "source_frame6_columns": np.concatenate(
                (
                    current_functional_rotation.as_matrix()[:, 0],
                    current_functional_rotation.as_matrix()[:, 1],
                )
            ).astype(np.float32),
        }
        perturbation = Rotation.from_euler("z", 20.0, degrees=True)
        translation = np.asarray([0.01, -0.02, 0.005])
        goal_object9 = intended_goal_object_pose9(sample)
        np.testing.assert_allclose(goal_object9[:3], goal_position, atol=1e-6)
        augmented = augment_recovery_sample(
            sample,
            translation=translation,
            rotation=perturbation,
            goal_object_pose9=goal_object9,
        )
        new_relative_position, new_relative_rotation = functional_frame9_transform(
            augmented["target_frame9"]
        )
        expected_current_position = (
            perturbation.apply(current_functional_position - object_pose[:3])
            + object_pose[:3]
            + translation
        )
        expected_current_rotation = perturbation * current_functional_rotation
        np.testing.assert_allclose(
            new_relative_position,
            goal_position - expected_current_position,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            new_relative_rotation.as_matrix(),
            (goal_rotation * expected_current_rotation.inv()).as_matrix(),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            pose9_transform(augmented["current_object_pose9"])[0]
            + augmented["goal_translation_error_xyz_m"],
            goal_position,
            atol=2e-6,
        )

    def test_resample_indices_include_endpoints(self):
        indices = resample_indices(101, 16)
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), 100)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_rigid_flow_preserves_pairwise_distance(self):
        generator = np.random.default_rng(3)
        points = generator.normal(size=(20, 3))
        first = transform_points(pose7_wxyz_to_matrix(pose([0, 0, 0], [0, 0, 0])), points)
        second = transform_points(
            pose7_wxyz_to_matrix(pose([0.3, -0.1, 0.2], [0.2, -0.4, 0.7])), points
        )
        first_distance = np.linalg.norm(first[:, None] - first[None], axis=-1)
        second_distance = np.linalg.norm(second[:, None] - second[None], axis=-1)
        np.testing.assert_allclose(first_distance, second_distance, atol=1e-7)

    def test_eef_delta_uses_world_frame_left_composition(self):
        current = pose([0.1, 0.2, 0.3], [0.1, -0.2, 0.3])
        world_delta = Rotation.from_rotvec([0.0, 0.0, 0.2])
        target_rotation = world_delta * Rotation.from_rotvec([0.1, -0.2, 0.3])
        x, y, z, w = target_rotation.as_quat()
        target = np.asarray([0.12, 0.18, 0.31, w, x, y, z])
        action = eef_delta_action14(current, 1.0, current, 1.0, target, 0.0, current, 1.0)
        np.testing.assert_allclose(action[:3], [0.02, -0.02, 0.01], atol=1e-7)
        np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.2], atol=1e-7)
        self.assertEqual(float(action[6]), 0.0)

    def test_flow_noise_preserves_rigidity_and_initial_step(self):
        generator = np.random.default_rng(4)
        initial = generator.normal(size=(8, 3)).astype(np.float32)
        flow = np.stack(
            [initial + (0.01 * step, 0.0, 0.0) for step in range(5)], axis=1
        )
        noisy = noisy_rigid_flow(
            flow,
            translation_std_m=0.01,
            rotation_std_deg=15.0,
            generator=np.random.default_rng(8),
        )
        np.testing.assert_allclose(noisy[:, 0], flow[:, 0], atol=1e-7)
        reference = np.linalg.norm(initial[:, None] - initial[None], axis=-1)
        for step in range(noisy.shape[1]):
            distance = np.linalg.norm(
                noisy[:, None, step] - noisy[None, :, step], axis=-1
            )
            np.testing.assert_allclose(distance, reference, atol=2e-6)

    def test_remaining_flow_starts_current_and_preserves_rigidity(self):
        generator = np.random.default_rng(21)
        initial = generator.normal(size=(12, 3))
        flow = np.stack(
            [
                initial @ Rotation.from_euler("z", step * 4, degrees=True).as_matrix().T
                + [0.01 * step, 0.0, 0.0]
                for step in range(8)
            ],
            axis=1,
        )
        current = flow[:, 4]
        remaining = remaining_flow_from_current(flow, current)
        np.testing.assert_allclose(remaining[:, 0], current, atol=1e-7)
        reference = np.linalg.norm(current[:, None] - current[None], axis=-1)
        for step in range(remaining.shape[1]):
            distance = np.linalg.norm(
                remaining[:, None, step] - remaining[None, :, step], axis=-1
            )
            np.testing.assert_allclose(distance, reference, atol=2e-6)

    def test_geometric_progress_finds_nearest_flow_step(self):
        initial = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
        flow = np.stack([initial + [0.1 * step, 0.0, 0.0] for step in range(5)], axis=1)
        self.assertAlmostEqual(geometric_flow_progress(flow, flow[:, 3]), 0.75)

    def test_rigid_flow_se3_tokens_recover_rotation(self):
        generator = np.random.default_rng(13)
        anchors = generator.normal(size=(20, 3))
        rotation = Rotation.from_euler("zyx", [35.0, -12.0, 8.0], degrees=True)
        target = rotation.apply(anchors) + [0.04, -0.03, 0.02]
        flow = np.stack((anchors, target), axis=1)
        token = rigid_flow_se3_tokens(flow, anchors, np.ones(3))
        np.testing.assert_allclose(token[0, :3], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            token[1, 3:].reshape(3, 2), rotation.as_matrix()[:, :2], atol=1e-6
        )

    def test_weighted_action_loss_emphasizes_active_arm(self):
        import torch

        target = torch.zeros((2, 1, 14))
        prediction = torch.zeros_like(target)
        prediction[0, 0, 0] = 1.0
        prediction[1, 0, 7] = 1.0
        active_right = torch.tensor([0.0, 1.0])
        phase = torch.zeros(2)
        base = weighted_action_loss(
            prediction,
            target,
            active_right,
            phase,
            active_arm_weight=1.0,
            relation_weight=1.0,
        )
        weighted = weighted_action_loss(
            prediction,
            target,
            active_right,
            phase,
            active_arm_weight=4.0,
            relation_weight=1.0,
        )
        self.assertGreater(float(weighted), float(base))

    def test_flow_transformer_predicts_action_chunk(self):
        import torch

        model = build_action_predictor("flow", 5, 3, "flow_transformer")
        batch = {
            "points_a": torch.randn(2, 7, 3),
            "points_b": torch.randn(2, 7, 3),
            "state": torch.randn(2, 20),
            "flow": torch.randn(2, 4, 18),
            "flow_progress": torch.rand(2, 1),
        }
        self.assertEqual(tuple(model(batch).shape), (2, 3, 14))

    def test_flow_bottleneck_has_no_point_cloud_action_path(self):
        import torch

        torch.manual_seed(11)
        model = build_action_predictor("flow", 5, 3, "flow_bottleneck", True)
        model.eval()
        batch = {
            "points_a": torch.randn(2, 7, 3),
            "points_b": torch.randn(2, 7, 3),
            "state": torch.randn(2, 20),
            "flow": torch.randn(2, 4, 18),
            "flow_progress": torch.rand(2, 1),
        }
        first = model(batch)
        batch["points_a"] = torch.randn_like(batch["points_a"]) * 100.0
        batch["points_b"] = torch.randn_like(batch["points_b"]) * 100.0
        second = model(batch)
        self.assertEqual(tuple(first.shape), (2, 3, 14))
        torch.testing.assert_close(first, second)

    def test_flow_se3_bottleneck_predicts_action_chunk(self):
        import torch

        model = build_action_predictor("flow", 5, 3, "flow_se3_bottleneck", True)
        batch = {
            "points_a": torch.randn(2, 7, 3),
            "points_b": torch.randn(2, 7, 3),
            "state": torch.randn(2, 20),
            "flow": torch.randn(2, 4, 18),
            "flow_se3": torch.randn(2, 5, 9),
            "flow_progress": torch.rand(2, 1),
        }
        self.assertEqual(tuple(model(batch).shape), (2, 3, 14))

    def test_recovery_sample_weights_change_loss_mass(self):
        import torch

        target = torch.zeros((2, 1, 14))
        prediction = torch.zeros_like(target)
        prediction[0, 0, 0] = 1.0
        prediction[1, 0, 0] = 3.0
        active_right = torch.zeros(2)
        phase = torch.ones(2)
        equal = weighted_action_loss(
            prediction,
            target,
            active_right,
            phase,
            active_arm_weight=1.0,
            relation_weight=1.0,
        )
        recovery_weighted = weighted_action_loss(
            prediction,
            target,
            active_right,
            phase,
            active_arm_weight=1.0,
            relation_weight=1.0,
            sample_weight=torch.tensor([1.0, 4.0]),
        )
        self.assertGreater(float(recovery_weighted), float(equal))

    def test_mlp_accepts_explicit_flow_progress(self):
        import torch

        model = build_action_predictor("flow", 5, 3, "mlp", True)
        batch = {
            "points_a": torch.randn(2, 7, 3),
            "points_b": torch.randn(2, 7, 3),
            "state": torch.randn(2, 20),
            "flow": torch.randn(2, 4, 18),
            "flow_progress": torch.rand(2, 1),
        }
        self.assertEqual(tuple(model(batch).shape), (2, 3, 14))


if __name__ == "__main__":
    unittest.main()
