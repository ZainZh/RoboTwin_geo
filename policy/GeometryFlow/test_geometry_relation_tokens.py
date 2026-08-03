from __future__ import annotations

import unittest

import numpy as np
import torch

from .train_geometry_relation_tokens import (
    GeometryRelationTokenPolicy,
    build_reference_pairs,
    gripper_x_flip,
    soft_correspondence,
    symmetry_invariant_functional_coordinates,
)


class GeometryRelationTokenTest(unittest.TestCase):
    def test_soft_correspondence_recovers_permutation(self):
        reference = torch.eye(4)[None]
        permutation = torch.tensor([2, 0, 3, 1])
        query = reference[:, permutation]
        weights, confidence, entropy, _ = soft_correspondence(
            query, reference, temperature=0.01
        )
        self.assertTrue(torch.equal(torch.argmax(weights[0], dim=-1), permutation))
        self.assertGreater(float(confidence.mean()), 0.95)
        self.assertLess(float(entropy.mean()), 0.05)

    def test_reference_pairs_are_cross_instance_and_arm_matched(self):
        payload = {
            "shoe_id_label_only": np.asarray([0, 1, 2, 3]),
            "active_arm_right": np.asarray([0, 0, 1, 1]),
            "episode_id": np.asarray([10, 11, 12, 13]),
        }
        pairs = build_reference_pairs(
            payload,
            np.arange(4),
            np.arange(4),
            references_per_target=1,
        )
        self.assertEqual(len(pairs), 4)
        for target, source in pairs:
            self.assertNotEqual(
                payload["shoe_id_label_only"][target],
                payload["shoe_id_label_only"][source],
            )
            self.assertEqual(
                payload["active_arm_right"][target],
                payload["active_arm_right"][source],
            )

    def test_policy_outputs_proper_short_grasp_target(self):
        torch.manual_seed(0)
        batch_size, points, descriptor_dim = 2, 16, 8
        model = GeometryRelationTokenPolicy(
            descriptor_dim=descriptor_dim,
            match_dim=8,
            feature_dim=32,
            heads=4,
            layers=1,
            dropout=0.0,
        )
        batch = {
            "points_world": torch.randn(batch_size, points, 3) * 0.05,
            "additional_features": torch.randn(
                batch_size, points, descriptor_dim
            ),
            "reference_points_world": torch.randn(batch_size, points, 3) * 0.05,
            "reference_features": torch.randn(
                batch_size, points, descriptor_dim
            ),
            "reference_keypoints_world": torch.randn(batch_size, 7, 3) * 0.05,
            "reference_transform_world": torch.eye(4)[None].repeat(
                batch_size, 1, 1
            ),
            "active_arm_right": torch.tensor([0.0, 1.0]),
        }
        output = model(batch)
        self.assertEqual(output.prediction.keypoints_world.shape, (2, 1, 7, 3))
        self.assertEqual(output.prediction.transforms.shape, (2, 1, 4, 4))
        rotation = output.prediction.transforms[:, 0, :3, :3]
        torch.testing.assert_close(
            torch.det(rotation), torch.ones(2), atol=1e-5, rtol=1e-5
        )

    def test_fused_matcher_starts_descriptor_independent(self):
        torch.manual_seed(1)
        model = GeometryRelationTokenPolicy(
            descriptor_dim=8,
            match_dim=8,
            feature_dim=32,
            heads=4,
            layers=1,
            dropout=0.0,
            matching_mode="fused",
        ).eval()
        batch = {
            "points_world": torch.randn(1, 12, 3) * 0.05,
            "additional_features": torch.randn(1, 12, 8),
            "reference_points_world": torch.randn(1, 12, 3) * 0.05,
            "reference_features": torch.randn(1, 12, 8),
            "reference_keypoints_world": torch.randn(1, 7, 3) * 0.05,
            "reference_transform_world": torch.eye(4)[None],
            "active_arm_right": torch.tensor([0.0]),
        }
        changed = {key: value.clone() for key, value in batch.items()}
        changed["additional_features"] = torch.randn_like(
            batch["additional_features"]
        )
        changed["reference_features"] = torch.randn_like(
            batch["reference_features"]
        )
        with torch.no_grad():
            first = model(batch).prediction.keypoints_world
            second = model(changed).prediction.keypoints_world
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_dual_memory_starts_descriptor_independent(self):
        torch.manual_seed(2)
        model = GeometryRelationTokenPolicy(
            descriptor_dim=8,
            match_dim=8,
            feature_dim=32,
            heads=4,
            layers=1,
            dropout=0.0,
            matching_mode="fused",
            task_refined_matching=True,
            fusion_style="dual_memory",
        ).eval()
        batch = {
            "points_world": torch.randn(1, 12, 3) * 0.05,
            "additional_features": torch.randn(1, 12, 8),
            "reference_points_world": torch.randn(1, 12, 3) * 0.05,
            "reference_features": torch.randn(1, 12, 8),
            "reference_keypoints_world": torch.randn(1, 7, 3) * 0.05,
            "reference_transform_world": torch.eye(4)[None],
            "active_arm_right": torch.tensor([0.0]),
        }
        changed = {key: value.clone() for key, value in batch.items()}
        changed["additional_features"] = torch.randn_like(
            batch["additional_features"]
        )
        changed["reference_features"] = torch.randn_like(
            batch["reference_features"]
        )
        with torch.no_grad():
            first = model(batch).prediction.keypoints_world
            second = model(changed).prediction.keypoints_world
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_functional_coordinates_ignore_parallel_gripper_x_flip(self):
        points = torch.randn(2, 9, 3)
        transform = torch.eye(4)[None].repeat(2, 1, 1)
        scale = torch.ones(2, 1)
        original = symmetry_invariant_functional_coordinates(
            points, transform, scale
        )
        flipped = symmetry_invariant_functional_coordinates(
            points,
            transform
            @ gripper_x_flip(dtype=transform.dtype, device=transform.device),
            scale,
        )
        torch.testing.assert_close(original, flipped)


if __name__ == "__main__":
    unittest.main()
