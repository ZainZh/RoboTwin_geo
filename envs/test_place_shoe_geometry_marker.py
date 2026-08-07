import unittest

import numpy as np

from .place_shoe_geometry_marker import (
    RAMP_FUNCTIONAL_ROTATION,
    geometry_marker_functional_matrix,
    geometry_marker_local_transform,
    oriented_rectangles_overlap,
    place_shoe_geometry_marker,
    resolve_shoe_id_candidates,
    resolve_shoe_modelname,
)
from .placement_metrics import functional_pose_alignment_errors


class TestPlaceShoeGeometryMarker(unittest.TestCase):
    def test_allowed_shoe_ids_default_and_explicit_subset(self):
        self.assertEqual(resolve_shoe_id_candidates({}), tuple(range(10)))
        self.assertEqual(resolve_shoe_id_candidates({"allowed_shoe_ids": [2, 7]}), (2, 7))

    def test_invalid_allowed_shoe_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            resolve_shoe_id_candidates({"allowed_shoe_ids": []})
        with self.assertRaisesRegex(ValueError, "duplicates"):
            resolve_shoe_id_candidates({"allowed_shoe_ids": [2, 2]})
        with self.assertRaisesRegex(ValueError, "invalid"):
            resolve_shoe_id_candidates({"allowed_shoe_ids": [10]})

    def test_external_assets_require_explicit_ids_without_legacy_cap(self):
        config = {
            "shoe_modelname": "gso_shoe_blind_v1",
            "allowed_shoe_ids": [4, 19],
        }
        self.assertEqual(resolve_shoe_modelname(config), "gso_shoe_blind_v1")
        self.assertEqual(resolve_shoe_id_candidates(config), (4, 19))
        with self.assertRaisesRegex(ValueError, "explicit frozen ID list"):
            resolve_shoe_id_candidates({"shoe_modelname": "gso_shoe_blind_v1"})
        with self.assertRaisesRegex(ValueError, "invalid"):
            resolve_shoe_modelname({"shoe_modelname": "../041_shoe"})

    def test_marker_transform_contains_requested_xy_and_yaw(self):
        transform = geometry_marker_local_transform(0.04, -0.02, np.pi / 2.0)
        np.testing.assert_allclose(transform[:3, 3], [0.04, -0.02, 0.0])
        np.testing.assert_allclose(
            transform[:3, :3] @ np.array([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1e-7,
        )

    def test_marker_functional_frame_composes_marker_and_ramp_frame(self):
        transform = geometry_marker_local_transform(0.03, 0.01, np.pi / 4.0)
        functional = geometry_marker_functional_matrix(0.03, 0.01, np.pi / 4.0)
        np.testing.assert_allclose(
            functional[:3, :3], transform[:3, :3] @ RAMP_FUNCTIONAL_ROTATION
        )
        np.testing.assert_allclose(functional[:3, 3], [0.03, 0.01, 0.0])

    def test_oriented_rectangle_overlap_rejects_penetration(self):
        axes = np.eye(2)
        self.assertTrue(
            oriented_rectangles_overlap(
                [0.18, -0.08], axes, [0.11, 0.05], [0.0, -0.08], axes, [0.18, 0.13]
            )
        )
        self.assertFalse(
            oriented_rectangles_overlap(
                [0.36, -0.08], axes, [0.04, 0.10], [0.0, -0.08], axes, [0.18, 0.13]
            )
        )

    def test_pose_errors_are_sign_invariant_for_quaternions(self):
        errors = functional_pose_alignment_errors(
            [0.01, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(errors["translation_error_norm_m"], 0.01)
        self.assertAlmostEqual(errors["rotation_error_deg"], 0.0)

    def test_success_requires_configured_consecutive_stable_checks(self):
        class FakePose:
            p = np.array([0.0, 0.0, 0.0])
            q = np.array([1.0, 0.0, 0.0, 0.0])

        class FakeActor:
            actor = object()

            def get_functional_point(self, *_args, **_kwargs):
                return FakePose()

            def get_name(self):
                return "fake"

        task = place_shoe_geometry_marker.__new__(place_shoe_geometry_marker)
        task.shoe = FakeActor()
        task.target_block = FakeActor()
        task.shoe_arm_name = "left"
        task.is_left_gripper_open = lambda: True
        task.check_actors_contact = lambda *_args: True
        task._shoe_velocity_norms = lambda: (0.0, 0.0)
        task.placement_success_config = {
            "require_contact": True,
            "stable_steps": 3,
            "max_linear_speed_mps": 0.02,
            "max_angular_speed_radps": 0.2,
        }
        task._success_hold_count = 0

        self.assertFalse(task.check_success())
        self.assertFalse(task.check_success())
        self.assertTrue(task.check_success())
        self.assertEqual(task._pose_alignment_step_count, 3)
        self.assertEqual(task._raw_success_step_count, 3)
        self.assertEqual(task._max_success_hold_count, 3)
        self.assertEqual(task._first_pose_alignment_step, 0)
        self.assertEqual(task._first_raw_success_step, 0)
        self.assertAlmostEqual(task._min_translation_error_norm_m, 0.0)
        self.assertAlmostEqual(task._min_rotation_error_deg, 0.0)
        self.assertAlmostEqual(task._best_combined_alignment_score, 0.0)
        task.check_actors_contact = lambda *_args: False
        self.assertFalse(task.check_success())
        self.assertEqual(task._success_hold_count, 0)
        self.assertEqual(task._pose_alignment_step_count, 4)
        self.assertEqual(task._raw_success_step_count, 3)
        self.assertEqual(task._max_success_hold_count, 3)


if __name__ == "__main__":
    unittest.main()
