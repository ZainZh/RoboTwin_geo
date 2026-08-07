from __future__ import annotations

import unittest

import numpy as np

from dinov2_visibility_utils import (
    compute_depth_visibility,
    fuse_multiview_features,
    project_xyz_to_camera,
    query_zbuffer_visibility,
    sample_feature_grid_bilinear,
)


class DinoV2VisibilityUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.asarray(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.identity = np.eye(4, dtype=np.float32)

    def test_projection_rejects_behind_camera_and_out_of_frame(self) -> None:
        xyz = np.asarray(
            [[0.0, 0.0, 1.0], [4.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )
        pixels, depth, valid = project_xyz_to_camera(
            xyz,
            camera_matrix=self.camera_matrix,
            t_cam_from_output=self.identity,
            image_shape_hw=(5, 5),
        )
        np.testing.assert_allclose(pixels[0], [2.0, 2.0])
        np.testing.assert_allclose(depth, [1.0, 1.0, -1.0])
        np.testing.assert_array_equal(valid, [True, False, False])

    def test_depth_visibility_rejects_occluded_invalid_and_masked_points(self) -> None:
        depth = np.ones((5, 5), dtype=np.float32)
        depth[2, 3] = 0.0
        mask = np.ones((5, 5), dtype=bool)
        mask[3, 2] = False
        xyz = np.asarray(
            [
                [0.0, 0.0, 1.0],  # visible at (2, 2)
                [0.0, 0.0, 1.2],  # occluded by observed depth and query z-buffer
                [1.0, 0.0, 1.0],  # invalid measured depth at (3, 2)
                [0.0, 1.0, 1.0],  # rejected by foreground mask at (2, 3)
            ],
            dtype=np.float32,
        )
        result = compute_depth_visibility(
            xyz,
            depth_m=depth,
            camera_matrix=self.camera_matrix,
            t_cam_from_output=self.identity,
            foreground_mask=mask,
            absolute_tolerance_m=0.03,
            relative_tolerance=0.0,
        )
        np.testing.assert_array_equal(result.visible, [True, False, False, False])
        np.testing.assert_array_equal(result.depth_valid, [True, True, False, True])
        np.testing.assert_array_equal(result.foreground_valid, [True, True, True, False])
        self.assertGreater(float(result.confidence_weight[0]), 0.0)
        np.testing.assert_array_equal(result.confidence_weight[1:], 0.0)

    def test_query_zbuffer_keeps_only_front_surface_beyond_tolerance(self) -> None:
        pixels = np.asarray([[2.1, 2.1], [2.2, 2.2], [1.0, 1.0]], dtype=np.float32)
        depth = np.asarray([1.0, 1.02, 2.0], dtype=np.float32)
        visible = query_zbuffer_visibility(
            pixels,
            depth,
            valid=np.ones(3, dtype=bool),
            image_shape_hw=(5, 5),
            tolerance_m=0.005,
        )
        np.testing.assert_array_equal(visible, [True, False, True])

    def test_bilinear_feature_sampling_uses_image_to_grid_alignment(self) -> None:
        grid = np.asarray([[[0.0], [2.0]], [[4.0], [6.0]]], dtype=np.float32)
        pixels = np.asarray([[1.5, 1.5], [0.0, 0.0], [9.0, 9.0]], dtype=np.float32)
        sampled = sample_feature_grid_bilinear(
            grid,
            pixels,
            image_shape_hw=(4, 4),
        )
        np.testing.assert_allclose(sampled[:, 0], [3.0, 0.0, 0.0], atol=1e-6)

    def test_multiview_weighted_fusion_and_all_invisible_zero(self) -> None:
        view_a = np.asarray([[1.0, 0.0], [4.0, 4.0], [9.0, 9.0]], dtype=np.float32)
        view_b = np.asarray([[0.0, 1.0], [2.0, 2.0], [8.0, 8.0]], dtype=np.float32)
        fused, valid, weight_sum, view_count = fuse_multiview_features(
            [view_a, view_b],
            [np.asarray([True, True, False]), np.asarray([True, False, False])],
            weights_by_view=[np.asarray([1.0, 2.0, 0.0]), np.asarray([3.0, 1.0, 0.0])],
        )
        np.testing.assert_allclose(fused[0], [0.25, 0.75])
        np.testing.assert_allclose(fused[1], [4.0, 4.0])
        np.testing.assert_array_equal(fused[2], [0.0, 0.0])
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_allclose(weight_sum, [4.0, 2.0, 0.0])
        np.testing.assert_array_equal(view_count, [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
