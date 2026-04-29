import unittest

import numpy as np


class RealSam2ObjectPointcloudPreviewTest(unittest.TestCase):
    def test_prepare_placeholder_clouds_filters_empty_rows_and_applies_placeholder_colors(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import prepare_placeholder_clouds

        object_pointcloud = {
            "{A}": np.array(
                [
                    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "{B}": np.array([[0.4, 0.5, 0.6, 0.2, 0.3, 0.4]], dtype=np.float32),
        }

        clouds = prepare_placeholder_clouds(
            object_pointcloud,
            placeholders=["{A}", "{B}"],
            color_mode="placeholder",
        )

        self.assertEqual(set(clouds), {"{A}", "{B}"})
        self.assertEqual(clouds["{A}"].shape, (1, 6))
        np.testing.assert_allclose(clouds["{A}"][0, :3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(clouds["{A}"][0, 3:6], [0.95, 0.25, 0.2])
        np.testing.assert_allclose(clouds["{B}"][0, 3:6], [0.2, 0.45, 1.0])

    def test_prepare_placeholder_clouds_can_keep_rgb_colors(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import prepare_placeholder_clouds

        object_pointcloud = {
            "{A}": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
        }

        clouds = prepare_placeholder_clouds(
            object_pointcloud,
            placeholders=["{A}"],
            color_mode="rgb",
        )

        np.testing.assert_allclose(clouds["{A}"][0, 3:6], [0.4, 0.5, 0.6])

    def test_parser_supports_single_object_and_headless_fps_mode(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--object_placeholders",
                "{A}",
                "--no_open3d",
                "--point_size",
                "8",
                "--open3d_width",
                "1800",
                "--open3d_height",
                "1100",
            ]
        )

        self.assertEqual(args.object_placeholders, "{A}")
        self.assertTrue(args.no_open3d)
        self.assertEqual(args.point_size, 8.0)
        self.assertEqual(args.open3d_width, 1800)
        self.assertEqual(args.open3d_height, 1100)
        self.assertTrue(args.show_workspace_box)

    def test_workspace_box_corners_are_transformed_to_output_frame(self):
        from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds
        from script.real_zed_inference.preview_sam2_object_pointcloud import workspace_box_corners_in_output_frame

        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = 10.0

        corners = workspace_box_corners_in_output_frame(
            WorkspaceBounds(
                x_min=0.0,
                x_max=1.0,
                y_min=0.0,
                y_max=2.0,
                z_min=0.0,
                z_max=3.0,
            ),
            t_output_from_workspace=transform,
        )

        self.assertEqual(corners.shape, (8, 3))
        self.assertAlmostEqual(float(corners[:, 0].min()), 10.0)
        self.assertAlmostEqual(float(corners[:, 0].max()), 11.0)
        self.assertAlmostEqual(float(corners[:, 1].max()), 2.0)
        self.assertAlmostEqual(float(corners[:, 2].max()), 3.0)


if __name__ == "__main__":
    unittest.main()
