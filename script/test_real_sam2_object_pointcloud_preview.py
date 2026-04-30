import unittest
from types import SimpleNamespace
from unittest import mock

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
        self.assertTrue(args.object_only)

    def test_object_only_preview_observation_skips_dense_scene_pointcloud_construction(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import (
            TimingRecorder,
            build_preview_observation,
        )

        live = SimpleNamespace(
            labels=["global"],
            label_to_calib={"global": "global"},
            calibrations={"global": SimpleNamespace(camera_matrix=np.eye(3, dtype=np.float32))},
            t_output_from_cam_by_label={"global": np.eye(4, dtype=np.float32)},
            t_workspace_from_cam_by_label={"global": np.eye(4, dtype=np.float32)},
            workspace_bounds=None,
        )
        args = SimpleNamespace(
            min_depth_m=0.05,
            max_depth_m=3.0,
            scene_point_num=4,
            parallel_camera_workers=1,
        )
        frame = {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth_m": np.ones((2, 2), dtype=np.float32),
            "camera_matrix": np.eye(3, dtype=np.float32),
        }

        with mock.patch(
            "script.real_zed_inference.preview_sam2_object_pointcloud.snapshot_frames",
            return_value={"global": frame},
        ), mock.patch(
            "script.real_zed_inference.preview_sam2_object_pointcloud.camera_frame_to_output_pc",
            side_effect=AssertionError("dense scene path should be skipped"),
        ):
            observation, dense_scene = build_preview_observation(
                args=args,
                live=live,
                timer=TimingRecorder(enabled=True),
                build_scene=False,
            )

        self.assertEqual(dense_scene.shape, (0, 6))
        self.assertEqual(observation["pointcloud"].shape, (4, 6))
        self.assertIn("global", observation["observation"])

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

    def test_timing_recorder_formats_nested_stage_names(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import TimingRecorder

        timer = TimingRecorder(enabled=True)
        timer.timings["build_obs.global_pc"] = 0.001
        timer.timings["sam2.global.track"] = 0.002

        formatted = timer.format()

        self.assertIn("build_obs.global_pc=1.0ms", formatted)
        self.assertIn("sam2.global.track=2.0ms", formatted)

    def test_parser_supports_parallel_camera_workers(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import build_arg_parser

        args = build_arg_parser().parse_args(["--parallel_camera_workers", "2", "--no-object_only"])

        self.assertEqual(args.parallel_camera_workers, 2)
        self.assertFalse(args.object_only)


if __name__ == "__main__":
    unittest.main()
