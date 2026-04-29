import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam2_pointcloud_utils import (
    Sam2CameraTrackingState,
    _display_to_image_point,
    _display_scale_for_image,
    _try_normalize_bbox_xyxy,
    extract_placeholder_point_clouds_sam2_online,
    fast_merge_object_point_clouds,
    load_sam2_bbox_prompt_file,
)


class FakeSam2Tracker:
    def __init__(self):
        self.init_calls = 0
        self.track_calls = 0

    def initialize(self, image, boxes_by_placeholder):
        self.init_calls += 1
        masks = {
            "{A}": np.zeros(image.shape[:2], dtype=bool),
            "{B}": np.zeros(image.shape[:2], dtype=bool),
        }
        masks["{A}"][1, 1] = True
        masks["{B}"][2, 2] = True
        return masks

    def track(self, image):
        self.track_calls += 1
        masks = {
            "{A}": np.zeros(image.shape[:2], dtype=bool),
            "{B}": np.zeros(image.shape[:2], dtype=bool),
        }
        masks["{A}"][1, 2] = True
        masks["{B}"][2, 1] = True
        return masks


class Sam2PointcloudUtilsTest(unittest.TestCase):
    def test_interactive_bbox_helpers_ignore_single_click_and_scale_hd_images(self):
        self.assertIsNone(_try_normalize_bbox_xyxy((1387, 681, 1387, 681), (1080, 1920)))
        self.assertEqual(_try_normalize_bbox_xyxy((100, 50, 200, 150), (1080, 1920)), [100, 50, 200, 150])
        self.assertAlmostEqual(_display_scale_for_image((1080, 1920), max_width=1280, max_height=720), 2.0 / 3.0)
        self.assertEqual(_display_to_image_point(640, 360, scale=2.0 / 3.0, image_shape_hw=(1080, 1920)), (960, 540))

    def test_fast_merge_object_point_clouds_returns_fixed_count_without_fps(self):
        cloud_a = np.concatenate(
            [
                np.arange(20, dtype=np.float32).reshape(10, 2),
                np.ones((10, 4), dtype=np.float32),
            ],
            axis=1,
        )
        cloud_b = np.concatenate(
            [
                np.arange(20, 40, dtype=np.float32).reshape(10, 2),
                np.ones((10, 4), dtype=np.float32) * 2.0,
            ],
            axis=1,
        )

        merged = fast_merge_object_point_clouds([cloud_a, cloud_b], target_num_points=8)

        self.assertEqual(merged.shape, (8, 6))
        np.testing.assert_allclose(merged[0], cloud_a[0, :6])
        np.testing.assert_allclose(merged[-1], cloud_b[-1, :6])

    def _observation(self):
        pointcloud = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 1.0, 0.0, 0.0, 1.0],
                [2.0, 1.0, 1.0, 0.5, 0.0, 0.0],
                [1.0, 2.0, 1.0, 0.0, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        return {
            "pointcloud": pointcloud,
            "observation": {
                "global": {
                    "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                    "intrinsic_cv": np.eye(3, dtype=np.float32),
                    "extrinsic_cv": np.eye(4, dtype=np.float32),
                }
            },
        }

    def test_sam2_online_initializes_once_and_returns_all_placeholders(self):
        tracker = FakeSam2Tracker()
        state_by_camera = {}

        clouds, meta = extract_placeholder_point_clouds_sam2_online(
            self._observation(),
            placeholders=["{A}", "{B}"],
            camera_names=["global"],
            tracker_factory=lambda _camera: tracker,
            tracking_state_by_camera=state_by_camera,
            bbox_prompts_by_camera={"global": {"{A}": [0, 0, 2, 2], "{B}": [1, 1, 3, 3]}},
            target_num_points=1,
            min_mask_points=1,
        )

        self.assertEqual(tracker.init_calls, 1)
        self.assertEqual(tracker.track_calls, 0)
        self.assertIsInstance(state_by_camera["global"], Sam2CameraTrackingState)
        np.testing.assert_allclose(clouds["{A}"][0, :3], np.array([1.0, 1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(clouds["{B}"][0, :3], np.array([2.0, 2.0, 1.0], dtype=np.float32))
        self.assertEqual(meta["cameras"]["global"]["mode"], "initialized")

    def test_sam2_online_tracks_without_reinitializing(self):
        tracker = FakeSam2Tracker()
        state_by_camera = {
            "global": Sam2CameraTrackingState(tracker=tracker, initialized=True),
        }

        clouds, meta = extract_placeholder_point_clouds_sam2_online(
            self._observation(),
            placeholders=["{A}", "{B}"],
            camera_names=["global"],
            tracker_factory=lambda _camera: tracker,
            tracking_state_by_camera=state_by_camera,
            bbox_prompts_by_camera={},
            target_num_points=1,
            min_mask_points=1,
        )

        self.assertEqual(tracker.init_calls, 0)
        self.assertEqual(tracker.track_calls, 1)
        np.testing.assert_allclose(clouds["{A}"][0, :3], np.array([2.0, 1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(clouds["{B}"][0, :3], np.array([1.0, 2.0, 1.0], dtype=np.float32))
        self.assertEqual(meta["cameras"]["global"]["mode"], "tracked")

    def test_sam2_online_can_lift_masked_depth_points_and_filter_workspace(self):
        tracker = FakeSam2Tracker()
        state_by_camera = {}
        depth = np.ones((4, 4), dtype=np.float32)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[1, 1] = [255, 0, 0]

        clouds, meta = extract_placeholder_point_clouds_sam2_online(
            {
                "pointcloud": np.zeros((0, 6), dtype=np.float32),
                "observation": {
                    "global": {
                        "rgb": rgb,
                        "depth": depth,
                        "intrinsic_cv": np.eye(3, dtype=np.float32),
                        "cam2world_gl": np.eye(4, dtype=np.float32),
                        "t_workspace_from_cam": np.eye(4, dtype=np.float32),
                        "workspace_bounds_m": np.array([-0.1, 1.5, -0.1, 1.5, 0.5, 1.5], dtype=np.float32),
                    }
                },
            },
            placeholders=["{A}", "{B}"],
            camera_names=["global"],
            tracker_factory=lambda _camera: tracker,
            tracking_state_by_camera=state_by_camera,
            bbox_prompts_by_camera={"global": {"{A}": [0, 0, 2, 2], "{B}": [1, 1, 3, 3]}},
            target_num_points=1,
            min_mask_points=1,
        )

        np.testing.assert_allclose(clouds["{A}"][0, :3], np.array([1.0, 1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(clouds["{A}"][0, 3:6], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(clouds["{B}"], np.zeros((1, 6), dtype=np.float32))
        self.assertEqual(meta["cameras"]["global"]["placeholders"]["{A}"]["mode"], "mask_depth_lifted")
        self.assertEqual(meta["cameras"]["global"]["placeholders"]["{B}"]["mode"], "too_few_mask_depth_points")

    def test_load_sam2_bbox_prompt_file_filters_cameras_and_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sam2_bbox_prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "records": {
                            "global": {
                                "{A}": {"bbox_xyxy": [0, 0, 2, 2]},
                                "{B}": {"bbox_xyxy": [1, 1, 3, 3]},
                            },
                            "left": {"{A}": {"bbox_xyxy": [4, 4, 5, 5]}},
                        }
                    }
                ),
                encoding="utf-8",
            )

            boxes = load_sam2_bbox_prompt_file(path, camera_names=["global"], placeholders=["{B}"])

            self.assertEqual(boxes, {"global": {"{B}": [1, 1, 3, 3]}})

    def test_load_sam2_prompt_file_accepts_point_prompt_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sam2_bbox_prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "records": {
                            "global": {
                                "{A}": {
                                    "prompt_type": "point",
                                    "points_xy": [[2, 3], [4, 5]],
                                    "point_labels": [1, 0],
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            prompts = load_sam2_bbox_prompt_file(path, camera_names=["global"], placeholders=["{A}"])

            self.assertEqual(
                prompts,
                {"global": {"{A}": {"prompt_type": "point", "points_xy": [[2, 3], [4, 5]], "point_labels": [1, 0]}}},
            )

    def test_load_sam2_prompt_file_preserves_bbox_shape_for_resized_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sam2_bbox_prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "records": {
                            "global": {
                                "{A}": {
                                    "bbox_xyxy": [100, 50, 300, 150],
                                    "image_shape_hw": [1000, 2000],
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            prompts = load_sam2_bbox_prompt_file(path, camera_names=["global"], placeholders=["{A}"])

            self.assertEqual(
                prompts,
                {"global": {"{A}": {"bbox_xyxy": [100, 50, 300, 150], "image_shape_hw": [1000, 2000]}}},
            )


if __name__ == "__main__":
    unittest.main()
