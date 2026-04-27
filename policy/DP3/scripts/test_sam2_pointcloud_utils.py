import unittest

import numpy as np

from sam2_pointcloud_utils import (
    Sam2CameraTrackingState,
    extract_placeholder_point_clouds_sam2_online,
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

    def test_load_sam2_bbox_prompt_file_filters_cameras_and_placeholders(self):
        import json
        import tempfile
        from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
