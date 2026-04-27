import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


class RealZedCollectionPipelineTest(unittest.TestCase):
    def test_collect_module_parses_camera_labels_without_hardware_dependencies(self):
        from script.real_zed_collection.collect_zed_robotwin_raw import parse_camera_labels

        self.assertEqual(parse_camera_labels("global,ego,side"), ["global", "ego", "side"])
        self.assertEqual(parse_camera_labels(["global", "ego", "side"]), ["global", "ego", "side"])

    def test_postprocess_writes_robotwin_hdf5_from_raw_episode_and_masks(self):
        from script.real_zed_collection.postprocess.postprocess_raw_to_robotwin_hdf5 import postprocess_episode

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_episode = root / "raw" / "episode_000000"
            raw_episode.mkdir(parents=True)
            mask_root = root / "masks"
            out_dir = root / "out"
            calib_path = root / "calibration.yaml"

            calib_path.write_text(
                "\n".join(
                    [
                        "type: three_camera_charuco_extrinsics",
                        "reference_camera: cam0",
                        "cameras:",
                        "  cam0:",
                        "    serial_number: 1",
                        "    camera_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]",
                        "  cam1:",
                        "    serial_number: 2",
                        "    camera_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]",
                        "relative_to_reference:",
                        "  cam0:",
                        "    t_ref_from_cam: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]",
                        "  cam1:",
                        "    t_ref_from_cam: [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]",
                    ]
                ),
                encoding="utf-8",
            )

            frames = []
            for frame_idx in range(3):
                robot_path = raw_episode / f"robot_{frame_idx:06d}.npz"
                np.savez(robot_path, joint_vector=np.full((14,), frame_idx, dtype=np.float32))
                camera_frames = {}
                for cam_label in ("cam0", "cam1"):
                    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
                    rgb[..., 0] = 10 + frame_idx
                    depth_m = np.ones((2, 2), dtype=np.float32) * (1.0 + 0.1 * frame_idx)
                    frame_path = raw_episode / f"{cam_label}_{frame_idx:06d}.npz"
                    np.savez(frame_path, rgb=rgb, depth_m=depth_m)
                    camera_frames[cam_label] = str(frame_path.name)

                    mask_dir = mask_root / "A" / cam_label
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask = np.zeros((2, 2), dtype=np.uint8)
                    mask[0, 0] = 255
                    mask[1, 1] = 255
                    from imageio.v2 import imwrite

                    imwrite(mask_dir / f"mask_{frame_idx:06d}.png", mask)

                frames.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_unix_sec": float(frame_idx),
                        "robot": str(robot_path.name),
                        "cameras": camera_frames,
                    }
                )

            (raw_episode / "manifest.json").write_text(
                json.dumps({"frames": frames, "camera_labels": ["cam0", "cam1"]}, indent=2),
                encoding="utf-8",
            )

            hdf5_path = postprocess_episode(
                raw_episode_dir=raw_episode,
                output_dir=out_dir,
                episode_index=0,
                calibration_path=calib_path,
                camera_labels=["cam0", "cam1"],
                object_prompts={"{A}": "mug"},
                mask_root=mask_root,
                scene_point_num=8,
                object_point_num=4,
                min_depth_m=0.1,
                max_depth_m=2.0,
            )

            self.assertTrue(hdf5_path.exists())
            with h5py.File(hdf5_path, "r") as root_h5:
                self.assertEqual(root_h5["/joint_action/vector"].shape, (3, 14))
                self.assertEqual(root_h5["/pointcloud"].shape, (3, 8, 6))
                self.assertEqual(root_h5["/object_pointcloud/{A}"].shape, (3, 4, 6))
                self.assertEqual(root_h5["/observation/cam0/rgb"].shape, (3, 2, 2, 3))
                self.assertEqual(root_h5["/observation/cam0/depth"].shape, (3, 2, 2))
                self.assertEqual(root_h5["/observation/cam0/intrinsic_cv"].shape, (3, 3))
                self.assertEqual(root_h5["/observation/cam1/cam2world_gl"].shape, (4, 4))
                np.testing.assert_allclose(root_h5["/joint_action/vector"][2], np.full((14,), 2))

            compact_hdf5_path = postprocess_episode(
                raw_episode_dir=raw_episode,
                output_dir=out_dir,
                episode_index=1,
                calibration_path=calib_path,
                camera_labels=["cam0", "cam1"],
                object_prompts={"{A}": "mug"},
                mask_root=mask_root,
                scene_point_num=8,
                object_point_num=4,
                min_depth_m=0.1,
                max_depth_m=2.0,
                store_observations=False,
            )
            with h5py.File(compact_hdf5_path, "r") as root_h5:
                self.assertNotIn("observation", root_h5)
                self.assertEqual(root_h5["/pointcloud"].shape, (3, 8, 6))
                self.assertEqual(root_h5["/object_pointcloud/{A}"].shape, (3, 4, 6))

    def test_load_three_zed_calibration_can_return_workspace_frame(self):
        from script.real_zed_collection.real_zed_utils import load_three_zed_calibration

        with tempfile.TemporaryDirectory() as tmp:
            calib_path = Path(tmp) / "calibration.yaml"
            calib_path.write_text(
                "\n".join(
                    [
                        "type: three_camera_charuco_extrinsics",
                        "reference_camera: cam0",
                        "cameras:",
                        "  cam0:",
                        "    serial_number: 1",
                        "    camera_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]",
                        "  cam1:",
                        "    serial_number: 2",
                        "    camera_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]",
                        "relative_to_reference:",
                        "  cam0:",
                        "    t_ref_from_cam: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]",
                        "  cam1:",
                        "    t_ref_from_cam: [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]",
                        "workspace:",
                        "  reference_camera: cam0",
                        "  t_workspace_from_ref: [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]",
                    ]
                ),
                encoding="utf-8",
            )

            calib = load_three_zed_calibration(calib_path, frame_mode="workspace")
            np.testing.assert_allclose(calib["cam0"].t_world_from_cam[:3, 3], np.array([1.0, 2.0, 3.0]))
            np.testing.assert_allclose(calib["cam1"].t_world_from_cam[:3, 3], np.array([1.1, 2.0, 3.0]))

    def test_workspace_bbox_projection_crop_updates_intrinsics_and_metadata(self):
        from script.real_zed_collection.workspace_crop_utils import (
            WorkspaceBounds,
            apply_workspace_crop_to_camera_frame,
            project_workspace_bbox_to_roi,
        )

        camera_matrix = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        t_workspace_from_cam = np.eye(4, dtype=np.float64)
        bounds = WorkspaceBounds(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, z_min=1.0, z_max=1.0)

        roi = project_workspace_bbox_to_roi(
            camera_matrix=camera_matrix,
            image_shape_hw=(100, 100),
            t_workspace_from_cam=t_workspace_from_cam,
            bounds=bounds,
            margin_px=2,
        )
        self.assertEqual(roi, (38, 38, 63, 63))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth_m = np.ones((100, 100), dtype=np.float32)
        cropped = apply_workspace_crop_to_camera_frame(
            rgb=rgb,
            depth_m=depth_m,
            camera_matrix=camera_matrix,
            t_workspace_from_cam=t_workspace_from_cam,
            bounds=bounds,
            margin_px=2,
        )

        self.assertEqual(cropped["rgb"].shape, (25, 25, 3))
        self.assertEqual(cropped["depth_m"].shape, (25, 25))
        self.assertEqual(cropped["depth_crop_box_xyxy"].tolist(), [38, 38, 63, 63])
        np.testing.assert_allclose(cropped["camera_matrix"][0, 2], 12.0)
        np.testing.assert_allclose(cropped["camera_matrix"][1, 2], 12.0)

    def test_workspace_axis_convention_flips_board_z_by_default(self):
        from script.real_zed_collection.calibrate_workspace_frame import _workspace_from_board_transform

        t_board_from_workspace = _workspace_from_board_transform(flip_z=True)
        np.testing.assert_allclose(t_board_from_workspace[:3, :3], np.diag([1.0, -1.0, -1.0]))
        self.assertAlmostEqual(float(np.linalg.det(t_board_from_workspace[:3, :3])), 1.0)

        board_z_from_workspace_z = t_board_from_workspace[:3, 2]
        np.testing.assert_allclose(board_z_from_workspace_z, np.array([0.0, 0.0, -1.0]))

    def test_calibration_label_map_prefers_manifest_serial_numbers(self):
        from script.real_zed_collection.real_zed_utils import CameraCalibration, calibration_label_map_from_manifest

        calib = {
            "global": CameraCalibration("global", 31021548, np.eye(3), np.eye(4)),
            "left": CameraCalibration("left", 37856216, np.eye(3), np.eye(4)),
            "right": CameraCalibration("right", 38968158, np.eye(3), np.eye(4)),
        }
        manifest = {
            "camera_labels": ["global", "left", "right"],
            "camera_serials": {"global": 38968158, "left": 31021548, "right": 37856216},
        }

        self.assertEqual(
            calibration_label_map_from_manifest(manifest, calib, ["global", "left", "right"]),
            {"global": "right", "left": "global", "right": "left"},
        )

    def test_camera_workspace_polygon_mask_rasterizes_and_loads(self):
        from imageio.v2 import imwrite

        from script.real_zed_collection.select_camera_workspace_masks import (
            load_camera_workspace_masks,
            rasterize_polygon_mask,
        )

        mask = rasterize_polygon_mask((10, 10), [(2, 2), (7, 2), (7, 7), (2, 7)])
        self.assertGreater(int(mask.sum()), 0)
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[4, 4]))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "global").mkdir()
            imwrite(root / "global" / "workspace_mask.png", mask.astype(np.uint8) * 255)
            loaded = load_camera_workspace_masks(root, ["global", "left"])

            self.assertIn("global", loaded)
            self.assertNotIn("left", loaded)
            np.testing.assert_array_equal(loaded["global"], mask)

    def test_sam2_bbox_prompt_round_trips(self):
        from script.real_zed_collection.select_sam2_bboxes import (
            load_sam2_bbox_prompts,
            normalize_bbox_xyxy,
            save_sam2_bbox_prompts,
        )

        self.assertEqual(normalize_bbox_xyxy((9, 8, 1, 2), (10, 10)), [1, 2, 9, 8])

        with tempfile.TemporaryDirectory() as tmp:
            path = save_sam2_bbox_prompts(
                output_root=Path(tmp),
                raw_episode_dir="/raw/episode",
                frame_index=0,
                image_shapes_by_camera={"global": (10, 20)},
                boxes_by_camera={"global": {"{A}": (9, 8, 1, 2), "{B}": (0, 0, 5, 5)}},
            )
            loaded = load_sam2_bbox_prompts(path, camera_labels=["global"], placeholders=["{A}", "{B}"])

            self.assertEqual(loaded["global"]["{A}"], [1, 2, 9, 8])
            self.assertEqual(loaded["global"]["{B}"], [0, 0, 5, 5])

    def test_sam2_logits_are_mapped_to_placeholder_masks(self):
        from script.real_zed_collection.sam2_tracking_utils import masks_from_sam2_logits

        logits = np.zeros((2, 1, 4, 5), dtype=np.float32)
        logits[0, 0, 1:3, 1:4] = 1.0
        logits[1, 0, 0:2, 0:2] = 1.0

        masks = masks_from_sam2_logits(
            out_obj_ids=[2, 1],
            out_mask_logits=logits,
            obj_id_to_placeholder={1: "{A}", 2: "{B}"},
            image_shape_hw=(4, 5),
        )

        self.assertEqual(int(masks["{A}"].sum()), 4)
        self.assertEqual(int(masks["{B}"].sum()), 6)

    def test_sam2_tracker_runs_predictor_calls_under_autocast(self):
        from script.real_zed_collection.sam2_tracking_utils import SAM2StreamingObjectTracker

        events = []

        class FakeAutocast:
            def __init__(self, *, device_type, dtype):
                self.device_type = device_type
                self.dtype = dtype

            def __enter__(self):
                events.append(("enter", self.device_type, self.dtype))

            def __exit__(self, exc_type, exc, tb):
                events.append(("exit", self.device_type, self.dtype))
                return False

        class FakeTorch:
            bfloat16 = "bfloat16"
            float16 = "float16"

            @staticmethod
            def autocast(*, device_type, dtype):
                return FakeAutocast(device_type=device_type, dtype=dtype)

        class FakePredictor:
            def load_first_frame(self, image):
                events.append(("load_first_frame", events[-1][0]))

            def add_new_prompt(self, **_kwargs):
                events.append(("add_new_prompt", events[-1][0]))
                logits = np.ones((1, 1, 4, 5), dtype=np.float32)
                return 0, [1], logits

            def track(self, image):
                events.append(("track", events[-1][0]))
                logits = np.ones((1, 1, 4, 5), dtype=np.float32)
                return [1], logits

        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            tracker = SAM2StreamingObjectTracker(
                predictor=FakePredictor(),
                placeholders=["{A}"],
                device="cuda:0",
                autocast_dtype="bfloat16",
            )
            tracker.initialize(np.zeros((4, 5, 3), dtype=np.uint8), {"{A}": [0, 0, 2, 2]})
            tracker.track(np.zeros((4, 5, 3), dtype=np.uint8))

        self.assertIn(("load_first_frame", "enter"), events)
        self.assertIn(("add_new_prompt", "enter"), events)
        self.assertIn(("track", "enter"), events)

    def test_sam2_segment_episode_writes_masks_with_fake_tracker(self):
        from imageio.v2 import imread

        from script.real_zed_collection.segment_objects_sam2 import segment_episode_sam2

        class FakeTracker:
            def __init__(self):
                self.init_calls = 0
                self.track_calls = 0

            def initialize(self, image, boxes_by_placeholder):
                self.init_calls += 1
                return {
                    "{A}": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (1, 2))),
                    "{B}": np.pad(np.ones((1, 3), dtype=bool), ((0, 3), (0, 2))),
                }

            def track(self, image):
                self.track_calls += 1
                return {
                    "{A}": np.pad(np.ones((1, 2), dtype=bool), ((2, 1), (2, 1))),
                    "{B}": np.pad(np.ones((2, 1), dtype=bool), ((1, 1), (3, 1))),
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_episode = root / "episode_000000"
            raw_episode.mkdir()
            frames = []
            for idx in range(2):
                frame_path = raw_episode / f"global_{idx:06d}.npz"
                np.savez(frame_path, rgb=np.full((4, 5, 3), idx, dtype=np.uint8), depth_m=np.ones((4, 5), dtype=np.float32))
                frames.append(
                    {
                        "frame_index": idx,
                        "timestamp_unix_sec": float(idx),
                        "cameras": {"global": frame_path.name},
                    }
                )
            (raw_episode / "manifest.json").write_text(
                json.dumps({"camera_labels": ["global"], "frames": frames}),
                encoding="utf-8",
            )

            tracker = FakeTracker()
            mask_root = segment_episode_sam2(
                raw_episode_dir=raw_episode,
                output_mask_root=root / "masks",
                bbox_prompts_by_camera={"global": {"{A}": [1, 1, 3, 3], "{B}": [0, 0, 3, 1]}},
                camera_labels=["global"],
                placeholders=["{A}", "{B}"],
                tracker_factory=lambda _label: tracker,
                max_frames=2,
            )

            self.assertEqual(tracker.init_calls, 1)
            self.assertEqual(tracker.track_calls, 1)
            self.assertEqual(int((imread(mask_root / "{A}" / "global" / "mask_000000.png") > 0).sum()), 4)
            self.assertEqual(int((imread(mask_root / "{A}" / "global" / "mask_000001.png") > 0).sum()), 2)
            meta = json.loads((mask_root / "sam2_mask_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["tracker"], "sam2_streaming")

    def test_sam2_objpc_batch_defaults_use_sam2_paths(self):
        from script.real_zed_collection.postprocess.postprocess_real_zed_sam2_objpc_dataset import (
            DEFAULT_TASK_CONFIG,
            default_bbox_prompt_root,
            default_output_dir,
        )

        self.assertEqual(DEFAULT_TASK_CONFIG, "demo_real_zed_sam2_objpc")
        self.assertEqual(
            default_output_dir("grasp_mug", DEFAULT_TASK_CONFIG, user="unit_user"),
            Path("/media/unit_user/Extreme SSD/geo_mani_data/grasp_mug/robotwin_objpc/demo_real_zed_sam2_objpc"),
        )
        self.assertEqual(
            default_bbox_prompt_root("grasp_mug", user="unit_user"),
            Path("/media/unit_user/Extreme SSD/geo_mani_data/grasp_mug/sam2_bbox_prompts"),
        )

    def test_sam2_objpc_batch_prefers_per_episode_bbox_prompts(self):
        from script.real_zed_collection.postprocess.postprocess_real_zed_sam2_objpc_dataset import (
            resolve_episode_bbox_prompts,
        )
        from script.real_zed_collection.select_sam2_bboxes import save_sam2_bbox_prompts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bbox_root = root / "bbox"
            raw_episode = root / "raw" / "episode_202604270001"
            raw_episode.mkdir(parents=True)

            save_sam2_bbox_prompts(
                output_root=bbox_root,
                raw_episode_dir="/global",
                frame_index=0,
                image_shapes_by_camera={"global": (10, 10)},
                boxes_by_camera={"global": {"{A}": [0, 0, 2, 2]}},
            )
            save_sam2_bbox_prompts(
                output_root=bbox_root / raw_episode.name,
                raw_episode_dir=raw_episode,
                frame_index=0,
                image_shapes_by_camera={"global": (10, 10)},
                boxes_by_camera={"global": {"{A}": [3, 3, 6, 6]}},
            )

            boxes, prompt_path = resolve_episode_bbox_prompts(
                bbox_prompt_root=bbox_root,
                raw_episode_dir=raw_episode,
                episode_index=0,
                camera_labels=["global"],
                placeholders=["{A}"],
            )

        self.assertEqual(boxes["global"]["{A}"], [3, 3, 6, 6])
        self.assertIn("episode_202604270001", prompt_path)

    def test_sam2_objpc_batch_can_require_per_episode_bbox_prompts(self):
        from script.real_zed_collection.postprocess.postprocess_real_zed_sam2_objpc_dataset import (
            resolve_episode_bbox_prompts,
        )
        from script.real_zed_collection.select_sam2_bboxes import save_sam2_bbox_prompts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bbox_root = root / "bbox"
            raw_episode = root / "raw" / "episode_202604270001"
            raw_episode.mkdir(parents=True)
            save_sam2_bbox_prompts(
                output_root=bbox_root,
                raw_episode_dir="/global",
                frame_index=0,
                image_shapes_by_camera={"global": (10, 10)},
                boxes_by_camera={"global": {"{A}": [0, 0, 2, 2]}},
            )

            with self.assertRaises(FileNotFoundError):
                resolve_episode_bbox_prompts(
                    bbox_prompt_root=bbox_root,
                    raw_episode_dir=raw_episode,
                    episode_index=0,
                    camera_labels=["global"],
                    placeholders=["{A}"],
                    require_per_episode=True,
                )

if __name__ == "__main__":
    unittest.main()
