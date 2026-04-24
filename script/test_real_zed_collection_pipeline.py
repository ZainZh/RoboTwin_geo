import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


class RealZedCollectionPipelineTest(unittest.TestCase):
    def test_collect_module_parses_camera_labels_without_hardware_dependencies(self):
        from script.real_zed_collection.collect_zed_robotwin_raw import parse_camera_labels

        self.assertEqual(parse_camera_labels("global,ego,side"), ["global", "ego", "side"])
        self.assertEqual(parse_camera_labels(["global", "ego", "side"]), ["global", "ego", "side"])

    def test_postprocess_writes_robotwin_hdf5_from_raw_episode_and_masks(self):
        from script.real_zed_collection.postprocess_raw_to_robotwin_hdf5 import postprocess_episode

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


if __name__ == "__main__":
    unittest.main()
