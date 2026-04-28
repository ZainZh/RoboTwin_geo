import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory
from pathlib import Path

import cv2
import numpy as np

from script.real_zed_collection.calibrate_three_zed_extrinsics import load_collection_camera_mapping
from script.real_zed_collection.calibrate_robot_camera_apriltag import (
    _resolve_camera_serial,
    candidate_aruco_dictionary_names,
    invert_transform,
    resolve_aruco_dictionary_id,
    solve_robot_camera_calibration,
)


def _axis_angle_transform(axis, angle_rad, translation):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


class RobotCameraAprilTagCalibrationTest(unittest.TestCase):
    def test_loads_label_serial_mapping_from_collection_config(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "real_zed_collection.yaml"
            config.write_text(
                "camera_labels: global,left,right\n"
                "zed_serials: [38968158, 31021548, 37856216]\n",
                encoding="utf-8",
            )

            labels, serial_by_label = load_collection_camera_mapping(config)

        self.assertEqual(labels, ["global", "left", "right"])
        self.assertEqual(
            serial_by_label,
            {"global": 38968158, "left": 31021548, "right": 37856216},
        )

    def test_robot_camera_serial_prefers_collection_config_over_calibration_yaml(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection_config = root / "real_zed_collection.yaml"
            collection_config.write_text(
                "camera_labels: global,left,right\n"
                "zed_serials: [38968158, 31021548, 37856216]\n",
                encoding="utf-8",
            )
            calibration_yaml = root / "three_camera_charuco_extrinsics.yaml"
            calibration_yaml.write_text(
                "cameras:\n"
                "  global:\n"
                "    serial_number: 31021548\n",
                encoding="utf-8",
            )
            args = Namespace(
                zed_serial=0,
                collection_config=str(collection_config),
                calibration_path=str(calibration_yaml),
                camera_label="global",
            )

            self.assertEqual(_resolve_camera_serial(args), 38968158)

    def test_resolve_aruco_dictionary_aliases(self):
        self.assertEqual(resolve_aruco_dictionary_id("DICT_4X4_50"), cv2.aruco.DICT_4X4_50)
        self.assertEqual(resolve_aruco_dictionary_id("4x4_50"), cv2.aruco.DICT_4X4_50)
        self.assertEqual(resolve_aruco_dictionary_id("apriltag_36h11"), cv2.aruco.DICT_APRILTAG_36h11)
        with self.assertRaises(ValueError):
            resolve_aruco_dictionary_id("DICT_NOT_A_REAL_MARKER_SET")

    def test_auto_dictionary_expands_to_common_marker_families(self):
        names = candidate_aruco_dictionary_names("auto")
        self.assertIn("DICT_4X4_50", names)
        self.assertIn("DICT_5X5_50", names)
        self.assertIn("DICT_6X6_50", names)
        self.assertIn("DICT_APRILTAG_36H11", names)
        self.assertEqual(candidate_aruco_dictionary_names("4x4_50"), ["DICT_4X4_50"])

    def test_solve_recovers_base_from_camera_for_moving_tag(self):
        camera_from_base = _axis_angle_transform([0.3, -0.2, 1.0], 0.55, [0.45, -0.25, 0.8])
        tag_from_gripper = _axis_angle_transform([1.0, 0.2, -0.4], -0.35, [0.02, 0.0, 0.09])
        samples = []
        axes = [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.4],
            [0.2, 0.1, 1.0],
            [1.0, 1.0, 0.0],
            [-0.2, 1.0, 0.7],
            [0.7, -0.3, 1.0],
        ]
        for idx, axis in enumerate(axes):
            base_from_gripper = _axis_angle_transform(
                axis,
                0.2 + 0.13 * idx,
                [0.2 + 0.03 * idx, -0.35 + 0.02 * idx, 0.18 + 0.04 * idx],
            )
            camera_from_tag = camera_from_base @ base_from_gripper @ invert_transform(tag_from_gripper)
            samples.append(
                {
                    "base_from_gripper": base_from_gripper,
                    "camera_from_tag": camera_from_tag,
                }
            )

        result = solve_robot_camera_calibration(samples)

        np.testing.assert_allclose(result["camera_from_base"], camera_from_base, atol=1e-6)
        np.testing.assert_allclose(result["base_from_camera"], invert_transform(camera_from_base), atol=1e-6)
        np.testing.assert_allclose(result["tag_from_gripper"], tag_from_gripper, atol=1e-6)
        self.assertLess(result["mean_translation_error_m"], 1e-6)
        self.assertLess(result["mean_rotation_error_deg"], 1e-5)


if __name__ == "__main__":
    unittest.main()
