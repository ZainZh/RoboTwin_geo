import unittest
import sys
import tempfile
import json
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_SCRIPTS_ROOT = REPO_ROOT / "policy" / "DP3" / "scripts"
if str(DP3_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_SCRIPTS_ROOT))

from eef_action_utils import (
    action20_to_eef14,
    eef14_to_action20,
    episode_eef_state_action_arrays,
    joint14_to_eef14,
    load_world_from_base_transforms,
    pose6_to_matrix,
    matrix_to_pose6,
    validate_eef_dataset_frame,
)


class TestEefActionUtils(unittest.TestCase):
    def test_pose6_matrix_roundtrip_preserves_pose(self):
        pose = np.asarray([0.12, -0.22, 0.34, 0.11, -0.08, 0.21], dtype=np.float64)

        recovered = matrix_to_pose6(pose6_to_matrix(pose))

        self.assertTrue(np.allclose(recovered, pose, atol=1e-8))

    def test_eef14_action20_roundtrip_preserves_two_arm_pose_and_grippers(self):
        eef14 = np.asarray(
            [
                0.10,
                -0.20,
                0.30,
                0.01,
                0.02,
                0.03,
                0.75,
                -0.10,
                -0.25,
                0.28,
                -0.04,
                0.03,
                -0.02,
                0.25,
            ],
            dtype=np.float64,
        )

        action20 = eef14_to_action20(eef14)
        recovered = action20_to_eef14(action20)

        self.assertEqual(action20.shape, (20,))
        self.assertTrue(np.allclose(recovered, eef14, atol=1e-8))

    def test_episode_state_action_uses_current_state_and_next_control(self):
        joint_vectors = np.zeros((4, 14), dtype=np.float64)
        control_vectors = np.zeros((4, 14), dtype=np.float64)
        for idx in range(4):
            joint_vectors[idx, 0] = 0.01 * idx
            joint_vectors[idx, 6] = 0.1 * idx
            joint_vectors[idx, 13] = 1.0 - 0.1 * idx
            control_vectors[idx, 0] = 0.2 + 0.01 * idx
            control_vectors[idx, 6] = 0.5 + 0.1 * idx
            control_vectors[idx, 13] = 0.2 + 0.1 * idx

        states, actions = episode_eef_state_action_arrays(
            joint_vectors,
            control_vectors=control_vectors,
            t_world_from_left_base=np.eye(4),
            t_world_from_right_base=np.eye(4),
        )

        expected_first_state = joint14_to_eef14(joint_vectors[0], np.eye(4), np.eye(4))
        expected_first_action_eef = joint14_to_eef14(control_vectors[1], np.eye(4), np.eye(4))

        self.assertEqual(states.shape, (3, 14))
        self.assertEqual(actions.shape, (3, 20))
        self.assertTrue(np.allclose(states[0], expected_first_state, atol=1e-8))
        recovered_first_action_eef = action20_to_eef14(actions[0])
        self.assertTrue(np.allclose(recovered_first_action_eef[:3], expected_first_action_eef[:3], atol=1e-8))
        self.assertTrue(
            np.allclose(
                pose6_to_matrix(recovered_first_action_eef[:6]),
                pose6_to_matrix(expected_first_action_eef[:6]),
                atol=1e-6,
            )
        )
        self.assertTrue(np.allclose(recovered_first_action_eef[6:10], expected_first_action_eef[6:10], atol=1e-6))
        self.assertTrue(
            np.allclose(
                pose6_to_matrix(recovered_first_action_eef[7:13]),
                pose6_to_matrix(expected_first_action_eef[7:13]),
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(float(recovered_first_action_eef[13]), float(expected_first_action_eef[13]), places=7)

    def test_episode_state_action_prefers_real_eef_pose_base_when_available(self):
        joint_vectors = np.zeros((3, 14), dtype=np.float64)
        control_vectors = np.zeros((3, 14), dtype=np.float64)
        eef_pose_base = np.zeros((3, 12), dtype=np.float64)
        for idx in range(3):
            joint_vectors[idx, 6] = 0.1 * idx
            joint_vectors[idx, 13] = 0.9 - 0.1 * idx
            control_vectors[idx, 6] = 0.4 + 0.1 * idx
            control_vectors[idx, 13] = 0.2 + 0.1 * idx
            eef_pose_base[idx, :6] = np.asarray([idx, 0.1, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
            eef_pose_base[idx, 6:12] = np.asarray([-idx, -0.1, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)

        t_world_from_left = np.eye(4, dtype=np.float64)
        t_world_from_left[0, 3] = 10.0
        t_world_from_right = np.eye(4, dtype=np.float64)
        t_world_from_right[1, 3] = 20.0

        states, actions = episode_eef_state_action_arrays(
            joint_vectors,
            control_vectors=control_vectors,
            eef_pose_base=eef_pose_base,
            t_world_from_left_base=t_world_from_left,
            t_world_from_right_base=t_world_from_right,
        )

        recovered_action0 = action20_to_eef14(actions[0])
        self.assertEqual(states.shape, (2, 14))
        np.testing.assert_allclose(states[0, :3], np.asarray([10.0, 0.1, 0.2], dtype=np.float32))
        np.testing.assert_allclose(states[0, 7:10], np.asarray([0.0, 19.9, 0.3], dtype=np.float32))
        self.assertAlmostEqual(float(states[0, 6]), 0.0, places=6)
        self.assertAlmostEqual(float(states[0, 13]), 0.9, places=6)
        np.testing.assert_allclose(recovered_action0[:3], np.asarray([11.0, 0.1, 0.2], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(recovered_action0[7:10], np.asarray([-1.0, 19.9, 0.3], dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(recovered_action0[6]), 0.5, places=6)
        self.assertAlmostEqual(float(recovered_action0[13]), 0.3, places=6)

    def test_load_world_from_base_transforms_can_use_right_base_as_world(self):
        def translate_x(x):
            transform = np.eye(4, dtype=np.float64)
            transform[0, 3] = x
            return transform

        fake_camera_calib = {
            "left_cam": SimpleNamespace(t_world_from_cam=translate_x(1.0)),
            "right_cam": SimpleNamespace(t_world_from_cam=translate_x(3.0)),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            left_robot_camera = tmp_path / "left.yaml"
            right_robot_camera = tmp_path / "right.yaml"
            calibration_path = tmp_path / "three_camera_workspace_extrinsics.yaml"
            calibration_path.write_text("{}", encoding="utf-8")
            left_robot_camera.write_text(
                json.dumps({"camera_label": "left_cam", "t_camera_from_base": np.eye(4).tolist()}),
                encoding="utf-8",
            )
            right_robot_camera.write_text(
                json.dumps({"camera_label": "right_cam", "t_camera_from_base": np.eye(4).tolist()}),
                encoding="utf-8",
            )

            with patch(
                "script.real_zed_collection.real_zed_utils.load_three_zed_calibration",
                return_value=fake_camera_calib,
            ):
                t_right_from_left, t_right_from_right = load_world_from_base_transforms(
                    calibration_path=calibration_path,
                    frame_mode="right_base",
                    left_robot_camera_calibration_path=left_robot_camera,
                    right_robot_camera_calibration_path=right_robot_camera,
                )

        np.testing.assert_allclose(t_right_from_left, translate_x(-2.0), atol=1e-8)
        np.testing.assert_allclose(t_right_from_right, np.eye(4), atol=1e-8)

    def test_validate_eef_dataset_frame_rejects_mismatched_real_zed_output_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            load_dir = Path(tmp_dir)
            (load_dir / "real_zed_sam2_objpc_meta.json").write_text(
                json.dumps({"output_frame": "workspace"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "output_frame=workspace"):
                validate_eef_dataset_frame(
                    action_mode="eef_absolute6d",
                    eef_frame_mode="right_base",
                    load_dir=load_dir,
                )

    def test_validate_eef_dataset_frame_accepts_matching_right_base_output_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            load_dir = Path(tmp_dir)
            (load_dir / "real_zed_sam2_objpc_meta.json").write_text(
                json.dumps({"output_frame": "right_base"}),
                encoding="utf-8",
            )

            validate_eef_dataset_frame(
                action_mode="eef_absolute6d",
                eef_frame_mode="right_base",
                load_dir=load_dir,
            )


if __name__ == "__main__":
    unittest.main()
