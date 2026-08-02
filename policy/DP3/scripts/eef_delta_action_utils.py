from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


EEF_DELTA_ACTION_MODE = "eef_delta_rotvec"
EEF_STATE_DIM = 20
EEF_DELTA_ACTION_DIM = 14


def _pose7_wxyz(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(-1)
    if value.size != 7:
        raise ValueError(f"Expected xyz + quaternion(wxyz), got shape {np.asarray(pose).shape}")
    quaternion = value[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Pose quaternion must be non-zero")
    value = value.copy()
    value[3:] = quaternion / norm
    return value


def pose7_wxyz_to_rotation(pose: np.ndarray) -> Rotation:
    value = _pose7_wxyz(pose)
    w, x, y, z = value[3:]
    return Rotation.from_quat([x, y, z, w])


def rotation_to_quaternion_wxyz(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def rotation_to_sixd(rotation: Rotation) -> np.ndarray:
    return rotation.as_matrix()[:, :2].reshape(6).astype(np.float64)


def eef_state20(
    left_pose7: np.ndarray,
    left_gripper: float,
    right_pose7: np.ndarray,
    right_gripper: float,
) -> np.ndarray:
    left = _pose7_wxyz(left_pose7)
    right = _pose7_wxyz(right_pose7)
    return np.concatenate(
        [
            left[:3],
            rotation_to_sixd(pose7_wxyz_to_rotation(left)),
            [float(left_gripper)],
            right[:3],
            rotation_to_sixd(pose7_wxyz_to_rotation(right)),
            [float(right_gripper)],
        ]
    ).astype(np.float32)


def eef_delta_action14(
    left_pose7: np.ndarray,
    left_gripper: float,
    right_pose7: np.ndarray,
    right_gripper: float,
    next_left_pose7: np.ndarray,
    next_left_gripper: float,
    next_right_pose7: np.ndarray,
    next_right_gripper: float,
) -> np.ndarray:
    left = _pose7_wxyz(left_pose7)
    right = _pose7_wxyz(right_pose7)
    next_left = _pose7_wxyz(next_left_pose7)
    next_right = _pose7_wxyz(next_right_pose7)
    left_delta_rotation = pose7_wxyz_to_rotation(next_left) * pose7_wxyz_to_rotation(left).inv()
    right_delta_rotation = pose7_wxyz_to_rotation(next_right) * pose7_wxyz_to_rotation(right).inv()
    return np.concatenate(
        [
            next_left[:3] - left[:3],
            left_delta_rotation.as_rotvec(),
            [float(next_left_gripper)],
            next_right[:3] - right[:3],
            right_delta_rotation.as_rotvec(),
            [float(next_right_gripper)],
        ]
    ).astype(np.float32)


def decode_eef_delta_action16(
    action14: np.ndarray,
    left_pose7: np.ndarray,
    right_pose7: np.ndarray,
    *,
    max_translation_delta_m: float | None = None,
    max_rotation_delta_rad: float | None = None,
) -> np.ndarray:
    action = np.asarray(action14, dtype=np.float64).reshape(-1)
    if action.size != EEF_DELTA_ACTION_DIM:
        raise ValueError(f"Expected {EEF_DELTA_ACTION_DIM}D EEF-delta action, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("EEF-delta action contains non-finite values")
    left = _pose7_wxyz(left_pose7)
    right = _pose7_wxyz(right_pose7)

    def decode_arm(offset: int, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        translation_delta = action[offset : offset + 3].copy()
        rotation_delta = action[offset + 3 : offset + 6].copy()
        if max_translation_delta_m is not None and float(max_translation_delta_m) > 0.0:
            norm = float(np.linalg.norm(translation_delta))
            if norm > float(max_translation_delta_m):
                translation_delta *= float(max_translation_delta_m) / norm
        if max_rotation_delta_rad is not None and float(max_rotation_delta_rad) > 0.0:
            norm = float(np.linalg.norm(rotation_delta))
            if norm > float(max_rotation_delta_rad):
                rotation_delta *= float(max_rotation_delta_rad) / norm
        target_rotation = Rotation.from_rotvec(rotation_delta) * pose7_wxyz_to_rotation(current)
        return (
            current[:3] + translation_delta,
            rotation_to_quaternion_wxyz(target_rotation),
            float(action[offset + 6]),
        )

    left_position, left_quaternion, left_gripper = decode_arm(0, left)
    right_position, right_quaternion, right_gripper = decode_arm(7, right)
    return np.concatenate(
        [
            left_position,
            left_quaternion,
            [left_gripper],
            right_position,
            right_quaternion,
            [right_gripper],
        ]
    ).astype(np.float32)


def episode_eef_delta_arrays(episode: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    endpose = episode.get("endpose_world")
    if not isinstance(endpose, dict):
        raise RuntimeError(
            "eef_delta_rotvec requires legacy simulator endpose datasets under /endpose"
        )
    required = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
    missing = [key for key in required if endpose.get(key) is None]
    if missing:
        raise RuntimeError(f"Missing simulator endpose arrays: {missing}")
    left_pose = np.asarray(endpose["left_endpose"], dtype=np.float64)
    right_pose = np.asarray(endpose["right_endpose"], dtype=np.float64)
    left_gripper = np.asarray(endpose["left_gripper"], dtype=np.float64).reshape(-1)
    right_gripper = np.asarray(endpose["right_gripper"], dtype=np.float64).reshape(-1)
    frame_count = int(left_pose.shape[0])
    if left_pose.shape != (frame_count, 7) or right_pose.shape != (frame_count, 7):
        raise ValueError(
            f"Expected end-effector poses shaped (T,7), got {left_pose.shape} and {right_pose.shape}"
        )
    if left_gripper.shape != (frame_count,) or right_gripper.shape != (frame_count,):
        raise ValueError("End-effector pose and gripper arrays have different frame counts")
    states = np.asarray(
        [
            eef_state20(left_pose[idx], left_gripper[idx], right_pose[idx], right_gripper[idx])
            for idx in range(frame_count - 1)
        ],
        dtype=np.float32,
    )
    actions = np.asarray(
        [
            eef_delta_action14(
                left_pose[idx],
                left_gripper[idx],
                right_pose[idx],
                right_gripper[idx],
                left_pose[idx + 1],
                left_gripper[idx + 1],
                right_pose[idx + 1],
                right_gripper[idx + 1],
            )
            for idx in range(frame_count - 1)
        ],
        dtype=np.float32,
    )
    return states, actions

