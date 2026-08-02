from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def functional_pose_alignment_errors(
    object_position: np.ndarray,
    object_quaternion: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
) -> dict[str, object]:
    object_position = np.asarray(object_position, dtype=np.float64).reshape(3)
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    object_quaternion = np.asarray(object_quaternion, dtype=np.float64).reshape(4)
    target_quaternion = np.asarray(target_quaternion, dtype=np.float64).reshape(4)
    translation = object_position - target_position
    object_norm = float(np.linalg.norm(object_quaternion))
    target_norm = float(np.linalg.norm(target_quaternion))
    if object_norm <= 1e-8 or target_norm <= 1e-8:
        rotation_error_deg = float("inf")
    else:
        alignment = abs(
            float(np.dot(object_quaternion / object_norm, target_quaternion / target_norm))
        )
        rotation_error_deg = float(
            np.degrees(2.0 * np.arccos(np.clip(alignment, 0.0, 1.0)))
        )
    return {
        "translation_error_xyz_m": translation.tolist(),
        "translation_error_abs_xyz_m": np.abs(translation).tolist(),
        "translation_error_norm_m": float(np.linalg.norm(translation)),
        "rotation_error_deg": rotation_error_deg,
    }


def functional_pose_alignment_success(
    object_position: np.ndarray,
    object_quaternion: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    *,
    position_tolerance: Sequence[float],
    min_quaternion_alignment: float,
) -> bool:
    """Return whether two functional poses are aligned within component tolerances."""
    object_position = np.asarray(object_position, dtype=np.float64).reshape(3)
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    object_quaternion = np.asarray(object_quaternion, dtype=np.float64).reshape(4)
    target_quaternion = np.asarray(target_quaternion, dtype=np.float64).reshape(4)
    tolerance = np.asarray(position_tolerance, dtype=np.float64).reshape(3)

    object_norm = float(np.linalg.norm(object_quaternion))
    target_norm = float(np.linalg.norm(target_quaternion))
    if object_norm <= 1e-8 or target_norm <= 1e-8:
        return False

    quaternion_alignment = abs(
        float(np.dot(object_quaternion / object_norm, target_quaternion / target_norm))
    )
    position_error = np.abs(object_position - target_position)
    return bool(
        np.all(position_error < tolerance)
        and quaternion_alignment > float(min_quaternion_alignment)
    )
