from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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

