"""Deterministic geometry-conditioned interaction models.

This package is intentionally separate from ``GeometryResidual`` and
``GeometryServo``.  Despite the name, it does not implement generative flow
matching: "flow" here means explicit object/gripper keypoint trajectories.
"""

from .geometry import (
    GRIPPER_KEYPOINTS_METERS,
    matrix_to_pose7_wxyz,
    pose7_wxyz_to_matrix,
    transform_points,
    weighted_kabsch,
)

__all__ = [
    "GRIPPER_KEYPOINTS_METERS",
    "matrix_to_pose7_wxyz",
    "pose7_wxyz_to_matrix",
    "transform_points",
    "weighted_kabsch",
]
