from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


# An asymmetric virtual gripper frame.  The long +X point is the approach
# direction and the +/-Y points encode the finger closing direction.  Using a
# frame rather than a symmetric box prevents the descriptor query from losing
# approach sign, which happened in the old NDF grasp experiment.
GRIPPER_KEYPOINTS_METERS = np.asarray(
    [
        [0.000, 0.000, 0.000],
        [0.070, 0.000, 0.000],
        [-0.025, 0.000, 0.000],
        [0.010, 0.040, 0.000],
        [0.010, -0.040, 0.000],
        [0.015, 0.000, 0.035],
        [0.015, 0.000, -0.020],
    ],
    dtype=np.float32,
)


def pose7_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(-1)
    if value.shape != (7,):
        raise ValueError(f"expected xyz+wxyz pose with shape (7,), got {value.shape}")
    quaternion = value[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("pose quaternion must be non-zero")
    w, x, y, z = quaternion / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def matrix_to_pose7_wxyz(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"expected transform with shape (4,4), got {value.shape}")
    x, y, z, w = Rotation.from_matrix(value[:3, :3]).as_quat()
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return np.concatenate((value[:3, 3], quaternion))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    xyz = np.asarray(points, dtype=np.float64)
    if matrix.shape != (4, 4) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"invalid transform/point shapes: {matrix.shape}, {xyz.shape}")
    return (xyz @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float64)


def normalize_object_points(
    points: torch.Tensor, *, minimum_scale: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize XYZ without ever treating similarity scale as an SE(3)."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"expected (B,N,3) points, got {tuple(points.shape)}")
    lower = torch.amin(points, dim=1)
    upper = torch.amax(points, dim=1)
    center = 0.5 * (lower + upper)
    scale = torch.amax(upper - lower, dim=-1, keepdim=True)
    scale = torch.clamp(scale, min=float(minimum_scale))
    normalized = (points - center[:, None, :]) / scale[:, None, :]
    return normalized, center, scale


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    vectors = value.reshape(*value.shape[:-1], 3, 2)
    first = F.normalize(vectors[..., 0], dim=-1, eps=1e-8)
    second_raw = vectors[..., 1]
    second = F.normalize(
        second_raw - torch.sum(first * second_raw, dim=-1, keepdim=True) * first,
        dim=-1,
        eps=1e-8,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def weighted_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``target = source @ R.T + t`` with a proper rotation.

    ``source`` and ``target`` may be ``(K,3)`` or batched ``(...,K,3)``.
    The returned rotation has shape ``(...,3,3)`` and determinant +1.
    """
    if source.shape != target.shape or source.shape[-1] != 3:
        raise ValueError(
            f"source and target must have identical (...,K,3) shapes, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.shape[-2] < 3:
        raise ValueError("Kabsch requires at least three keypoints")
    if weights is None:
        weights = torch.ones_like(source[..., 0])
    if weights.shape != source.shape[:-1]:
        raise ValueError(
            f"weights must have shape {tuple(source.shape[:-1])}, got {tuple(weights.shape)}"
        )
    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-8)
    source_center = torch.sum(source * weights[..., None], dim=-2)
    target_center = torch.sum(target * weights[..., None], dim=-2)
    source_zero = source - source_center[..., None, :]
    target_zero = target - target_center[..., None, :]
    covariance = source_zero.transpose(-1, -2) @ (target_zero * weights[..., None])
    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-1, -2)
    correction = torch.ones_like(source_center)
    correction[..., -1] = torch.where(
        torch.det(v @ u.transpose(-1, -2)) < 0.0,
        -torch.ones_like(correction[..., -1]),
        torch.ones_like(correction[..., -1]),
    )
    rotation = v @ torch.diag_embed(correction) @ u.transpose(-1, -2)
    translation = target_center - (rotation @ source_center[..., None]).squeeze(-1)
    return rotation, translation


def keypoints_to_transform(
    target_keypoints: torch.Tensor,
    canonical_keypoints: torch.Tensor | None = None,
) -> torch.Tensor:
    if canonical_keypoints is None:
        canonical_keypoints = torch.as_tensor(
            GRIPPER_KEYPOINTS_METERS,
            dtype=target_keypoints.dtype,
            device=target_keypoints.device,
        )
    if canonical_keypoints.ndim == 2:
        canonical_keypoints = canonical_keypoints.expand(
            *target_keypoints.shape[:-2], -1, -1
        )
    rotation, translation = weighted_kabsch(canonical_keypoints, target_keypoints)
    result = torch.zeros(
        (*target_keypoints.shape[:-2], 4, 4),
        dtype=target_keypoints.dtype,
        device=target_keypoints.device,
    )
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
    result[..., 3, 3] = 1.0
    return result


def transform_keypoints_torch(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    canonical_keypoints: torch.Tensor | None = None,
) -> torch.Tensor:
    if canonical_keypoints is None:
        canonical_keypoints = torch.as_tensor(
            GRIPPER_KEYPOINTS_METERS,
            dtype=rotation.dtype,
            device=rotation.device,
        )
    return canonical_keypoints @ rotation.transpose(-1, -2) + translation[..., None, :]


def rotation_angle_degrees(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted @ target.transpose(-1, -2)
    trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))
