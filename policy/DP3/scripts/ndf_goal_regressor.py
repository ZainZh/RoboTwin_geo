#!/usr/bin/env python3
"""Frozen-NDF features and a small object-relation regression head.

Only observed point clouds are model inputs.  Object IDs, simulator poses and
functional annotations are intentionally excluded from the feature API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ndf_feature_utils import load_ndf_model


RADIAL_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
GEOMETRY_EXTRA_DIM = 2 * (1 + len(RADIAL_QUANTILES))


def valid_xyz(point_cloud: np.ndarray) -> np.ndarray:
    value = np.asarray(point_cloud, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"point cloud must have shape (N, >=3), got {value.shape}")
    xyz = value[:, :3]
    mask = np.all(np.isfinite(xyz), axis=1) & ~np.isclose(xyz, 0.0).all(axis=1)
    xyz = xyz[mask]
    if len(xyz) < 8:
        raise ValueError(f"point cloud has only {len(xyz)} valid points")
    return xyz.astype(np.float32)


def normalize_xyz(point_cloud: np.ndarray) -> tuple[np.ndarray, float]:
    xyz = valid_xyz(point_cloud)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    # Radius-derived diameter is invariant to world rotation; an axis-aligned
    # extent would leak observed pose into the NDF vector-norm feature.
    radius = np.linalg.norm(centered, axis=1)
    scale = max(float(2.0 * np.quantile(radius, 0.98)), 1e-6)
    return (centered / scale).astype(np.float32), scale


def deterministic_resample(points: np.ndarray, target_num_points: int) -> np.ndarray:
    """Deterministic point-count normalization without an orientation frame."""

    value = np.asarray(points, dtype=np.float32)
    target = int(target_num_points)
    if target < 8:
        raise ValueError("target_num_points must be at least 8")
    if len(value) == target:
        return value
    if len(value) < target:
        repeats = int(np.ceil(float(target) / float(len(value))))
        return np.tile(value, (repeats, 1))[:target].astype(np.float32)

    center = value.mean(axis=0)
    selected = np.zeros((target,), dtype=np.int64)
    distance = np.full((len(value),), np.inf, dtype=np.float32)
    farthest = int(np.argmax(np.sum((value - center[None, :]) ** 2, axis=1)))
    for index in range(target):
        selected[index] = farthest
        squared = np.sum((value - value[farthest][None, :]) ** 2, axis=1)
        distance = np.minimum(distance, squared)
        farthest = int(np.argmax(distance))
    return value[selected].astype(np.float32)


def invariant_geometry_extras(
    object_pointcloud_a: np.ndarray,
    object_pointcloud_b: np.ndarray,
) -> np.ndarray:
    """Rotation/translation-invariant scale and radial summaries for A and B."""

    result: list[float] = []
    for point_cloud in (object_pointcloud_a, object_pointcloud_b):
        normalized, scale = normalize_xyz(point_cloud)
        radius = np.linalg.norm(normalized, axis=1)
        result.append(float(scale))
        result.extend(np.quantile(radius, RADIAL_QUANTILES).tolist())
    value = np.asarray(result, dtype=np.float32)
    if value.shape != (GEOMETRY_EXTRA_DIM,):
        raise RuntimeError(f"unexpected geometry-extra shape {value.shape}")
    return value


def rotation_to_6d_torch(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in 3x3, got {tuple(rotation.shape)}")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 6:
        raise ValueError(f"rotation 6D input must end in 6, got {tuple(value.shape)}")
    first = F.normalize(value[..., 0:3], dim=-1, eps=1e-6)
    second_raw = value[..., 3:6]
    second = F.normalize(
        second_raw - torch.sum(first * second_raw, dim=-1, keepdim=True) * first,
        dim=-1,
        eps=1e-6,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def goal_matrix_to_target(goal_t_a_from_b: np.ndarray) -> np.ndarray:
    matrix = np.asarray(goal_t_a_from_b, dtype=np.float32).reshape(4, 4)
    rotation = torch.from_numpy(matrix[:3, :3])
    rotation_6d = rotation_to_6d_torch(rotation).numpy()
    return np.concatenate((matrix[:3, 3], rotation_6d), axis=0).astype(np.float32)


def target_to_goal_matrix(target: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(np.asarray(target, dtype=np.float32).reshape(1, 9))
    rotation = rotation_6d_to_matrix(value[:, 3:9])[0].cpu().numpy()
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.astype(np.float64)
    result[:3, 3] = value[0, :3].cpu().numpy().astype(np.float64)
    return result


class GoalRelationRegressor(nn.Module):
    """MLP predicting actor-local ``T_A_from_B`` from invariant geometry."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        dimensions = [int(input_dim), *(int(value) for value in hidden_dims)]
        layers: list[nn.Module] = []
        for input_width, output_width in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                (
                    nn.Linear(input_width, output_width),
                    nn.LayerNorm(output_width),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                )
            )
        self.backbone = nn.Sequential(*layers)
        final_dim = dimensions[-1]
        self.head = nn.Linear(final_dim, 9)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[3] = 1.0
            self.head.bias[7] = 1.0

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(feature))


@dataclass(frozen=True)
class FrozenNdfFeatureConfig:
    latent_dim: int = 256
    target_num_points: int = 1024

    @property
    def output_dim(self) -> int:
        return int(self.latent_dim) + GEOMETRY_EXTRA_DIM


class FrozenNdfGeometryEncoder:
    """NDF shoe encoder followed by a rotation-invariant vector norm."""

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str | torch.device = "cuda:0",
        config: FrozenNdfFeatureConfig | None = None,
    ) -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        self.config = config or FrozenNdfFeatureConfig()
        self.model = load_ndf_model(
            checkpoint,
            dgcnn=False,
            device=self.device,
            latent_dim=int(self.config.latent_dim),
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def _prepare_a(self, point_clouds_a: Sequence[np.ndarray]) -> torch.Tensor:
        normalized = []
        for point_cloud in point_clouds_a:
            value, _ = normalize_xyz(point_cloud)
            normalized.append(
                deterministic_resample(value, int(self.config.target_num_points))
            )
        return torch.from_numpy(np.stack(normalized)).to(self.device)

    @torch.no_grad()
    def encode_batch(
        self,
        point_clouds_a: Sequence[np.ndarray],
        point_clouds_b: Sequence[np.ndarray],
    ) -> torch.Tensor:
        if len(point_clouds_a) != len(point_clouds_b):
            raise ValueError("A/B point-cloud batch lengths differ")
        if not point_clouds_a:
            return torch.empty((0, self.config.output_dim), device=self.device)
        latent = self.model.extract_latent({"point_cloud": self._prepare_a(point_clouds_a)})
        if latent.ndim != 3 or latent.shape[-1] != 3:
            raise RuntimeError(
                "expected an equivariant NDF latent with shape (B, C, 3), "
                f"got {tuple(latent.shape)}"
            )
        invariant_ndf = torch.linalg.vector_norm(latent, dim=-1)
        extras = np.stack(
            [
                invariant_geometry_extras(point_cloud_a, point_cloud_b)
                for point_cloud_a, point_cloud_b in zip(point_clouds_a, point_clouds_b)
            ]
        )
        extras_t = torch.from_numpy(extras).to(self.device)
        result = torch.cat((invariant_ndf, extras_t), dim=-1)
        if result.shape[-1] != self.config.output_dim:
            raise RuntimeError(
                f"feature width {result.shape[-1]} != expected {self.config.output_dim}"
            )
        return result

    def encode(
        self,
        object_pointcloud_a: np.ndarray,
        object_pointcloud_b: np.ndarray,
    ) -> torch.Tensor:
        return self.encode_batch([object_pointcloud_a], [object_pointcloud_b])[0]


def rotation_loss(prediction_6d: torch.Tensor, target_rotation: torch.Tensor) -> torch.Tensor:
    predicted_rotation = rotation_6d_to_matrix(prediction_6d)
    relative = predicted_rotation.transpose(-1, -2) @ target_rotation
    cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
    return (1.0 - cosine.clamp(-1.0, 1.0)).mean()


def rotation_error_deg(prediction_6d: torch.Tensor, target_rotation: torch.Tensor) -> torch.Tensor:
    predicted_rotation = rotation_6d_to_matrix(prediction_6d)
    relative = predicted_rotation.transpose(-1, -2) @ target_rotation
    cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


__all__ = (
    "FrozenNdfFeatureConfig",
    "FrozenNdfGeometryEncoder",
    "GEOMETRY_EXTRA_DIM",
    "GoalRelationRegressor",
    "goal_matrix_to_target",
    "invariant_geometry_extras",
    "normalize_xyz",
    "rotation_6d_to_matrix",
    "rotation_error_deg",
    "rotation_loss",
    "rotation_to_6d_torch",
    "target_to_goal_matrix",
)
