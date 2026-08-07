from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from object_pointcloud_utils import strip_zero_points
from utonia_feature_utils import (
    DEBUG_PLACEHOLDER_COLORS_RGB,
    DEFAULT_DEBUG_PLACEHOLDER_COLOR_RGB,
    _estimate_normals,
    _extract_support_features,
    _nearest_neighbor_features,
    _normalize_support_and_query,
)


def fallback_normals(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normals = np.zeros_like(centered, dtype=np.float32)
    valid = norms[:, 0] > 1e-8
    normals[valid] = centered[valid] / norms[valid]
    return normals.astype(np.float32)


def prepare_utonia_input_point_cloud(
    object_point_cloud: np.ndarray,
    *,
    placeholder: str = "",
    color_mode: str = "debug_placeholder",
) -> np.ndarray:
    cloud = np.asarray(object_point_cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"Expected point cloud [N,C>=3], got {cloud.shape}")
    if cloud.shape[1] < 6:
        padded = np.zeros((len(cloud), 6), dtype=np.float32)
        padded[:, : cloud.shape[1]] = cloud
        cloud = padded
    else:
        cloud = cloud[:, :6].copy()

    mode = str(color_mode)
    if mode == "stored":
        return cloud
    if mode == "stored_scaled":
        if len(cloud) > 0 and float(np.nanmax(cloud[:, 3:6])) <= 1.0:
            cloud[:, 3:6] *= 255.0
        return cloud
    if mode == "debug_placeholder":
        color = DEBUG_PLACEHOLDER_COLORS_RGB.get(
            str(placeholder), DEFAULT_DEBUG_PLACEHOLDER_COLOR_RGB
        )
        cloud[:, 3:6] = color[None, :]
        return cloud
    raise ValueError(f"Unsupported Utonia input color mode: {color_mode}")


@torch.no_grad()
def compute_utonia_features_at_queries(
    artifacts: Dict[str, object],
    object_point_cloud: np.ndarray,
    query_world_xyz: np.ndarray,
    *,
    placeholder: str = "",
    color_mode: str = "debug_placeholder",
    normal_mode: str = "fallback",
) -> np.ndarray:
    """Extract frozen Utonia features without changing the supplied query xyz."""
    queries = np.asarray(query_world_xyz, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError(f"Expected query_world_xyz [Q,3], got {queries.shape}")

    point_cloud = prepare_utonia_input_point_cloud(
        object_point_cloud,
        placeholder=str(placeholder),
        color_mode=str(color_mode),
    )
    point_cloud = strip_zero_points(point_cloud)
    feat_dim = int(artifacts["feature_dim"])
    if len(point_cloud) == 0:
        return np.concatenate(
            [queries.copy(), np.zeros((len(queries), feat_dim), dtype=np.float32)],
            axis=1,
        )

    support_xyz = point_cloud[:, :3].astype(np.float32)
    support_rgb = point_cloud[:, 3:6].astype(np.float32)
    if str(normal_mode) == "fallback":
        support_normals = fallback_normals(support_xyz)
    elif str(normal_mode) == "estimated":
        support_normals = _estimate_normals(support_xyz)
    else:
        raise ValueError(f"Unsupported Utonia normal mode: {normal_mode}")

    support_normed, query_normed = _normalize_support_and_query(
        support_xyz,
        queries,
        coord_scale=float(artifacts["coord_scale"]),
        center_shift_z=bool(artifacts["center_shift_z"]),
    )
    support_features = _extract_support_features(
        artifacts,
        support_normed=support_normed,
        support_colors=support_rgb,
        support_normals=support_normals,
    )
    query_features = _nearest_neighbor_features(
        support_points=support_normed,
        query_points=query_normed,
        support_features=support_features,
    )
    return np.concatenate([queries, query_features], axis=1).astype(np.float32)


def nearest_context_support(
    context_point_cloud: np.ndarray,
    query_world_xyz: np.ndarray,
    *,
    support_num_points: int,
) -> np.ndarray:
    """Recover an approximate positive support from a merged context for pipeline smoke only."""
    context = strip_zero_points(np.asarray(context_point_cloud, dtype=np.float32))
    queries = np.asarray(query_world_xyz, dtype=np.float32)
    if len(context) == 0 or int(support_num_points) <= 0:
        return np.zeros((0, 6), dtype=np.float32)
    dist2 = np.sum(
        (context[:, None, :3] - queries[None, :, :]) ** 2,
        axis=-1,
    )
    nearest_distance = np.min(dist2, axis=1)
    count = min(int(support_num_points), len(context))
    indices = np.argpartition(nearest_distance, kth=count - 1)[:count]
    return context[indices].astype(np.float32)
