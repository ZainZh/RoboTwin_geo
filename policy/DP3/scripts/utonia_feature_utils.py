import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from object_pointcloud_utils import resample_point_cloud, strip_zero_points


REPO_ROOT = Path(__file__).resolve().parents[3]
SEM_ROOT = REPO_ROOT / "include" / "3d_semantic_train"
if SEM_ROOT.exists() and str(SEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SEM_ROOT))

from my_datasets.partnext_canonical_field import UTONIA  # noqa: E402
from models.canonical_field.utonia_feature_extractor import UtoniaFeatureExtractor  # noqa: E402


def _build_utonia_transform(grid_size: float):
    return UTONIA.transform.Compose(
        [
            dict(
                type="GridSample",
                grid_size=float(grid_size),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
    )


def _normalize_support_and_query(
    support_points: np.ndarray,
    query_points: np.ndarray,
    *,
    coord_scale: float,
    center_shift_z: bool,
):
    centroid = np.mean(support_points, axis=0, keepdims=True)
    support = support_points - centroid
    query = query_points - centroid

    radius = np.linalg.norm(support, axis=1).max()
    radius = max(float(radius), 1e-6)
    support = support / radius
    query = query / radius

    if center_shift_z:
        x_min, y_min, z_min = support.min(axis=0)
        x_max, y_max, _ = support.max(axis=0)
        shift = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min],
            dtype=np.float32,
        )
        support = support - shift
        query = query - shift

    return (support * float(coord_scale)).astype(np.float32), (query * float(coord_scale)).astype(np.float32)


def _estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return np.zeros_like(points, dtype=np.float32)

    k = max(3, min(int(k), len(points) - 1))
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    neighbor_idx = np.argpartition(dist2, kth=k, axis=1)[:, :k]

    normals = np.zeros_like(points, dtype=np.float32)
    for idx in range(len(points)):
        neighbors = points[neighbor_idx[idx]]
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(neighbors), 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        normal = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        normals[idx] = normal
    return normals.astype(np.float32)


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _build_utonia_input(
    transform,
    support_points: np.ndarray,
    support_colors: np.ndarray,
    support_normals: np.ndarray,
):
    utonia_input = transform(
        {
            "coord": support_points.copy(),
            "color": support_colors.copy(),
            "normal": support_normals.copy(),
        }
    )
    return UTONIA.data.collate_fn([utonia_input])


@torch.no_grad()
def _extract_support_features(
    artifacts: Dict[str, object],
    support_normed: np.ndarray,
    support_colors: np.ndarray,
    support_normals: np.ndarray,
) -> np.ndarray:
    device = artifacts["device"]
    utonia_input = _build_utonia_input(
        artifacts["utonia_transform"],
        support_normed,
        support_colors,
        support_normals,
    )
    utonia_input = _move_to_device(utonia_input, device)
    features = artifacts["extractor"](
        utonia_input=utonia_input,
        num_support_points=int(support_normed.shape[0]),
    )
    return features.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _nearest_neighbor_features(
    support_points: np.ndarray,
    query_points: np.ndarray,
    support_features: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    support_points = np.asarray(support_points, dtype=np.float32)
    query_points = np.asarray(query_points, dtype=np.float32)
    support_features = np.asarray(support_features, dtype=np.float32)
    result = np.zeros((len(query_points), support_features.shape[1]), dtype=np.float32)
    for start in range(0, len(query_points), int(chunk_size)):
        stop = min(start + int(chunk_size), len(query_points))
        chunk = query_points[start:stop]
        dist2 = np.sum((chunk[:, None, :] - support_points[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(dist2, axis=1)
        result[start:stop] = support_features[nearest]
    return result


def load_utonia_model(
    *,
    device: torch.device,
    checkpoint: str = "auto",
    repo_id: str = "Pointcept/Utonia",
    upcast_levels: int = 0,
    grid_size: float = 0.01,
    coord_scale: float = 1.0,
    center_shift_z: bool = True,
) -> Dict[str, object]:
    extractor = UtoniaFeatureExtractor(
        checkpoint=checkpoint,
        repo_id=repo_id,
        upcast_levels=int(upcast_levels),
        freeze_encoder=True,
    )
    extractor = extractor.to(device)
    extractor.eval()

    artifacts = {
        "checkpoint": checkpoint,
        "repo_id": repo_id,
        "upcast_levels": int(upcast_levels),
        "device": device,
        "extractor": extractor,
        "utonia_transform": _build_utonia_transform(float(grid_size)),
        "coord_scale": float(coord_scale),
        "center_shift_z": bool(center_shift_z),
    }

    dummy_support = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    dummy_colors = np.zeros((4, 3), dtype=np.float32)
    dummy_normals = _estimate_normals(dummy_support)
    support_normed, _ = _normalize_support_and_query(
        dummy_support,
        dummy_support,
        coord_scale=float(coord_scale),
        center_shift_z=bool(center_shift_z),
    )
    feature_dim = int(
        _extract_support_features(
            artifacts,
            support_normed=support_normed,
            support_colors=dummy_colors,
            support_normals=dummy_normals,
        ).shape[1]
    )
    artifacts["feature_dim"] = feature_dim
    return artifacts


@torch.no_grad()
def compute_utonia_pointwise_cloud(
    artifacts: Dict[str, object],
    object_point_cloud: np.ndarray,
    *,
    target_num_points: int,
) -> np.ndarray:
    point_cloud = strip_zero_points(np.asarray(object_point_cloud, dtype=np.float32))
    feat_dim = int(artifacts["feature_dim"])
    if len(point_cloud) == 0:
        return np.zeros((int(target_num_points), 3 + feat_dim), dtype=np.float32)

    support_pc = point_cloud.astype(np.float32)
    query_pc = resample_point_cloud(point_cloud, int(target_num_points))

    support_xyz = support_pc[:, :3].astype(np.float32)
    support_rgb = (
        support_pc[:, 3:6].astype(np.float32)
        if support_pc.shape[1] >= 6
        else np.zeros((len(support_pc), 3), dtype=np.float32)
    )
    support_normals = _estimate_normals(support_xyz)
    query_world_xyz = query_pc[:, :3].astype(np.float32)

    support_normed, query_normed = _normalize_support_and_query(
        support_xyz,
        query_world_xyz,
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
    return np.concatenate([query_world_xyz, query_features], axis=1).astype(np.float32)
