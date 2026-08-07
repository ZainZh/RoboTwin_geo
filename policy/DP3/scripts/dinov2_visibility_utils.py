"""Pure NumPy geometry helpers for visibility-aware 2D-to-3D feature lifting.

This module deliberately contains no DINOv2/model-loading code.  It defines the
camera/depth contract needed by a lifting baseline and can therefore be tested
without downloading model weights or requiring a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VisibilityResult:
    pixels_uv: np.ndarray
    camera_depth_m: np.ndarray
    observed_depth_m: np.ndarray
    depth_error_m: np.ndarray
    in_frame: np.ndarray
    depth_valid: np.ndarray
    foreground_valid: np.ndarray
    zbuffer_visible: np.ndarray
    visible: np.ndarray
    confidence_weight: np.ndarray


def _as_image_hw(image_shape_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(image_shape_hw[0]), int(image_shape_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape_hw must be positive, got {(height, width)}")
    return height, width


def project_xyz_to_camera(
    xyz_output: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    t_cam_from_output: np.ndarray,
    image_shape_hw: tuple[int, int],
    min_camera_depth_m: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project output-frame XYZ to image pixels.

    Returns ``(pixels_uv, camera_depth_m, in_frame)``.  ``in_frame`` is false
    for non-finite points, points behind the camera, and out-of-image pixels.
    """

    xyz = np.asarray(xyz_output, dtype=np.float32).reshape(-1, 3)
    k = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
    transform = np.asarray(t_cam_from_output, dtype=np.float32).reshape(4, 4)
    height, width = _as_image_hw(image_shape_hw)

    homogeneous = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
    )
    xyz_camera = (transform @ homogeneous.T).T[:, :3]
    z = xyz_camera[:, 2]
    finite = np.isfinite(xyz_camera).all(axis=1) & np.isfinite(z)
    positive_depth = finite & (z > float(min_camera_depth_m))
    safe_z = np.where(positive_depth, z, 1.0)

    projected = (k @ xyz_camera.T).T
    u = projected[:, 0] / safe_z
    v = projected[:, 1] / safe_z
    pixels = np.stack([u, v], axis=1).astype(np.float32, copy=False)
    in_frame = (
        positive_depth
        & np.isfinite(pixels).all(axis=1)
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )
    return pixels, z.astype(np.float32, copy=False), in_frame


def nearest_pixel_indices(
    pixels_uv: np.ndarray,
    *,
    image_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map continuous ``(u, v)`` pixels to clipped nearest integer indices."""

    height, width = _as_image_hw(image_shape_hw)
    pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
    x = np.floor(pixels[:, 0] + 0.5).astype(np.int64)
    y = np.floor(pixels[:, 1] + 0.5).astype(np.int64)
    return np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)


def query_zbuffer_visibility(
    pixels_uv: np.ndarray,
    camera_depth_m: np.ndarray,
    *,
    valid: np.ndarray,
    image_shape_hw: tuple[int, int],
    tolerance_m: float = 0.005,
) -> np.ndarray:
    """Keep only front-most query points at each projected image pixel."""

    height, width = _as_image_hw(image_shape_hw)
    pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
    depth = np.asarray(camera_depth_m, dtype=np.float32).reshape(-1)
    valid_mask = np.asarray(valid, dtype=bool).reshape(-1)
    if not (pixels.shape[0] == depth.shape[0] == valid_mask.shape[0]):
        raise ValueError("pixels, camera depth, and valid mask must have the same length")

    y, x = nearest_pixel_indices(pixels, image_shape_hw=(height, width))
    flat = y * width + x
    nearest = np.full(height * width, np.inf, dtype=np.float32)
    selected = valid_mask & np.isfinite(depth) & (depth > 0.0)
    np.minimum.at(nearest, flat[selected], depth[selected])
    result = np.zeros(depth.shape[0], dtype=bool)
    result[selected] = depth[selected] <= (
        nearest[flat[selected]] + max(float(tolerance_m), 0.0)
    )
    return result


def compute_depth_visibility(
    xyz_output: np.ndarray,
    *,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    t_cam_from_output: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
    absolute_tolerance_m: float = 0.02,
    relative_tolerance: float = 0.01,
    query_zbuffer_tolerance_m: float = 0.005,
) -> VisibilityResult:
    """Evaluate in-frame, mask, RGB-D consistency, and query z-buffer visibility.

    A point is visible only when its camera depth agrees with the measured depth
    at the projected pixel within ``max(absolute_tolerance_m,
    relative_tolerance * measured_depth)``.  Invalid/zero/NaN depth is rejected.
    ``foreground_mask`` is optional because a matched object query cloud is
    already produced from SAM2, but can be supplied for an additional mask check.
    """

    depth_image = np.asarray(depth_m, dtype=np.float32)
    if depth_image.ndim != 2:
        raise ValueError(f"depth_m must be a 2-D image, got {depth_image.shape}")
    image_shape = (int(depth_image.shape[0]), int(depth_image.shape[1]))
    pixels, camera_depth, in_frame = project_xyz_to_camera(
        xyz_output,
        camera_matrix=camera_matrix,
        t_cam_from_output=t_cam_from_output,
        image_shape_hw=image_shape,
    )
    y, x = nearest_pixel_indices(pixels, image_shape_hw=image_shape)
    observed_depth = depth_image[y, x]
    depth_valid = (
        in_frame
        & np.isfinite(observed_depth)
        & (observed_depth >= float(min_depth_m))
        & (observed_depth <= float(max_depth_m))
    )

    if foreground_mask is None:
        foreground_valid = np.ones(camera_depth.shape[0], dtype=bool)
    else:
        mask = np.asarray(foreground_mask)
        if mask.shape != depth_image.shape:
            raise ValueError(
                f"foreground mask/depth shape mismatch: {mask.shape} vs {depth_image.shape}"
            )
        foreground_valid = in_frame & mask[y, x].astype(bool)

    depth_error = np.abs(camera_depth - observed_depth).astype(np.float32, copy=False)
    tolerance = np.maximum(
        float(absolute_tolerance_m),
        float(relative_tolerance) * observed_depth,
    ).astype(np.float32, copy=False)
    depth_consistent = depth_valid & (depth_error <= tolerance)
    candidate = in_frame & foreground_valid & depth_consistent
    zbuffer_visible = query_zbuffer_visibility(
        pixels,
        camera_depth,
        valid=candidate,
        image_shape_hw=image_shape,
        tolerance_m=float(query_zbuffer_tolerance_m),
    )
    visible = candidate & zbuffer_visible

    safe_tolerance = np.maximum(tolerance, np.finfo(np.float32).eps)
    normalized_error = depth_error / safe_tolerance
    confidence = np.exp(-0.5 * np.square(normalized_error)) / np.maximum(
        np.square(camera_depth), 1e-6
    )
    confidence = np.where(visible, confidence, 0.0).astype(np.float32, copy=False)
    return VisibilityResult(
        pixels_uv=pixels,
        camera_depth_m=camera_depth,
        observed_depth_m=observed_depth.astype(np.float32, copy=False),
        depth_error_m=depth_error,
        in_frame=in_frame,
        depth_valid=depth_valid,
        foreground_valid=foreground_valid,
        zbuffer_visible=zbuffer_visible,
        visible=visible,
        confidence_weight=confidence,
    )


def sample_feature_grid_bilinear(
    feature_grid: np.ndarray,
    pixels_uv: np.ndarray,
    *,
    image_shape_hw: tuple[int, int],
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Bilinearly sample an ``Hf x Wf x D`` feature grid at image pixels."""

    grid = np.asarray(feature_grid, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError(f"feature_grid must have shape [Hf,Wf,D], got {grid.shape}")
    image_h, image_w = _as_image_hw(image_shape_hw)
    grid_h, grid_w, feature_dim = grid.shape
    pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
    sample_valid = np.isfinite(pixels).all(axis=1)
    sample_valid &= (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(image_w))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(image_h))
    )
    if valid is not None:
        provided = np.asarray(valid, dtype=bool).reshape(-1)
        if provided.shape[0] != pixels.shape[0]:
            raise ValueError("valid mask and pixels must have the same length")
        sample_valid &= provided

    gx = (pixels[:, 0] + 0.5) * float(grid_w) / float(image_w) - 0.5
    gy = (pixels[:, 1] + 0.5) * float(grid_h) / float(image_h) - 0.5
    gx = np.clip(gx, 0.0, float(max(grid_w - 1, 0)))
    gy = np.clip(gy, 0.0, float(max(grid_h - 1, 0)))
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = np.minimum(x0 + 1, grid_w - 1)
    y1 = np.minimum(y0 + 1, grid_h - 1)
    wx = (gx - x0).astype(np.float32)
    wy = (gy - y0).astype(np.float32)

    sampled = (
        grid[y0, x0] * ((1.0 - wx) * (1.0 - wy))[:, None]
        + grid[y0, x1] * (wx * (1.0 - wy))[:, None]
        + grid[y1, x0] * ((1.0 - wx) * wy)[:, None]
        + grid[y1, x1] * (wx * wy)[:, None]
    )
    sampled = sampled.astype(np.float32, copy=False)
    sampled[~sample_valid] = np.zeros((feature_dim,), dtype=np.float32)
    return sampled


def fuse_multiview_features(
    features_by_view: Sequence[np.ndarray],
    visible_by_view: Sequence[np.ndarray],
    *,
    weights_by_view: Sequence[np.ndarray] | None = None,
    l2_normalize_inputs: bool = False,
    l2_normalize_output: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse per-view point features with visibility-aware weighted averaging.

    Returns ``(fused, valid, weight_sum, visible_view_count)``.  Points unseen in
    every view receive an all-zero feature rather than a fabricated mean feature.
    """

    if not features_by_view:
        raise ValueError("features_by_view must contain at least one view")
    if len(features_by_view) != len(visible_by_view):
        raise ValueError("features_by_view and visible_by_view lengths differ")
    if weights_by_view is not None and len(weights_by_view) != len(features_by_view):
        raise ValueError("weights_by_view and features_by_view lengths differ")

    feature_arrays = [np.asarray(item, dtype=np.float32) for item in features_by_view]
    first_shape = feature_arrays[0].shape
    if len(first_shape) != 2 or any(item.shape != first_shape for item in feature_arrays):
        raise ValueError("all feature arrays must share shape [N,D]")
    point_count, feature_dim = first_shape
    weighted_sum = np.zeros((point_count, feature_dim), dtype=np.float32)
    weight_sum = np.zeros(point_count, dtype=np.float32)
    view_count = np.zeros(point_count, dtype=np.int32)

    for view_index, feature in enumerate(feature_arrays):
        visible = np.asarray(visible_by_view[view_index], dtype=bool).reshape(-1)
        if visible.shape[0] != point_count:
            raise ValueError("visibility mask and feature point count differ")
        finite = np.isfinite(feature).all(axis=1)
        visible = visible & finite
        if weights_by_view is None:
            weight = np.ones(point_count, dtype=np.float32)
        else:
            weight = np.asarray(weights_by_view[view_index], dtype=np.float32).reshape(-1)
            if weight.shape[0] != point_count:
                raise ValueError("weight and feature point count differ")
        weight = np.where(visible & np.isfinite(weight) & (weight > 0.0), weight, 0.0)

        current = feature
        if l2_normalize_inputs:
            norm = np.linalg.norm(current, axis=1, keepdims=True)
            current = current / np.maximum(norm, 1e-12)
        weighted_sum += current * weight[:, None]
        weight_sum += weight
        view_count += (weight > 0.0).astype(np.int32)

    valid = weight_sum > 0.0
    fused = np.zeros_like(weighted_sum)
    fused[valid] = weighted_sum[valid] / weight_sum[valid, None]
    if l2_normalize_output:
        norm = np.linalg.norm(fused, axis=1, keepdims=True)
        fused[valid] /= np.maximum(norm[valid], 1e-12)
    return fused, valid, weight_sum, view_count


__all__ = [
    "VisibilityResult",
    "compute_depth_visibility",
    "fuse_multiview_features",
    "nearest_pixel_indices",
    "project_xyz_to_camera",
    "query_zbuffer_visibility",
    "sample_feature_grid_bilinear",
]
