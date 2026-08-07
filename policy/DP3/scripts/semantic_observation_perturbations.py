from __future__ import annotations

import numpy as np


def _validate_arrays(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_values = np.asarray(points, dtype=np.float32)
    normal_values = np.asarray(normals, dtype=np.float32)
    color_values = np.asarray(colors, dtype=np.float32)
    if point_values.ndim != 2 or point_values.shape[1] != 3 or len(point_values) < 8:
        raise ValueError(f"points must have shape [N, 3] with N >= 8, got {point_values.shape}")
    if normal_values.shape != point_values.shape or color_values.shape != point_values.shape:
        raise ValueError("points, normals, and colors must share shape [N, 3]")
    return point_values, normal_values, color_values


def replace_with_aabb_shell_outliers(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    expansion: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace a fixed fraction of support rows with synthetic mask contaminants.

    Outlier points are sampled from an expanded AABB shell, so every replacement
    lies outside the original support AABB in at least one coordinate. Total
    support count stays fixed, isolating contamination from point-count changes.
    """

    point_values, normal_values, color_values = _validate_arrays(points, normals, colors)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"outlier ratio must be in (0, 1), got {ratio}")
    if expansion <= 1.0:
        raise ValueError(f"AABB expansion must be > 1, got {expansion}")

    count = len(point_values)
    replace_count = min(count - 1, max(1, int(round(count * ratio))))
    replace_indices = rng.choice(count, size=replace_count, replace=False)
    lower = point_values.min(axis=0)
    upper = point_values.max(axis=0)
    center = (lower + upper) * 0.5
    half_extent = np.maximum((upper - lower) * 0.5, 1e-6)
    expanded_lower = center - expansion * half_extent
    expanded_upper = center + expansion * half_extent

    contaminants = np.empty((replace_count, 3), dtype=np.float32)
    filled = 0
    while filled < replace_count:
        candidates = rng.uniform(
            expanded_lower,
            expanded_upper,
            size=(max(16, 2 * (replace_count - filled)), 3),
        ).astype(np.float32)
        outside = np.any((candidates < lower) | (candidates > upper), axis=1)
        accepted = candidates[outside]
        take = min(len(accepted), replace_count - filled)
        contaminants[filled : filled + take] = accepted[:take]
        filled += take

    random_normals = rng.normal(size=(replace_count, 3)).astype(np.float32)
    random_normals /= np.maximum(
        np.linalg.norm(random_normals, axis=1, keepdims=True),
        1e-8,
    )
    color_min = color_values.min(axis=0)
    color_max = color_values.max(axis=0)
    random_colors = rng.uniform(
        color_min,
        np.maximum(color_max, color_min + 1e-6),
        size=(replace_count, 3),
    ).astype(np.float32)

    output_points = point_values.copy()
    output_normals = normal_values.copy()
    output_colors = color_values.copy()
    output_points[replace_indices] = contaminants
    output_normals[replace_indices] = random_normals
    output_colors[replace_indices] = random_colors
    return output_points, output_normals, output_colors


def select_normal_facing_view(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    keep_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate a single-view observation using surface-normal visibility.

    This is intentionally labelled an approximation: it selects the support
    points whose normals face a sampled camera direction most strongly, but does
    not perform mesh z-buffering or self-occlusion ray tests.
    """

    point_values, normal_values, color_values = _validate_arrays(points, normals, colors)
    if not 0.0 < keep_ratio < 1.0:
        raise ValueError(f"normal-view keep ratio must be in (0, 1), got {keep_ratio}")
    camera_direction = rng.normal(size=3).astype(np.float32)
    camera_direction /= max(float(np.linalg.norm(camera_direction)), 1e-8)
    normal_lengths = np.maximum(np.linalg.norm(normal_values, axis=1), 1e-8)
    facing_scores = (normal_values @ camera_direction) / normal_lengths
    keep_count = max(8, int(round(len(point_values) * keep_ratio)))
    indices = np.argpartition(facing_scores, -keep_count)[-keep_count:]
    return (
        point_values[indices].copy(),
        normal_values[indices].copy(),
        color_values[indices].copy(),
    )
