from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_field_eval_utils import (
    apply_support_normalization,
    compute_support_normalization,
)


@dataclass(frozen=True)
class SupportCondition:
    label: str
    kind: str
    value: float


def parse_support_condition(label: str) -> SupportCondition:
    value = str(label).strip().lower()
    if value == "clean":
        return SupportCondition(label="clean", kind="clean", value=1.0)
    if value.startswith("count_"):
        count = int(value.removeprefix("count_"))
        if count <= 0:
            raise ValueError(f"support count must be positive, got {count}")
        return SupportCondition(label=value, kind="resample_count", value=float(count))
    if value.startswith("crop_keep_"):
        ratio = float(value.removeprefix("crop_keep_"))
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"crop keep ratio must be in (0, 1], got {ratio}")
        return SupportCondition(label=value, kind="view_crop", value=ratio)
    raise ValueError(f"unsupported support condition: {label}")


def normalize_condition_weights(
    labels: list[str],
    weights: list[float] | None,
) -> tuple[list[SupportCondition], np.ndarray]:
    conditions = [parse_support_condition(label) for label in labels]
    if not conditions:
        raise ValueError("at least one support condition is required")
    if len({condition.label for condition in conditions}) != len(conditions):
        raise ValueError("support condition labels must be unique")

    if weights is None:
        probabilities = np.full(len(conditions), 1.0 / len(conditions), dtype=np.float64)
    else:
        if len(weights) != len(conditions):
            raise ValueError("condition weights must match the number of conditions")
        probabilities = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError("condition weights must be finite and non-negative")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("at least one condition weight must be positive")
        probabilities = probabilities / total
    return conditions, probabilities


def choose_support_condition(
    conditions: list[SupportCondition],
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> SupportCondition:
    index = int(rng.choice(len(conditions), p=probabilities))
    return conditions[index]


def _validate_support_arrays(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_values = np.asarray(points, dtype=np.float32)
    normal_values = np.asarray(normals, dtype=np.float32)
    color_values = np.asarray(colors, dtype=np.float32)
    if point_values.ndim != 2 or point_values.shape[1] != 3 or len(point_values) == 0:
        raise ValueError(f"support points must have shape [N, 3], got {point_values.shape}")
    if normal_values.shape != point_values.shape or color_values.shape != point_values.shape:
        raise ValueError("support points, normals, and colors must share shape [N, 3]")
    return point_values, normal_values, color_values


def select_observed_support(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    condition: SupportCondition,
    rng: np.random.Generator,
) -> dict:
    point_values, normal_values, color_values = _validate_support_arrays(
        points, normals, colors
    )
    count = len(point_values)
    direction = None
    if condition.kind == "clean":
        indices = np.arange(count, dtype=np.int64)
    elif condition.kind == "resample_count":
        keep_count = min(count, int(condition.value))
        indices = rng.choice(count, size=keep_count, replace=False)
    elif condition.kind == "view_crop":
        keep_count = min(count, max(8, int(round(count * condition.value))))
        direction = rng.normal(size=3).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        projection = point_values @ direction
        indices = np.argpartition(projection, -keep_count)[-keep_count:]
    else:
        raise ValueError(f"unsupported condition kind: {condition.kind}")

    return {
        "points": point_values[indices].copy(),
        "normals": normal_values[indices].copy(),
        "colors": color_values[indices].copy(),
        "source_indices": np.asarray(indices, dtype=np.int64),
        "direction": direction,
        "observed_source_points": int(len(indices)),
    }


def _resample_rows(
    arrays: tuple[np.ndarray, ...],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    if count <= 0:
        raise ValueError(f"output support count must be positive, got {count}")
    source_count = len(arrays[0])
    replace = source_count < count
    indices = rng.choice(source_count, size=count, replace=replace)
    return tuple(np.asarray(values)[indices].copy() for values in arrays)


def build_fixed_query_support_pair(
    *,
    reference_points: np.ndarray,
    reference_normals: np.ndarray,
    reference_colors: np.ndarray,
    raw_query_points: np.ndarray,
    condition: SupportCondition,
    output_support_count: int,
    coord_scale: float,
    center_shift_z: bool,
    rng: np.random.Generator,
    rotation: np.ndarray | None = None,
) -> dict:
    points, normals, colors = _validate_support_arrays(
        reference_points, reference_normals, reference_colors
    )
    queries = np.asarray(raw_query_points, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError(f"raw query points must have shape [Q, 3], got {queries.shape}")

    observed = select_observed_support(points, normals, colors, condition, rng)
    clean_normalization = compute_support_normalization(
        points,
        coord_scale=coord_scale,
        center_shift_z=center_shift_z,
    )
    perturbed_normalization = compute_support_normalization(
        observed["points"],
        coord_scale=coord_scale,
        center_shift_z=center_shift_z,
    )
    clean_points = apply_support_normalization(points, clean_normalization)
    clean_queries = apply_support_normalization(queries, clean_normalization)
    perturbed_source = apply_support_normalization(
        observed["points"], perturbed_normalization
    )
    perturbed_queries = apply_support_normalization(queries, perturbed_normalization)

    clean_points, clean_normals, clean_colors = _resample_rows(
        (clean_points, normals, colors), output_support_count, rng
    )
    perturbed_points, perturbed_normals, perturbed_colors = _resample_rows(
        (perturbed_source, observed["normals"], observed["colors"]),
        output_support_count,
        rng,
    )

    if rotation is None:
        rotation_values = np.eye(3, dtype=np.float32)
    else:
        rotation_values = np.asarray(rotation, dtype=np.float32)
        if rotation_values.shape != (3, 3):
            raise ValueError(f"rotation must have shape [3, 3], got {rotation_values.shape}")

    clean_points = (clean_points @ rotation_values.T).astype(np.float32)
    perturbed_points = (perturbed_points @ rotation_values.T).astype(np.float32)
    clean_queries = (clean_queries @ rotation_values.T).astype(np.float32)
    perturbed_queries = (perturbed_queries @ rotation_values.T).astype(np.float32)
    clean_normals = (clean_normals @ rotation_values.T).astype(np.float32)
    perturbed_normals = (perturbed_normals @ rotation_values.T).astype(np.float32)

    return {
        "clean_support_points": clean_points,
        "clean_support_normals": clean_normals,
        "clean_support_colors": clean_colors.astype(np.float32),
        "clean_query_points": clean_queries,
        "perturbed_support_points": perturbed_points,
        "perturbed_support_normals": perturbed_normals,
        "perturbed_support_colors": perturbed_colors.astype(np.float32),
        "perturbed_query_points": perturbed_queries,
        "condition": condition.label,
        "observed_source_points": observed["observed_source_points"],
        "crop_direction": (
            observed["direction"].tolist() if observed["direction"] is not None else None
        ),
    }
