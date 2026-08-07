from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_legacy_config_path(
    configured_path: str | Path | None,
    local_config_root: str | Path,
) -> tuple[Path | None, dict | None]:
    if configured_path is None:
        return None, None

    original = Path(configured_path).expanduser()
    local = Path(local_config_root) / original.name
    original_exists = original.is_file()
    local_exists = local.is_file()

    if not original_exists and not local_exists:
        raise FileNotFoundError(
            f"Alias config does not exist at configured or local path: "
            f"{original}, {local}"
        )

    original_hash = sha256_file(original) if original_exists else None
    local_hash = sha256_file(local) if local_exists else None
    if original_hash is not None and local_hash is not None and original_hash != local_hash:
        raise RuntimeError(
            "Configured and repository-local alias configs have different contents: "
            f"{original} ({original_hash}) vs {local} ({local_hash})"
        )

    resolved = local.resolve() if local_exists else original.resolve()
    return resolved, {
        "configured_path": str(original),
        "configured_exists": original_exists,
        "repository_path": str(local.resolve()) if local_exists else str(local),
        "repository_exists": local_exists,
        "resolved_path": str(resolved),
        "sha256": local_hash or original_hash,
    }


def confusion_matrix_from_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> tuple[np.ndarray, int]:
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got "
            f"{predictions.shape} and {targets.shape}"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    valid = (
        (targets != int(ignore_index))
        & (targets >= 0)
        & (targets < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )
    valid_count = int(valid.sum())
    encoded = targets[valid] * num_classes + predictions[valid]
    matrix = np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return matrix.astype(np.int64, copy=False), valid_count


def metrics_from_confusion(
    confusion: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> dict:
    matrix = np.asarray(confusion, dtype=np.int64)
    num_classes = len(label_names)
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(
            f"confusion must have shape {(num_classes, num_classes)}, got {matrix.shape}"
        )

    true_positive = np.diag(matrix).astype(np.float64)
    target_count = matrix.sum(axis=1).astype(np.float64)
    prediction_count = matrix.sum(axis=0).astype(np.float64)
    union = target_count + prediction_count - true_positive
    total = float(matrix.sum())

    iou = np.divide(
        true_positive,
        union,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=union > 0,
    )
    precision = np.divide(
        true_positive,
        prediction_count,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=prediction_count > 0,
    )
    recall = np.divide(
        true_positive,
        target_count,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=target_count > 0,
    )

    valid_iou = np.isfinite(iou)
    valid_recall = np.isfinite(recall)
    target_frequency = target_count / total if total > 0 else np.zeros(num_classes)

    per_class = []
    for index, name in enumerate(label_names):
        per_class.append(
            {
                "class_id": index,
                "label": str(name),
                "iou": float(iou[index]) if np.isfinite(iou[index]) else None,
                "precision": (
                    float(precision[index]) if np.isfinite(precision[index]) else None
                ),
                "recall": float(recall[index]) if np.isfinite(recall[index]) else None,
                "true_positive": int(true_positive[index]),
                "target_points": int(target_count[index]),
                "predicted_points": int(prediction_count[index]),
                "union_points": int(union[index]),
            }
        )

    return {
        "valid_points": int(total),
        "accuracy": float(true_positive.sum() / total) if total > 0 else None,
        "mean_iou": float(np.mean(iou[valid_iou])) if valid_iou.any() else None,
        "balanced_accuracy": (
            float(np.mean(recall[valid_recall])) if valid_recall.any() else None
        ),
        "frequency_weighted_iou": (
            float(np.sum(target_frequency[valid_iou] * iou[valid_iou]))
            if total > 0 and valid_iou.any()
            else None
        ),
        "num_classes_in_union": int(valid_iou.sum()),
        "num_classes_in_targets": int(valid_recall.sum()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def merge_confusions(confusions: list[np.ndarray], num_classes: int) -> np.ndarray:
    merged = np.zeros((num_classes, num_classes), dtype=np.int64)
    for matrix in confusions:
        array = np.asarray(matrix, dtype=np.int64)
        if array.shape != merged.shape:
            raise ValueError(f"Cannot merge confusion with shape {array.shape}")
        merged += array
    return merged


def compute_support_normalization(
    support_points: np.ndarray,
    coord_scale: float,
    center_shift_z: bool,
) -> dict:
    support = np.asarray(support_points, dtype=np.float32)
    if support.ndim != 2 or support.shape[1] != 3 or len(support) == 0:
        raise ValueError(f"support_points must have shape [N, 3], got {support.shape}")

    centroid = support.mean(axis=0, keepdims=True)
    centered = support - centroid
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    unit_support = centered / radius
    shift = np.zeros((1, 3), dtype=np.float32)
    if center_shift_z:
        x_min, y_min, z_min = unit_support.min(axis=0)
        x_max, y_max, _ = unit_support.max(axis=0)
        shift[0] = np.asarray(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min],
            dtype=np.float32,
        )
    return {
        "centroid": centroid.astype(np.float32),
        "radius": radius,
        "shift": shift,
        "coord_scale": float(coord_scale),
        "center_shift_z": bool(center_shift_z),
    }


def apply_support_normalization(points: np.ndarray, parameters: dict) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    normalized = (values - parameters["centroid"]) / float(parameters["radius"])
    normalized = normalized - parameters["shift"]
    return (normalized * float(parameters["coord_scale"])).astype(np.float32)


def embedding_stability_metrics(
    reference_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> dict:
    reference = np.asarray(reference_embeddings, dtype=np.float32)
    candidate = np.asarray(candidate_embeddings, dtype=np.float32)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(
            "reference and candidate embeddings must share shape [N, D], got "
            f"{reference.shape} and {candidate.shape}"
        )
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    candidate_norm = np.linalg.norm(candidate, axis=1, keepdims=True)
    reference_unit = reference / np.maximum(reference_norm, 1e-8)
    candidate_unit = candidate / np.maximum(candidate_norm, 1e-8)
    cosine = np.sum(reference_unit * candidate_unit, axis=1)
    l2 = np.linalg.norm(reference - candidate, axis=1)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(np.quantile(cosine, 0.05)),
        "cosine_p50": float(np.quantile(cosine, 0.50)),
        "cosine_p95": float(np.quantile(cosine, 0.95)),
        "cosine_min": float(cosine.min()),
        "embedding_l2_mean": float(l2.mean()),
        "embedding_l2_p95": float(np.quantile(l2, 0.95)),
    }
