#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import GRIPPER_KEYPOINTS_METERS


FOLDS = (
    ((2, 3, 4, 7, 8, 9), (0, 5)),
    ((0, 3, 4, 5, 8, 9), (1, 6)),
    ((0, 1, 4, 5, 6, 9), (2, 7)),
    ((0, 1, 2, 5, 6, 7), (3, 8)),
    ((1, 2, 3, 6, 7, 8), (4, 9)),
)


def normalize_rows(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), eps)


def normalized_xyz(points: np.ndarray) -> np.ndarray:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    scale = max(float(np.max(high - low)), 1e-8)
    return (points - center) / scale


def pca_canonical_xyz(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1]
    coordinates = centered @ basis
    third_moment = np.mean(coordinates**3, axis=0)
    signs = np.where(third_moment < 0.0, -1.0, 1.0)
    coordinates = coordinates * signs[None]
    scale = max(float(np.max(np.ptp(coordinates, axis=0))), 1e-8)
    return coordinates / scale


def mutual_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_features: np.ndarray | None,
    target_features: np.ndarray | None,
    *,
    maximum: int,
    cosine_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if source_features is None:
        source_repr = normalized_xyz(source_points)
        target_repr = normalized_xyz(target_points)
        distance = np.linalg.norm(
            source_repr[:, None, :] - target_repr[None, :, :], axis=-1
        )
    elif cosine_features:
        source_repr = normalize_rows(source_features)
        target_repr = normalize_rows(target_features)
        distance = 1.0 - source_repr @ target_repr.T
    else:
        distance = np.linalg.norm(
            source_features[:, None, :] - target_features[None, :, :], axis=-1
        )
    source_to_target = np.argmin(distance, axis=1)
    target_to_source = np.argmin(distance, axis=0)
    source_indices = np.arange(len(source_points))
    mutual = target_to_source[source_to_target] == source_indices
    selected_source = source_indices[mutual]
    if len(selected_source) < 6:
        selected_source = source_indices
    selected_target = source_to_target[selected_source]
    selected_distance = distance[selected_source, selected_target]
    order = np.argsort(selected_distance)[: int(maximum)]
    selected_source = selected_source[order]
    selected_target = selected_target[order]
    selected_distance = selected_distance[order]
    median = max(float(np.median(selected_distance)), 1e-6)
    weights = np.exp(-selected_distance / median)
    weights = np.maximum(weights, 1e-4)
    return (
        selected_source,
        selected_target,
        weights,
        float(np.mean(selected_distance)),
    )


def weighted_rigid_fit(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = weights / np.sum(weights)
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * weights[:, None]).T @ target_zero
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    aligned = source @ rotation.T + translation
    rmse = float(np.sqrt(np.sum(weights * np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, rmse


def transfer_pose(
    source_pose: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation @ source_pose[:3, :3]
    result[:3, 3] = rotation @ source_pose[:3, 3] + translation
    return result


def pose_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    translation = float(np.linalg.norm(predicted[:3, 3] - target[:3, 3]))
    relative = predicted[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    rotation = float(np.rad2deg(np.arccos(cosine)))
    canonical = np.asarray(GRIPPER_KEYPOINTS_METERS, dtype=np.float64)
    predicted_keypoints = canonical @ predicted[:3, :3].T + predicted[:3, 3]
    target_keypoints = canonical @ target[:3, :3].T + target[:3, 3]
    keypoint = float(np.sqrt(np.mean((predicted_keypoints - target_keypoints) ** 2)))
    return {
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
        "keypoint_rmse_m": keypoint,
        "success_2cm_15deg": bool(translation < 0.02 and rotation < 15.0),
        "success_3cm_30deg": bool(translation < 0.03 and rotation < 30.0),
    }


def summarize(records: list[dict], route: str) -> dict:
    def values(field: str) -> np.ndarray:
        return np.asarray([row[route][field] for row in records], dtype=np.float64)

    return {
        "samples": len(records),
        "translation_error_m_mean": float(values("translation_error_m").mean()),
        "translation_error_m_median": float(np.median(values("translation_error_m"))),
        "rotation_error_deg_mean": float(values("rotation_error_deg").mean()),
        "rotation_error_deg_median": float(np.median(values("rotation_error_deg"))),
        "keypoint_rmse_m_mean": float(values("keypoint_rmse_m").mean()),
        "success_2cm_15deg": float(values("success_2cm_15deg").mean()),
        "success_3cm_30deg": float(values("success_3cm_30deg").mean()),
    }


def rotate_about_center(
    points: np.ndarray, pose: np.ndarray, yaw_degrees: float
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(float(yaw_degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    rotated_points = (points - center) @ rotation.T + center
    rotated_pose = pose.copy()
    rotated_pose[:3, :3] = rotation @ pose[:3, :3]
    rotated_pose[:3, 3] = rotation @ (pose[:3, 3] - center) + center
    return rotated_points, rotated_pose


def evaluate_mode(
    payload: dict[str, np.ndarray], mode: str, yaw_degrees: tuple[float, ...]
) -> tuple[list[dict], dict]:
    points = np.asarray(payload["points_world"], dtype=np.float64)
    poses = np.asarray(payload["target_transform_world"], dtype=np.float64)
    shoe_ids = np.asarray(payload["shoe_id_label_only"], dtype=np.int64)
    arms = np.asarray(payload["active_arm_right"], dtype=np.int64)
    episodes = np.asarray(payload["episode_id"], dtype=np.int64)
    features = None
    if mode == "ndf":
        features = np.asarray(payload["ndf_features"][..., :256], dtype=np.float64)
    elif mode == "pca":
        features = np.stack([pca_canonical_xyz(cloud) for cloud in points], axis=0)
    records = []
    for fold, (train_shoes, test_shoes) in enumerate(FOLDS):
        train_indices = np.flatnonzero(np.isin(shoe_ids, train_shoes))
        test_indices = np.flatnonzero(np.isin(shoe_ids, test_shoes))
        for target_index in test_indices:
            for yaw in yaw_degrees:
                target_points, target_pose = rotate_about_center(
                    points[target_index], poses[target_index], yaw
                )
                compatible = train_indices[arms[train_indices] == arms[target_index]]
                if not len(compatible):
                    compatible = train_indices
                candidates = []
                for source_index in compatible:
                    source_ids, target_ids, weights, descriptor_cost = mutual_correspondences(
                        points[source_index],
                        target_points,
                        None if features is None else features[source_index],
                        None if features is None else features[target_index],
                        maximum=96,
                        cosine_features=mode == "ndf",
                    )
                    rotation, translation, fit_rmse = weighted_rigid_fit(
                        points[source_index, source_ids],
                        target_points[target_ids],
                        weights,
                    )
                    prediction = transfer_pose(poses[source_index], rotation, translation)
                    target_scale = max(
                        float(np.max(np.ptp(target_points, axis=0))), 1e-6
                    )
                    # Geometry consistency makes scores comparable across source
                    # demonstrations; descriptor cost breaks close ties for NDF.
                    score = fit_rmse / target_scale + 0.1 * descriptor_cost
                    candidates.append(
                        {
                            "source_episode": int(episodes[source_index]),
                            "score": float(score),
                            "metrics": pose_metrics(prediction, target_pose),
                        }
                    )
                top = min(candidates, key=lambda row: row["score"])
                oracle = min(
                    candidates, key=lambda row: row["metrics"]["keypoint_rmse_m"]
                )
                records.append(
                    {
                        "fold": fold,
                        "episode": int(episodes[target_index]),
                        "yaw_degrees": float(yaw),
                        "shoe_id": int(shoe_ids[target_index]),
                        "active_arm": "right" if arms[target_index] else "left",
                        "top1_source_episode": top["source_episode"],
                        "oracle_source_episode": oracle["source_episode"],
                        "candidate_count": len(candidates),
                        "top1": top["metrics"],
                        "oracle_at_demo": oracle["metrics"],
                    }
                )
    return records, {
        "top1": summarize(records, "top1"),
        "oracle_at_demo": summarize(records, "oracle_at_demo"),
        "records": records,
    }


def clustered_difference(
    first: list[dict],
    second: list[dict],
    route: str,
    field: str,
    samples: int,
    label: str,
) -> dict:
    def key(row: dict) -> tuple[int, float]:
        return int(row["episode"]), float(row["yaw_degrees"])

    second_by_episode = {key(row): row for row in second}
    shoes = np.unique([row["shoe_id"] for row in first])
    by_shoe = {
        shoe: [row for row in first if row["shoe_id"] == shoe] for shoe in shoes
    }

    def difference(row: dict) -> float:
        return float(row[route][field]) - float(second_by_episode[key(row)][route][field])

    observed = float(np.mean([difference(row) for row in first]))
    rng = np.random.RandomState(0)
    draws = []
    for _ in range(samples):
        selected = rng.choice(shoes, size=len(shoes), replace=True)
        draws.append(
            float(np.mean([difference(row) for shoe in selected for row in by_shoe[shoe]]))
        )
    return {
        label: observed,
        "shoe_cluster_bootstrap_95_percent_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo-conditioned rigid relation transfer from camera point clouds."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--yaw-degrees", type=float, nargs="+", default=(0.0,))
    args = parser.parse_args()
    with np.load(args.dataset) as stored:
        payload = {key: np.asarray(stored[key]) for key in stored.files}
    yaw_degrees = tuple(float(value) for value in args.yaw_degrees)
    raw_records, raw = evaluate_mode(payload, "raw", yaw_degrees)
    pca_records, pca = evaluate_mode(payload, "pca", yaw_degrees)
    ndf_records, ndf = evaluate_mode(payload, "ndf", yaw_degrees)
    result = {
        "schema_version": 1,
        "protocol": "five object-disjoint folds; same-arm train demos only",
        "target_yaw_degrees": list(yaw_degrees),
        "raw_correspondence": raw,
        "pca_correspondence": pca,
        "ndf_correspondence": ndf,
        "paired_ndf_minus_raw": {},
        "paired_ndf_minus_pca": {},
    }
    for key, first, second, label in (
        ("paired_ndf_minus_raw", ndf_records, raw_records, "ndf_minus_raw"),
        ("paired_ndf_minus_pca", ndf_records, pca_records, "ndf_minus_pca"),
    ):
        for route in ("top1", "oracle_at_demo"):
            result[key][route] = {}
            for field in (
                "translation_error_m",
                "rotation_error_deg",
                "keypoint_rmse_m",
                "success_2cm_15deg",
                "success_3cm_30deg",
            ):
                result[key][route][field] = clustered_difference(
                    first,
                    second,
                    route,
                    field,
                    args.bootstrap_samples,
                    label,
                )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    compact = {
        "protocol": result["protocol"],
        "raw_correspondence": {key: raw[key] for key in ("top1", "oracle_at_demo")},
        "pca_correspondence": {key: pca[key] for key in ("top1", "oracle_at_demo")},
        "ndf_correspondence": {key: ndf[key] for key in ("top1", "oracle_at_demo")},
        "paired_ndf_minus_raw": result["paired_ndf_minus_raw"],
        "paired_ndf_minus_pca": result["paired_ndf_minus_pca"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
