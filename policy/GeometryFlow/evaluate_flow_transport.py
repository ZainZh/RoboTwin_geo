#!/usr/bin/env python3
"""Retrieve and rigidly transport object flows across held-out shoe shapes.

The source and target episodes are first expressed in camera-estimated target
frames.  A representation selects a geometrically similar training shoe.  Its
rigid motion schedule is then endpoint-warped onto the target shoe's initial
PCA frame while retaining exact rigidity.  This explicitly uses an object
representation to construct a task-space action condition instead of merely
concatenating a feature vector into a policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .evaluate_demo_transport import mutual_correspondences, weighted_rigid_fit
from .train_flow_generator import frame_to_world, rigid_rotation, summarize_flow, world_to_frame
from .train_task_flow_benchmark import FOLDS


# Hybrid modes make retrieval and geometric alignment independently testable.
# The first name selects a source demonstration; the second aligns its flow.
MODE_COMPONENTS = {
    "raw": ("raw", "raw"),
    "pca": ("pca", "pca"),
    "ndf": ("ndf", "ndf"),
    "utonia": ("utonia", "utonia"),
    "utonia_pca": ("utonia", "pca"),
    "pca_utonia": ("pca", "utonia"),
    "utonia_s0": ("utonia_s0", "utonia_s0"),
    "utonia_s1": ("utonia_s1", "utonia_s1"),
    "utonia_s2": ("utonia_s2", "utonia_s2"),
    "utonia_s3": ("utonia_s3", "utonia_s3"),
    "utonia_s4": ("utonia_s4", "utonia_s4"),
    "pca_ndf_sign": ("pca", "pca_ndf_sign"),
    "pca_utonia_sign": ("pca", "pca_utonia_sign"),
}

# The public checkpoint uses encoder widths (54, 108, 216, 432, 576).
# ``extract_utonia_features`` concatenates them from finest to deepest.
UTONIA_FEATURE_SLICES = {
    "utonia_s0": slice(0, 54),
    "utonia_s1": slice(54, 162),
    "utonia_s2": slice(162, 378),
    "utonia_s3": slice(378, 810),
    "utonia_s4": slice(810, 1386),
}


def mode_components(mode: str) -> tuple[str, str]:
    try:
        return MODE_COMPONENTS[mode]
    except KeyError as error:
        raise ValueError(f"unknown representation {mode}") from error


def representation_features(
    payload: dict[str, np.ndarray], episode: int, mode: str
) -> np.ndarray:
    if mode == "ndf":
        if "ndf_features" not in payload:
            raise KeyError("mode=ndf requires ndf_features in the dataset")
        return payload["ndf_features"][episode, :, :256]
    if mode == "utonia" or mode in UTONIA_FEATURE_SLICES:
        if "utonia_features" not in payload:
            raise KeyError(f"mode={mode} requires utonia_features in the dataset")
        value = payload["utonia_features"][episode]
        if mode in UTONIA_FEATURE_SLICES:
            value = value[:, UTONIA_FEATURE_SLICES[mode]]
        return value
    raise ValueError(f"mode={mode} does not define point descriptors")


def normalize_rows(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), eps)


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(points, dtype=np.float64)
    center = value.mean(axis=0)
    centered = value - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(len(value), 1))
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    coordinates = centered @ basis
    signs = np.where(np.mean(coordinates**3, axis=0) < 0.0, -1.0, 1.0)
    basis = basis * signs[None]
    coordinates = centered @ basis
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
        coordinates[:, -1] *= -1.0
    scale = max(float(np.max(np.ptp(coordinates, axis=0))), 1e-8)
    return center, basis, coordinates / scale


def symmetric_chamfer(first: np.ndarray, second: np.ndarray) -> float:
    square = np.sum((first[:, None] - second[None]) ** 2, axis=-1)
    return float(0.5 * (square.min(axis=0).mean() + square.min(axis=1).mean()))


PROPER_AXIS_SIGNS = np.asarray(
    ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))
)


def align_pca_signs(
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    target_basis: np.ndarray,
) -> tuple[float, np.ndarray]:
    costs = np.asarray(
        [symmetric_chamfer(source_coordinates, target_coordinates * sign) for sign in PROPER_AXIS_SIGNS]
    )
    sign = PROPER_AXIS_SIGNS[int(np.argmin(costs))]
    return float(np.min(costs)), target_basis @ np.diag(sign)


def descriptor_disambiguated_pca_signs(
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    target_basis: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Let descriptors choose only the discrete, proper PCA-axis sign.

    Direct descriptor-based rigid fits inherit noisy continuous rotations.  In
    contrast, this scorer retains PCA's precise orthogonal axes and asks the
    descriptor for the smaller problem it is better suited to: selecting one
    of the four determinant-positive sign candidates.
    """
    source = np.asarray(source_coordinates, dtype=np.float64)
    target = np.asarray(target_coordinates, dtype=np.float64)
    source_descriptor = normalize_rows(np.asarray(source_features, dtype=np.float64))
    target_descriptor = normalize_rows(np.asarray(target_features, dtype=np.float64))
    if len(source) != len(source_descriptor) or len(target) != len(target_descriptor):
        raise ValueError("PCA coordinates and point descriptors must align")
    costs = []
    for sign in PROPER_AXIS_SIGNS:
        candidate = target * sign[None]
        square = np.sum((source[:, None] - candidate[None]) ** 2, axis=-1)
        source_to_target = np.argmin(square, axis=1)
        target_to_source = np.argmin(square, axis=0)
        source_ids = np.flatnonzero(
            target_to_source[source_to_target] == np.arange(len(source))
        )
        if len(source_ids) < 8:
            source_ids = np.arange(len(source))
        target_ids = source_to_target[source_ids]
        cosine = np.sum(
            source_descriptor[source_ids] * target_descriptor[target_ids], axis=-1
        )
        semantic_cost = float(np.mean(1.0 - cosine))
        geometric_cost = float(np.sqrt(square[source_ids, target_ids]).mean())
        costs.append(semantic_cost + 0.05 * geometric_cost)
    row = int(np.argmin(costs))
    return float(costs[row]), target_basis @ np.diag(PROPER_AXIS_SIGNS[row])


def representation_cost(
    payload: dict[str, np.ndarray], source: int, target: int, mode: str
) -> float:
    mode, _ = mode_components(mode)
    source_points = payload["points_a_xyzrgb"][source, :, :3]
    target_points = payload["points_a_xyzrgb"][target, :, :3]
    _, _, source_pca = pca_frame(source_points)
    _, _, target_pca = pca_frame(target_points)
    if mode == "raw":
        source_extent = np.sort(np.ptp(source_pca, axis=0))[::-1]
        target_extent = np.sort(np.ptp(target_pca, axis=0))[::-1]
        source_moment = np.sort(np.mean(np.abs(source_pca) ** 2, axis=0))[::-1]
        target_moment = np.sort(np.mean(np.abs(target_pca) ** 2, axis=0))[::-1]
        return float(
            np.linalg.norm(source_extent - target_extent)
            + np.linalg.norm(source_moment - target_moment)
        )
    if mode == "pca":
        cost, _ = align_pca_signs(source_pca, target_pca, np.eye(3))
        return cost
    if mode == "ndf" or mode == "utonia" or mode in UTONIA_FEATURE_SLICES:
        source_feature = representation_features(payload, source, mode)
        target_feature = representation_features(payload, target, mode)
        source_feature = normalize_rows(source_feature)
        target_feature = normalize_rows(target_feature)
        similarity = source_feature @ target_feature.T
        return float(1.0 - 0.5 * (similarity.max(axis=0).mean() + similarity.max(axis=1).mean()))
    raise ValueError(f"unknown retrieval representation {mode}")


def rigid_trajectory(
    initial: np.ndarray, trajectory: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotations, centers = [], []
    for step in range(trajectory.shape[1]):
        rotations.append(rigid_rotation(initial, trajectory[:, step]))
        centers.append(trajectory[:, step].mean(axis=0))
    return np.asarray(rotations), np.asarray(centers)


def motion_progress(rotations: np.ndarray, centers: np.ndarray) -> np.ndarray:
    translation = np.linalg.norm(np.diff(centers, axis=0), axis=-1)
    rotation = []
    for previous, current in zip(rotations[:-1], rotations[1:]):
        rotation.append(Rotation.from_matrix(current @ previous.T).magnitude())
    increments = translation + 0.06 * np.asarray(rotation)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    if cumulative[-1] < 1e-8:
        return np.linspace(0.0, 1.0, len(centers))
    return cumulative / cumulative[-1]


def interpolate_rotation(rotation: np.ndarray, fraction: float) -> np.ndarray:
    vector = Rotation.from_matrix(rotation).as_rotvec() * float(fraction)
    return Rotation.from_rotvec(vector).as_matrix()


def transport_flow(
    payload: dict[str, np.ndarray], source: int, target: int, mode: str
) -> np.ndarray:
    _, mode = mode_components(mode)
    source_frame = payload["target_frame_camera"][source]
    target_frame = payload["target_frame_camera"][target]
    source_points = world_to_frame(
        payload["points_a_xyzrgb"][source, :, :3], source_frame
    )
    target_points = world_to_frame(
        payload["points_a_xyzrgb"][target, :, :3], target_frame
    )
    source_flow = world_to_frame(payload["episode_flow"][source], source_frame)
    target_anchors = world_to_frame(payload["initial_anchors"][target], target_frame)
    source_initial = source_flow[:, 0]
    source_delta_rotation, source_centers = rigid_trajectory(source_initial, source_flow)
    source_initial_center, source_initial_basis, source_coordinates = pca_frame(source_points)
    target_initial_center, target_initial_basis, target_coordinates_pca = pca_frame(target_points)
    if mode == "pca":
        _, target_initial_basis = align_pca_signs(
            source_coordinates, target_coordinates_pca, target_initial_basis
        )
    elif mode in {"pca_ndf_sign", "pca_utonia_sign"}:
        descriptor_mode = "ndf" if mode == "pca_ndf_sign" else "utonia"
        _, target_initial_basis = descriptor_disambiguated_pca_signs(
            source_coordinates,
            target_coordinates_pca,
            target_initial_basis,
            representation_features(payload, source, descriptor_mode),
            representation_features(payload, target, descriptor_mode),
        )
    elif mode == "ndf" or mode == "utonia" or mode in UTONIA_FEATURE_SLICES:
        source_feature = representation_features(payload, source, mode)
        target_feature = representation_features(payload, target, mode)
        source_ids, target_ids, weights, _ = mutual_correspondences(
            source_points,
            target_points,
            source_feature,
            target_feature,
            maximum=96,
            cosine_features=True,
        )
        alignment, _, _ = weighted_rigid_fit(
            source_points[source_ids], target_points[target_ids], weights
        )
        target_initial_basis = alignment @ source_initial_basis
    # The 32 anchors and the 256-point cloud have slightly different means.
    # Use the anchor centroid for motion while retaining the full-cloud PCA axes.
    source_anchor_center = source_initial.mean(axis=0)
    target_anchor_center = target_anchors.mean(axis=0)
    source_bases = np.einsum("tij,jk->tik", source_delta_rotation, source_initial_basis)
    progress = motion_progress(source_delta_rotation, source_centers)

    base_centers = target_anchor_center[None] + (source_centers - source_anchor_center)
    center_correction = source_centers[-1] - base_centers[-1]
    base_bases = np.einsum("tij,jk->tik", source_delta_rotation, target_initial_basis)
    final_basis_correction = source_bases[-1] @ base_bases[-1].T
    target_coordinates = (target_anchors - target_initial_center) @ target_initial_basis
    # Recenter intrinsic coordinates on the anchor set, not the denser support.
    target_coordinates -= target_coordinates.mean(axis=0, keepdims=True)
    prediction = []
    for step, fraction in enumerate(progress):
        basis = interpolate_rotation(final_basis_correction, fraction) @ base_bases[step]
        center = base_centers[step] + fraction * center_correction
        prediction.append(target_coordinates @ basis.T + center)
    prediction = np.stack(prediction, axis=1)
    prediction[:, 0] = target_anchors
    return frame_to_world(prediction, target_frame).astype(np.float32)


def evaluate_mode(
    payload: dict[str, np.ndarray], mode: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    shoe_ids = payload["shoe_id"]
    all_indices, all_top, all_target = [], [], []
    records = []
    for fold_index, fold in enumerate(FOLDS):
        train = np.flatnonzero(np.isin(shoe_ids, fold["train"]))
        test = np.flatnonzero(np.isin(shoe_ids, fold["test"]))
        for target in test:
            # Arm routing is camera-only: the expert selects the arm from the
            # initial shoe X position.  It is not read from the stored label.
            target_right = payload["points_a_xyzrgb"][target, :, 0].mean() >= 0.0
            compatible = train[
                np.asarray(
                    [
                        payload["points_a_xyzrgb"][source, :, 0].mean() >= 0.0
                        for source in train
                    ]
                )
                == target_right
            ]
            if not len(compatible):
                compatible = train
            costs = np.asarray(
                [representation_cost(payload, int(source), int(target), mode) for source in compatible]
            )
            candidate_flows = [
                transport_flow(payload, int(source), int(target), mode) for source in compatible
            ]
            target_flow = payload["episode_flow"][target]
            candidate_final_errors = np.asarray(
                [
                    np.linalg.norm(value[:, -1] - target_flow[:, -1], axis=-1).mean()
                    for value in candidate_flows
                ]
            )
            top_row = int(np.argmin(costs))
            oracle_row = int(np.argmin(candidate_final_errors))
            top_source = int(compatible[top_row])
            oracle_source = int(compatible[oracle_row])
            top_prediction = candidate_flows[top_row]
            all_indices.append(int(target))
            all_top.append(top_prediction)
            all_target.append(target_flow)
            records.append(
                {
                    "fold": fold_index,
                    "target_episode": int(target),
                    "target_shoe": int(shoe_ids[target]),
                    "top_source_episode": top_source,
                    "top_source_shoe": int(shoe_ids[top_source]),
                    "top_representation_cost": float(costs[top_row]),
                    "top_final_anchor_epe_cm": float(candidate_final_errors[top_row] * 100.0),
                    "oracle_source_episode": oracle_source,
                    "oracle_source_shoe": int(shoe_ids[oracle_source]),
                    "oracle_final_anchor_epe_cm": float(candidate_final_errors[oracle_row] * 100.0),
                }
            )
    indices = np.asarray(all_indices, dtype=np.int64)
    order = np.argsort(indices)
    prediction = np.asarray(all_top, dtype=np.float32)[order]
    target = np.asarray(all_target, dtype=np.float32)[order]
    summary = summarize_flow(prediction, target, indices[order])
    summary["oracle_source_final_anchor_epe_cm"] = float(
        np.mean([row["oracle_final_anchor_epe_cm"] for row in records])
    )
    summary["records"] = records
    return summary, indices[order], prediction


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--modes", nargs="+", default=["raw", "pca", "ndf"])
    return result


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    summaries = {}
    for mode in args.modes:
        summary, indices, prediction = evaluate_mode(payload, mode)
        summaries[mode] = summary
        np.savez_compressed(
            args.output_dir / f"{mode}_predictions.npz",
            episode_indices=indices,
            predicted_flow=prediction,
            target_flow=payload["episode_flow"][indices],
        )
        (args.output_dir / f"{mode}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"mode": mode, "summary": summary}, indent=2), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
