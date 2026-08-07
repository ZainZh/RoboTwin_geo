#!/usr/bin/env python3
"""Replace the relative-frame token with a camera-only NDF/marker estimate.

At inference this construction uses only the current A point cloud, a cached
early B marker observation, and fold-specific calibration constants learned on
training objects.  Simulator poses are used only to fit those constants and to
report diagnostic errors; they are not copied into ``target_frame9``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .functional_action_frame import frame9_rotation, pose9_rotation
from .target_frame import (
    estimate_cached_geometry_marker_frame,
)
from .train_task_flow_benchmark import FOLDS


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset containing source_frame6_columns from the NDF Z/Y heads.",
    )
    result.add_argument("--data-dir", action="append", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def cached_marker_frames(
    data_dirs: list[Path], episode_count: int
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    frames = []
    confidence = []
    diagnostics = []
    for data_dir in data_dirs:
        episode = 0
        while len(frames) < int(episode_count):
            path = data_dir / f"episode{episode}.hdf5"
            if not path.is_file():
                break
            with h5py.File(path, "r") as archive:
                observations = archive["object_pointcloud/{B}"]
                frame, diagnostic = estimate_cached_geometry_marker_frame(
                    observations
                )
            reliable = bool(
                not diagnostic["sparse_fallback"]
                and int(diagnostic["marker_points"]) >= 20
                and float(diagnostic["fit_score_m2"]) <= 2.0e-5
            )
            frames.append(frame)
            confidence.append(float(reliable))
            diagnostics.append(diagnostic)
            episode += 1
    if len(frames) != int(episode_count):
        raise ValueError(
            f"raw data directories provide {len(frames)} episodes, expected {episode_count}"
        )
    return (
        np.asarray(frames, dtype=np.float32),
        np.asarray(confidence, dtype=np.float32),
        diagnostics,
    )


def camera_goal_calibration(
    marker: np.ndarray,
    true_goal_position: np.ndarray,
    true_goal_rotation: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marker_rotation = np.asarray(marker[:, :3, :3], dtype=np.float64)
    marker_position = np.asarray(marker[:, :3, 3], dtype=np.float64)
    local_rotation = np.swapaxes(marker_rotation, -1, -2) @ true_goal_rotation
    local_position = np.einsum(
        "eji,ej->ei",
        marker_rotation,
        true_goal_position - marker_position,
    )
    train_indices = np.flatnonzero(train_mask)
    train_rotation = local_rotation[train_indices]
    pairwise = rotation_error_deg(
        train_rotation[:, None], train_rotation[None]
    )
    medoid = int(np.argmin(np.median(pairwise, axis=1)))
    distance = pairwise[medoid]
    inlier_local = distance <= 30.0
    if int(inlier_local.sum()) < max(3, int(np.ceil(0.5 * len(train_indices)))):
        keep = max(3, int(np.ceil(0.5 * len(train_indices))))
        inlier_local = np.zeros(len(train_indices), dtype=bool)
        inlier_local[np.argsort(distance)[:keep]] = True
    calibration_mask = np.zeros(len(marker), dtype=bool)
    calibration_mask[train_indices[inlier_local]] = True
    rotation_offset = Rotation.from_matrix(
        local_rotation[calibration_mask]
    ).mean().as_matrix()
    position_offset = np.median(local_position[calibration_mask], axis=0)
    return (
        position_offset.astype(np.float32),
        rotation_offset.astype(np.float32),
        calibration_mask,
    )


def predict_goal_pose(
    marker: np.ndarray,
    position_offset: np.ndarray,
    rotation_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    marker_rotation = np.asarray(marker[:, :3, :3], dtype=np.float64)
    marker_position = np.asarray(marker[:, :3, 3], dtype=np.float64)
    position = marker_position + np.einsum(
        "eij,j->ei", marker_rotation, np.asarray(position_offset)
    )
    rotation = marker_rotation @ np.asarray(rotation_offset)
    return position.astype(np.float32), rotation.astype(np.float32)


def calibrate_current_origin_offset(
    points: np.ndarray,
    predicted_rotation: np.ndarray,
    true_position: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    centroid = np.asarray(points[..., :3], dtype=np.float64).mean(axis=1)
    local = np.einsum(
        "nji,nj->ni",
        np.asarray(predicted_rotation, dtype=np.float64),
        np.asarray(true_position, dtype=np.float64) - centroid,
    )
    return np.median(local[train_mask], axis=0).astype(np.float32)


def predict_current_position(
    points: np.ndarray,
    predicted_rotation: np.ndarray,
    local_offset: np.ndarray,
) -> np.ndarray:
    centroid = np.asarray(points[..., :3], dtype=np.float64).mean(axis=1)
    return (
        centroid
        + np.einsum(
            "nij,j->ni", np.asarray(predicted_rotation), np.asarray(local_offset)
        )
    ).astype(np.float32)


def rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.asarray(predicted) @ np.swapaxes(np.asarray(target), -1, -2)
    leading = relative.shape[:-2]
    angle = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    return np.rad2deg(angle).reshape(leading)


def error_summary(translation_m: np.ndarray, rotation_deg: np.ndarray) -> dict:
    translation_cm = np.linalg.norm(translation_m, axis=-1) * 100.0
    return {
        "samples": int(len(translation_cm)),
        "translation_cm_mean": float(translation_cm.mean()),
        "translation_cm_median": float(np.median(translation_cm)),
        "translation_cm_p90": float(np.percentile(translation_cm, 90)),
        "rotation_deg_mean": float(rotation_deg.mean()),
        "rotation_deg_median": float(np.median(rotation_deg)),
        "rotation_deg_p90": float(np.percentile(rotation_deg, 90)),
    }


def main() -> None:
    args = parser().parse_args()
    if args.fold < 0 or args.fold >= len(FOLDS):
        raise ValueError(f"fold must lie in [0,{len(FOLDS) - 1}]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "source_frame6_columns",
        "current_object_pose9",
        "goal_translation_error_xyz_m",
        "goal_rotation_error_rotvec",
        "episode_index",
        "episode_shoe_id",
        "points_a",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"dataset is missing fields: {missing}")
    episode_shoe = np.asarray(payload["episode_shoe_id"], dtype=np.int64)
    episode_count = len(episode_shoe)
    marker, episode_frame_confidence, marker_diagnostics = cached_marker_frames(
        [path.expanduser() for path in args.data_dir], episode_count
    )
    sample_episode = np.asarray(payload["episode_index"], dtype=np.int64)
    representative = np.asarray(
        [np.flatnonzero(sample_episode == episode)[0] for episode in range(episode_count)],
        dtype=np.int64,
    )
    true_current_rotation = pose9_rotation(payload["current_object_pose9"])
    intended_delta_rotation = Rotation.from_rotvec(
        np.asarray(payload["goal_rotation_error_rotvec"], dtype=np.float64)
    ).as_matrix()
    true_goal_rotation_sample = intended_delta_rotation @ true_current_rotation
    true_goal_position_sample = (
        np.asarray(payload["current_object_pose9"][:, :3], dtype=np.float64)
        + np.asarray(payload["goal_translation_error_xyz_m"], dtype=np.float64)
    )
    true_goal_rotation = true_goal_rotation_sample[representative]
    true_goal_position = true_goal_position_sample[representative]
    train_episode_mask = np.isin(
        episode_shoe, np.asarray(FOLDS[int(args.fold)]["train"], dtype=np.int64)
    )
    (
        goal_position_offset,
        goal_rotation_offset,
        goal_calibration_inlier,
    ) = camera_goal_calibration(
        marker,
        true_goal_position,
        true_goal_rotation,
        train_episode_mask,
    )
    predicted_goal_position, predicted_goal_rotation = predict_goal_pose(
        marker, goal_position_offset, goal_rotation_offset
    )

    zeros = np.zeros((len(sample_episode), 3), dtype=np.float32)
    predicted_current_rotation = frame9_rotation(
        np.concatenate((zeros, payload["source_frame6_columns"]), axis=-1)
    )
    train_sample_mask = np.isin(
        np.asarray(payload["shoe_id"], dtype=np.int64),
        np.asarray(FOLDS[int(args.fold)]["train"], dtype=np.int64),
    )
    current_origin_offset = calibrate_current_origin_offset(
        payload["points_a"],
        predicted_current_rotation,
        np.asarray(payload["current_object_pose9"][:, :3], dtype=np.float32),
        train_sample_mask,
    )
    predicted_current_position = predict_current_position(
        payload["points_a"], predicted_current_rotation, current_origin_offset
    )
    relative_translation = (
        predicted_goal_position[sample_episode] - predicted_current_position
    )
    relative_rotation = (
        predicted_goal_rotation[sample_episode]
        @ np.swapaxes(predicted_current_rotation, -1, -2)
    )
    target_frame9 = np.concatenate(
        (
            relative_translation,
            relative_rotation[..., :, 0],
            relative_rotation[..., :, 1],
        ),
        axis=-1,
    ).astype(np.float32)
    goal_frame9 = np.concatenate(
        (
            predicted_goal_position[sample_episode],
            predicted_goal_rotation[sample_episode, :, 0],
            predicted_goal_rotation[sample_episode, :, 1],
        ),
        axis=-1,
    ).astype(np.float32)

    true_relative_translation = np.asarray(
        payload["goal_translation_error_xyz_m"], dtype=np.float32
    )
    true_relative_rotation = intended_delta_rotation
    relative_rotation_error = rotation_error_deg(
        relative_rotation, true_relative_rotation
    )
    diagnostics = {}
    sample_shoe = np.asarray(payload["shoe_id"], dtype=np.int64)
    for split, ids in FOLDS[int(args.fold)].items():
        mask = np.isin(sample_shoe, np.asarray(ids, dtype=np.int64))
        diagnostics[split] = error_summary(
            relative_translation[mask] - true_relative_translation[mask],
            relative_rotation_error[mask],
        )
    goal_diagnostics = {}
    goal_rotation_error = rotation_error_deg(
        predicted_goal_rotation, true_goal_rotation
    )
    for split, ids in FOLDS[int(args.fold)].items():
        mask = np.isin(episode_shoe, np.asarray(ids, dtype=np.int64))
        goal_diagnostics[split] = error_summary(
            predicted_goal_position[mask] - true_goal_position[mask],
            goal_rotation_error[mask],
        )

    output_payload = dict(payload)
    output_payload["target_frame9"] = target_frame9
    # Unlike target_frame9 (the changing current-to-goal error), this frame is
    # constant within an episode.  It is the deployable task coordinate system
    # used by geometry-aligned point/state/action policies.
    output_payload["goal_frame9"] = goal_frame9
    output_payload["goal_frame_confidence"] = episode_frame_confidence[
        sample_episode
    ].astype(np.float32)
    source_frame_confidence = np.asarray(
        payload.get(
            "source_frame_confidence",
            np.ones(len(sample_episode), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    if source_frame_confidence.shape != (len(sample_episode),):
        raise ValueError(
            "source_frame_confidence must have one scalar per policy sample"
        )
    output_payload["target_frame_confidence"] = (
        output_payload["goal_frame_confidence"] * source_frame_confidence
    ).astype(np.float32)
    output_payload["camera_relative_frame_error_rotation_deg_label_only"] = (
        relative_rotation_error.astype(np.float32)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_payload)
    source_metadata_path = args.dataset.with_suffix(".json")
    metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    metadata.update(
        {
            "schema_version": max(int(metadata.get("schema_version", 1)), 6),
            "functional_frame_encoding": "camera_ndf_current_marker_goal_se3_columns_v1",
            "functional_frame_source": "camera-only NDF A frame plus cached B marker frame",
            "policy_goal_frame_encoding": "camera_marker_goal_se3_columns_v1",
            "deployable": True,
            "fold": int(args.fold),
            "data_dirs": [str(path.resolve()) for path in args.data_dir],
            "current_origin_offset_local3": current_origin_offset.tolist(),
            "goal_position_offset_marker_local3": goal_position_offset.tolist(),
            "goal_rotation_offset_marker_local9": goal_rotation_offset.reshape(-1).tolist(),
            "sparse_marker_fallback_episodes": int(
                sum(bool(item["sparse_fallback"]) for item in marker_diagnostics)
            ),
            "confident_target_frame_episodes": int(episode_frame_confidence.sum()),
            "confident_source_frame_samples": int(source_frame_confidence.sum()),
            "confident_combined_frame_samples": int(
                output_payload["target_frame_confidence"].sum()
            ),
            "marker_diagnostics": marker_diagnostics,
            "goal_calibration_inlier_episodes": int(goal_calibration_inlier.sum()),
            "relative_frame_error": diagnostics,
            "goal_frame_error": goal_diagnostics,
        }
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "relative_frame_error": diagnostics,
                "goal_frame_error": goal_diagnostics,
            }
        )
    )


if __name__ == "__main__":
    main()
