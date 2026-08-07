#!/usr/bin/env python3
"""Build a fully camera-derived rotation relation token for hanging mugs.

The current mug rotation comes from a frozen task-aligned equivariant frame
estimator.  The desired mug rotation comes from registering the observed rack
cloud to one training reference rack.  Only that reference episode's goal
rotation label calibrates rack geometry to the task frame.  Test labels are
used for diagnostics only and never enter a prediction.

Translation is deliberately zeroed in this gate.  The resulting archive tests
whether camera geometry alone retains the oracle token's orientation benefit;
it must not be described as a complete camera SE(3) estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .build_grasp_dataset import valid_xyz
from .functional_action_frame import pose9_rotation


PROPER_AXIS_SIGNS = np.asarray(
    [
        np.diag(sign)
        for sign in (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        )
    ],
    dtype=np.float64,
)


def proper_rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``R,t`` such that row-vector points map as ``x @ R.T + t``."""

    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError(f"paired point sets must both be [N,3], got {first.shape}/{second.shape}")
    if len(first) < 3:
        raise ValueError("a rigid fit requires at least three point pairs")
    first_center = first.mean(axis=0)
    second_center = second.mean(axis=0)
    left, _, right = np.linalg.svd(
        (first - first_center).T @ (second - second_center)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = second_center - rotation @ first_center
    return rotation, translation


def trimmed_icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 30,
    trim_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Register two same-object camera clouds from a close pose initialization."""

    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (3,) or second.shape[1:] != (3,):
        raise ValueError(f"ICP point sets must be [N,3], got {first.shape}/{second.shape}")
    if len(first) < 4 or len(second) < 4:
        raise ValueError("ICP requires at least four points in each cloud")
    if int(iterations) <= 0 or not 0.5 <= float(trim_fraction) <= 1.0:
        raise ValueError("iterations must be positive and trim_fraction must lie in [0.5,1]")

    rotation = np.eye(3, dtype=np.float64)
    translation = second.mean(axis=0) - first.mean(axis=0)
    tree = cKDTree(second)
    residual = np.empty((len(first),), dtype=np.float64)
    keep_count = max(3, int(np.floor(float(trim_fraction) * len(first))))
    for _ in range(int(iterations)):
        transformed = first @ rotation.T + translation
        residual, neighbors = tree.query(transformed)
        keep = np.argpartition(residual, keep_count - 1)[:keep_count]
        delta_rotation, delta_translation = proper_rigid_fit(
            transformed[keep], second[neighbors[keep]]
        )
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
    transformed = first @ rotation.T + translation
    residual, _ = tree.query(transformed)
    cost = float(np.mean(np.partition(residual, keep_count - 1)[:keep_count]))
    return rotation, translation, cost


def compose_rotation_token(
    predicted_current: np.ndarray, predicted_goal: np.ndarray
) -> np.ndarray:
    current = np.asarray(predicted_current, dtype=np.float64)
    goal = np.asarray(predicted_goal, dtype=np.float64)
    if current.shape != goal.shape or current.ndim != 3 or current.shape[-2:] != (3, 3):
        raise ValueError(f"current/goal rotations must be matching [N,3,3], got {current.shape}/{goal.shape}")
    relative = goal @ np.swapaxes(current, -1, -2)
    return np.concatenate(
        (
            np.zeros((len(relative), 3), dtype=np.float64),
            relative[..., :, 0],
            relative[..., :, 1],
        ),
        axis=-1,
    ).astype(np.float32)


def compose_goal_rotation_frame(predicted_goal: np.ndarray) -> np.ndarray:
    """Encode the camera-derived absolute goal rotation with zero translation.

    The translation is deliberately unavailable in this rotation-only gate.
    ``goal_action`` policies use only the rotation to canonicalize delta actions,
    so this remains deployable without simulator position state.
    """

    goal = np.asarray(predicted_goal, dtype=np.float64)
    if goal.ndim != 3 or goal.shape[-2:] != (3, 3):
        raise ValueError(f"goal rotations must be [N,3,3], got {goal.shape}")
    return np.concatenate(
        (
            np.zeros((len(goal), 3), dtype=np.float64),
            goal[..., :, 0],
            goal[..., :, 1],
        ),
        axis=-1,
    ).astype(np.float32)


def stabilize_frame_symmetry(
    frames: np.ndarray,
    episode_ids: np.ndarray,
    frame_indices: np.ndarray,
    *,
    maximum_step_deg: float = 30.0,
    minimum_margin_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve proper axis-sign ambiguity using only within-episode continuity.

    Every candidate remains a right-handed SO(3) frame.  The first observation
    is unchanged; later observations select the candidate closest to the
    previous stabilized frame.  Observable step size and runner-up margin
    provide a label-free confidence signal.
    """

    value = np.asarray(frames, dtype=np.float64)
    episodes = np.asarray(episode_ids, dtype=np.int64)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if value.ndim != 3 or value.shape[-2:] != (3, 3):
        raise ValueError(f"frames must be [N,3,3], got {value.shape}")
    if episodes.shape != (len(value),) or indices.shape != (len(value),):
        raise ValueError("episode_ids and frame_indices must have one value per frame")
    if float(maximum_step_deg) <= 0.0 or float(minimum_margin_deg) < 0.0:
        raise ValueError("temporal symmetry thresholds must be non-negative")

    output = value.copy()
    confident = np.zeros((len(value),), dtype=np.float32)
    variants = np.zeros((len(value),), dtype=np.int64)
    step_deg = np.full((len(value),), np.nan, dtype=np.float32)
    margin_deg = np.full((len(value),), np.nan, dtype=np.float32)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(indices[rows], kind="stable")]
        previous = output[int(rows[0])]
        for row in rows[1:]:
            candidates = value[int(row)][None] @ PROPER_AXIS_SIGNS
            distance = np.rad2deg(
                Rotation.from_matrix(candidates @ previous.T).magnitude()
            )
            order = np.argsort(distance)
            selected = int(order[0])
            runner_up = int(order[1])
            variants[int(row)] = selected
            output[int(row)] = candidates[selected]
            step_deg[int(row)] = float(distance[selected])
            margin_deg[int(row)] = float(distance[runner_up] - distance[selected])
            confident[int(row)] = float(
                distance[selected] <= float(maximum_step_deg)
                and distance[runner_up] - distance[selected]
                >= float(minimum_margin_deg)
            )
            previous = output[int(row)]
    return output, confident, variants, step_deg, margin_deg


def rotation_summary(error: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(error, dtype=np.float64)
    return {
        "samples": int(len(values)),
        "mean_deg": float(values.mean()),
        "median_deg": float(np.median(values)),
        "p90_deg": float(np.percentile(values, 90)),
        "max_deg": float(values.max()),
        "flip_over_90deg": float(np.mean(values >= 90.0)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--current-frame-predictions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--reference-episode", type=int, required=True)
    result.add_argument("--train-ids", nargs="+", type=int, required=True)
    result.add_argument("--validation-ids", nargs="+", type=int, required=True)
    result.add_argument("--test-ids", nargs="+", type=int, required=True)
    result.add_argument("--registration-points", type=int, default=512)
    result.add_argument("--iterations", type=int, default=30)
    result.add_argument("--trim-fraction", type=float, default=0.8)
    result.add_argument("--temporal-symmetry-stabilization", action="store_true")
    result.add_argument("--temporal-maximum-step-deg", type=float, default=30.0)
    result.add_argument("--temporal-minimum-margin-deg", type=float, default=30.0)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    split = {
        "train": tuple(args.train_ids),
        "validation": tuple(args.validation_ids),
        "test": tuple(args.test_ids),
    }
    split_sets = {name: set(values) for name, values in split.items()}
    if any(not values for values in split_sets.values()) or any(
        split_sets[first] & split_sets[second]
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("train/validation/test IDs must be nonempty and disjoint")

    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.current_frame_predictions, allow_pickle=False) as archive:
        predictions = {key: np.asarray(archive[key]) for key in archive.files}
    samples = len(payload["shoe_id"])
    current = np.asarray(predictions["frames"], dtype=np.float64)
    if current.shape != (samples, 3, 3):
        raise ValueError(f"current frames must be {(samples, 3, 3)}, got {current.shape}")
    for key in ("episode_id", "shoe_id"):
        if key in predictions and not np.array_equal(predictions[key], payload[key]):
            raise ValueError(f"current frame predictions are not row-aligned: {key}")

    episode_ids = np.asarray(payload["episode_id"], dtype=np.int64)
    shoe_ids = np.asarray(payload["shoe_id"], dtype=np.int64)
    raw_current = current.copy()
    temporal_confidence = np.zeros((samples,), dtype=np.float32)
    symmetry_variant = np.zeros((samples,), dtype=np.int64)
    temporal_step_deg = np.full((samples,), np.nan, dtype=np.float32)
    temporal_margin_deg = np.full((samples,), np.nan, dtype=np.float32)
    if args.temporal_symmetry_stabilization:
        current, temporal_confidence, symmetry_variant, temporal_step_deg, temporal_margin_deg = (
            stabilize_frame_symmetry(
                current,
                episode_ids,
                np.asarray(payload["frame_index"], dtype=np.int64),
                maximum_step_deg=float(args.temporal_maximum_step_deg),
                minimum_margin_deg=float(args.temporal_minimum_margin_deg),
            )
        )
    episodes = sorted(int(value) for value in np.unique(episode_ids))
    reference = int(args.reference_episode)
    if reference not in episodes:
        raise ValueError(f"reference episode {reference} is absent from the dataset")
    reference_row = int(np.flatnonzero(episode_ids == reference)[0])
    if int(shoe_ids[reference_row]) not in split_sets["train"]:
        raise ValueError("reference episode must belong to a training object")

    clouds: dict[int, np.ndarray] = {}
    for episode in episodes:
        path = args.data_dir / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as root:
            points = valid_xyz(root["object_pointcloud/{B}"][0])
        count = min(int(args.registration_points), len(points))
        if count < 4:
            raise ValueError(f"episode {episode} rack cloud has only {len(points)} points")
        indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
        clouds[episode] = np.asarray(points[indices], dtype=np.float64)

    true_current = pose9_rotation(payload["current_object_pose9"])
    intended_delta = Rotation.from_rotvec(
        np.asarray(payload["goal_rotation_error_rotvec"], dtype=np.float64)
    ).as_matrix()
    # This is the sole task-state value used to make predictions.  It is a
    # training calibration label attached to the frozen reference episode.
    reference_goal = intended_delta[reference_row] @ true_current[reference_row]
    goal_by_episode: dict[int, np.ndarray] = {}
    registration_cost: dict[int, float] = {}
    for episode in episodes:
        rack_rotation, _, cost = trimmed_icp(
            clouds[reference],
            clouds[episode],
            iterations=int(args.iterations),
            trim_fraction=float(args.trim_fraction),
        )
        goal_by_episode[episode] = rack_rotation @ reference_goal
        registration_cost[episode] = cost
    predicted_goal = np.stack([goal_by_episode[int(value)] for value in episode_ids])
    target_frame9 = compose_rotation_token(current, predicted_goal)
    goal_frame9 = compose_goal_rotation_frame(predicted_goal)

    train_episodes = [
        episode
        for episode in episodes
        if int(shoe_ids[np.flatnonzero(episode_ids == episode)[0]]) in split_sets["train"]
    ]
    registration_threshold = float(
        np.percentile([registration_cost[episode] for episode in train_episodes], 95)
    )
    icp_confidence = np.asarray(
        [float(registration_cost[int(value)] <= registration_threshold) for value in episode_ids],
        dtype=np.float32,
    )
    current_confidence = np.asarray(
        predictions.get("frame_confidence", np.ones(samples)), dtype=np.float32
    )
    if current_confidence.shape != (samples,):
        raise ValueError("current frame confidence must have one value per row")
    if args.temporal_symmetry_stabilization:
        current_confidence = np.maximum(current_confidence, temporal_confidence)
    confidence = current_confidence * icp_confidence

    output = dict(payload)
    output["target_frame9"] = target_frame9
    output["goal_frame9"] = goal_frame9
    output["target_frame_confidence"] = confidence
    output["source_frame6_columns"] = np.concatenate(
        (current[..., :, 0], current[..., :, 1]), axis=-1
    ).astype(np.float32)
    output["source_frame_confidence"] = current_confidence
    output["camera_rack_registration_cost_m"] = np.asarray(
        [registration_cost[int(value)] for value in episode_ids], dtype=np.float32
    )
    output["camera_rack_registration_confidence"] = icp_confidence
    if args.temporal_symmetry_stabilization:
        output["camera_current_frame_raw"] = raw_current.astype(np.float32)
        output["camera_current_frame_symmetry_variant"] = symmetry_variant
        output["camera_current_frame_temporal_confidence"] = temporal_confidence
        output["camera_current_frame_temporal_step_deg"] = temporal_step_deg
        output["camera_current_frame_temporal_margin_deg"] = temporal_margin_deg

    predicted_relative = predicted_goal @ np.swapaxes(current, -1, -2)
    error = np.rad2deg(
        Rotation.from_matrix(
            predicted_relative @ np.swapaxes(intended_delta, -1, -2)
        ).magnitude()
    )
    diagnostics = {
        name: rotation_summary(error[np.isin(shoe_ids, ids)])
        for name, ids in split.items()
    }
    diagnostics["confidence_acceptance"] = {
        name: float(confidence[np.isin(shoe_ids, ids)].mean())
        for name, ids in split.items()
    }
    diagnostics["reference_episode"] = reference
    diagnostics["reference_object_id"] = int(shoe_ids[reference_row])
    diagnostics["registration_train_p95_m"] = registration_threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        "schema_version": 1,
        "method": "camera_mug_equivariant_frame_plus_reference_rack_trimmed_icp",
        "dataset": str(args.dataset.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "current_frame_predictions": str(args.current_frame_predictions.resolve()),
        "reference_episode": reference,
        "split": {name: list(ids) for name, ids in split.items()},
        "translation_token": "exact zero; no simulator translation",
        "rotation_token": "fully camera-derived after one training reference calibration",
        "goal_frame9": (
            "camera-derived rack goal rotation with exact-zero translation; "
            "deployable for goal_action canonicalization"
        ),
        "temporal_symmetry_stabilization": bool(args.temporal_symmetry_stabilization),
        "temporal_maximum_step_deg": float(args.temporal_maximum_step_deg),
        "temporal_minimum_margin_deg": float(args.temporal_minimum_margin_deg),
        "deployable_rotation": True,
        "deployable_full_se3": False,
        "diagnostics_use_task_state_labels": True,
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "diagnostics": diagnostics}))


if __name__ == "__main__":
    main()
