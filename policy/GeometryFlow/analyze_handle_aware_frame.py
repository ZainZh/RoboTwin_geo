#!/usr/bin/env python3
"""Decompose a two-axis mug frame estimator into functional-axis ablations.

The handled-mug meshes use local +Z as the directed handle axis and local +Y
as the cup-body axis.  The evaluated frame head predicts both.  Single-axis
controls reconstruct the missing axis from a train-object-only mean prior;
the shuffled-handle control preserves the predicted body axis while replacing
handle evidence with a different test episode at matched normalized time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .functional_action_frame import pose9_rotation


def normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def frame_from_vertical_handle(vertical: np.ndarray, handle: np.ndarray) -> np.ndarray:
    """Build R=[X,Y,Z] from body Y and directed handle Z."""

    vertical = normalize(vertical)
    handle = handle - vertical * np.sum(vertical * handle, axis=-1, keepdims=True)
    handle = normalize(handle)
    side = normalize(np.cross(vertical, handle))
    # Recompute Y after projection to avoid numerical non-orthogonality.
    vertical = normalize(np.cross(handle, side))
    return np.stack((side, vertical, handle), axis=-1)


def single_axis_frame(
    axis: np.ndarray, missing_axis_prior: np.ndarray, *, axis_name: str
) -> np.ndarray:
    prior = np.broadcast_to(
        normalize(np.asarray(missing_axis_prior, dtype=np.float64).reshape(1, 3)),
        np.asarray(axis).shape,
    )
    if axis_name == "handle":
        return frame_from_vertical_handle(prior, axis)
    if axis_name == "vertical":
        return frame_from_vertical_handle(axis, prior)
    raise ValueError(f"unknown axis {axis_name!r}")


def shuffle_episode_axis(
    axis: np.ndarray, episode_ids: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Cross-episode intervention with approximately matched temporal phase."""

    result = np.asarray(axis, dtype=np.float64).copy()
    subset_episodes = sorted(int(value) for value in np.unique(episode_ids[indices]))
    if len(subset_episodes) < 2:
        raise ValueError("shuffled-handle control requires at least two test episodes")
    donor = {
        episode: subset_episodes[(position + 1) % len(subset_episodes)]
        for position, episode in enumerate(subset_episodes)
    }
    for episode in subset_episodes:
        target_rows = indices[episode_ids[indices] == episode]
        donor_rows = indices[episode_ids[indices] == donor[episode]]
        if len(target_rows) == 1:
            positions = np.zeros((1,), dtype=np.int64)
        else:
            positions = np.rint(
                np.linspace(0, len(donor_rows) - 1, len(target_rows))
            ).astype(np.int64)
        result[target_rows] = axis[donor_rows[positions]]
    return result


def signed_axis_angles(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    cosine = np.sum(normalize(predicted) * normalize(target), axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def so3_angles(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = predicted @ np.swapaxes(target, -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def aggregate_by_episode(
    values: np.ndarray, episode_ids: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            np.mean(values[episode_ids[indices] == episode])
            for episode in np.unique(episode_ids[indices])
        ],
        dtype=np.float64,
    )


def frame_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    episode_ids: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int]:
    predicted_subset = predicted[indices]
    target_subset = target[indices]
    full = so3_angles(predicted_subset, target_subset)
    handle = signed_axis_angles(predicted_subset[..., :, 2], target_subset[..., :, 2])
    vertical = signed_axis_angles(predicted_subset[..., :, 1], target_subset[..., :, 1])
    episode_full = aggregate_by_episode(full, episode_ids, indices)
    episode_handle = aggregate_by_episode(handle, episode_ids, indices)
    episode_vertical = aggregate_by_episode(vertical, episode_ids, indices)
    return {
        "frames": int(len(indices)),
        "episodes": int(len(np.unique(episode_ids[indices]))),
        "so3_mean_deg": float(np.mean(full)),
        "so3_median_deg": float(np.median(full)),
        "so3_p90_deg": float(np.percentile(full, 90)),
        "so3_flip_rate_over_90": float(np.mean(full >= 90.0)),
        "handle_axis_mean_deg": float(np.mean(handle)),
        "vertical_axis_mean_deg": float(np.mean(vertical)),
        "episode_mean_so3_deg": float(np.mean(episode_full)),
        "episode_mean_handle_axis_deg": float(np.mean(episode_handle)),
        "episode_mean_vertical_axis_deg": float(np.mean(episode_vertical)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--frames", type=Path, nargs="+", required=True)
    result.add_argument("--train-ids", type=int, nargs="+", required=True)
    result.add_argument("--test-ids", type=int, nargs="+", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        target = pose9_rotation(np.asarray(archive["current_object_pose9"]))
        object_ids = np.asarray(archive["shoe_id"], dtype=np.int64)
        episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
    train = np.flatnonzero(np.isin(object_ids, args.train_ids))
    test = np.flatnonzero(np.isin(object_ids, args.test_ids))
    if len(train) == 0 or len(test) == 0:
        raise ValueError("train/test object split has no rows")
    if set(args.train_ids) & set(args.test_ids):
        raise ValueError("train and test object IDs must be disjoint")

    # These priors use training identities only and therefore cannot encode a
    # held-out object's instantaneous functional orientation.
    vertical_prior = normalize(np.mean(target[train, :, 1], axis=0))
    handle_prior = normalize(np.mean(target[train, :, 2], axis=0))
    runs = []
    for frame_path in args.frames:
        with np.load(frame_path, allow_pickle=False) as archive:
            predicted = np.asarray(archive["frames"], dtype=np.float64)
        if predicted.shape != target.shape:
            raise ValueError(
                f"{frame_path} frames {predicted.shape} do not match {target.shape}"
            )
        predicted_vertical = predicted[:, :, 1]
        predicted_handle = predicted[:, :, 2]
        shuffled_handle = shuffle_episode_axis(predicted_handle, episode_ids, test)
        conditions = {
            "two_axis": predicted,
            "handle_only_train_prior": single_axis_frame(
                predicted_handle, vertical_prior, axis_name="handle"
            ),
            "vertical_only_train_prior": single_axis_frame(
                predicted_vertical, handle_prior, axis_name="vertical"
            ),
            "vertical_plus_shuffled_handle": frame_from_vertical_handle(
                predicted_vertical, shuffled_handle
            ),
            "oracle_handle_train_vertical_prior": single_axis_frame(
                target[:, :, 2], vertical_prior, axis_name="handle"
            ),
            "oracle_vertical_train_handle_prior": single_axis_frame(
                target[:, :, 1], handle_prior, axis_name="vertical"
            ),
        }
        runs.append(
            {
                "frames": str(frame_path),
                "conditions": {
                    name: frame_metrics(value, target, episode_ids, test)
                    for name, value in conditions.items()
                },
            }
        )
    summary = {}
    for condition in runs[0]["conditions"]:
        metrics = runs[0]["conditions"][condition]
        summary[condition] = {
            key: {
                "mean": float(np.mean([run["conditions"][condition][key] for run in runs])),
                "sample_sd": float(
                    np.std(
                        [run["conditions"][condition][key] for run in runs],
                        ddof=1,
                    )
                )
                if len(runs) > 1
                else 0.0,
            }
            for key in metrics
            if key not in {"frames", "episodes"}
        }
    result = {
        "schema_version": 1,
        "protocol": "object_disjoint_handle_vertical_axis_ablation_v1",
        "dataset": str(args.dataset),
        "train_ids": list(args.train_ids),
        "test_ids": list(args.test_ids),
        "axis_semantics": {
            "handle": "039_mug local +Z, directed",
            "vertical": "039_mug local +Y, directed",
            "frame": "R=[X,Y,Z], X=Y cross Z",
        },
        "runs": runs,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
