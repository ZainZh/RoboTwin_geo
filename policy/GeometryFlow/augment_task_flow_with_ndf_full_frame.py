#!/usr/bin/env python3
"""Attach a single-checkpoint NDF functional frame to a policy dataset.

The NDF model sees only the current A camera cloud.  Simulator object pose is
used here only as a supervised diagnostic and to retain the intended-goal pose
for the partial-perception policy ceiling.  A subsequent camera-goal augmenter
can replace that label-derived goal and produce the deployable camera-only
relative SE(3) token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .functional_action_frame import pose9_rotation
from .train_task_flow_benchmark import FOLDS


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument(
        "--goal-label-dataset",
        type=Path,
        help=(
            "Optional row-aligned dataset providing intended-goal labels when "
            "the camera policy archive predates those fields."
        ),
    )
    result.add_argument("--frame-predictions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.asarray(predicted) @ np.swapaxes(np.asarray(target), -1, -2)
    return np.rad2deg(
        Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    ).reshape(relative.shape[:-2])


def frame_summary(error: np.ndarray) -> dict:
    value = np.asarray(error, dtype=np.float64)
    return {
        "samples": int(len(value)),
        "mean_deg": float(value.mean()),
        "median_deg": float(np.median(value)),
        "p90_deg": float(np.percentile(value, 90)),
        "max_deg": float(value.max()),
        "within_15deg": float(np.mean(value <= 15.0)),
        "within_30deg": float(np.mean(value <= 30.0)),
        "flip_over_90deg": float(np.mean(value >= 90.0)),
    }


def validate_alignment(
    payload: dict[str, np.ndarray], predictions: dict[str, np.ndarray]
) -> None:
    if "frames" not in predictions:
        raise KeyError("frame prediction archive is missing frames")
    frames = np.asarray(predictions["frames"])
    if frames.shape != (len(payload["shoe_id"]), 3, 3):
        raise ValueError(
            f"predicted frames must be {(len(payload['shoe_id']), 3, 3)}, "
            f"got {frames.shape}"
        )
    for key in ("shoe_id", "episode_id"):
        if key in predictions and not np.array_equal(payload[key], predictions[key]):
            raise ValueError(f"frame prediction row alignment mismatch for {key}")


def augment_payload(
    payload: dict[str, np.ndarray], predicted_current: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    required = {
        "current_object_pose9",
        "goal_translation_error_xyz_m",
        "goal_rotation_error_rotvec",
        "shoe_id",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"source dataset is missing fields: {missing}")
    predicted_current = np.asarray(predicted_current, dtype=np.float64)
    true_current = pose9_rotation(payload["current_object_pose9"])
    intended_delta = Rotation.from_rotvec(
        np.asarray(payload["goal_rotation_error_rotvec"], dtype=np.float64)
    ).as_matrix()
    true_goal = intended_delta @ true_current
    predicted_delta = true_goal @ np.swapaxes(predicted_current, -1, -2)
    target_frame9 = np.concatenate(
        (
            np.asarray(payload["goal_translation_error_xyz_m"], dtype=np.float32),
            predicted_delta[..., :, 0],
            predicted_delta[..., :, 1],
        ),
        axis=-1,
    ).astype(np.float32)
    frame_error = rotation_error_deg(predicted_current, true_current)
    output = dict(payload)
    output["target_frame9"] = target_frame9
    output["source_frame6_columns"] = np.concatenate(
        (predicted_current[..., :, 0], predicted_current[..., :, 1]), axis=-1
    ).astype(np.float32)
    output["source_frame_error_deg_label_only"] = frame_error.astype(np.float32)
    return output, frame_error


def main() -> None:
    args = parser().parse_args()
    if args.fold < 0 or args.fold >= len(FOLDS):
        raise ValueError(f"fold must lie in [0,{len(FOLDS) - 1}]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    if args.goal_label_dataset is not None:
        with np.load(args.goal_label_dataset, allow_pickle=False) as archive:
            labels = {key: np.asarray(archive[key]) for key in archive.files}
        for key in ("episode_id", "frame_index", "shoe_id"):
            if key not in payload or key not in labels:
                raise KeyError(f"row-alignment validation requires {key}")
            if not np.array_equal(payload[key], labels[key]):
                raise ValueError(
                    f"goal-label dataset is not row-aligned with policy data: {key}"
                )
        for key in (
            "goal_translation_error_xyz_m",
            "goal_rotation_error_rotvec",
        ):
            if key not in labels:
                raise KeyError(f"goal-label dataset is missing {key}")
            if labels[key].shape != (len(payload["shoe_id"]), 3):
                raise ValueError(f"goal-label field {key} has shape {labels[key].shape}")
            payload[key] = labels[key].astype(np.float32)
    with np.load(args.frame_predictions, allow_pickle=False) as archive:
        predictions = {key: np.asarray(archive[key]) for key in archive.files}
    validate_alignment(payload, predictions)
    output, frame_error = augment_payload(payload, predictions["frames"])
    if "frame_confidence" in predictions:
        confidence = np.asarray(predictions["frame_confidence"], dtype=np.float32)
        if confidence.shape != (len(payload["shoe_id"]),):
            raise ValueError(
                f"frame_confidence must be {(len(payload['shoe_id']),)}, "
                f"got {confidence.shape}"
            )
        output["source_frame_confidence"] = confidence
    if "frame_disagreement_deg" in predictions:
        output["source_frame_disagreement_deg"] = np.asarray(
            predictions["frame_disagreement_deg"], dtype=np.float32
        )
    shoe_id = np.asarray(payload["shoe_id"], dtype=np.int64)
    diagnostics = {
        split: frame_summary(
            frame_error[np.isin(shoe_id, np.asarray(ids, dtype=np.int64))]
        )
        for split, ids in FOLDS[int(args.fold)].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    source_metadata_path = args.dataset.with_suffix(".json")
    metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    metadata.update(
        {
            "schema_version": max(int(metadata.get("schema_version", 1)), 5),
            "functional_frame_encoding": "single_ndf_current_oracle_goal_se3_columns_v1",
            "functional_frame_source": (
                "partial ceiling: single-checkpoint NDF current frame; "
                "label-only intended goal"
            ),
            "deployable": False,
            "frame_predictions": str(args.frame_predictions.resolve()),
            "goal_label_dataset": (
                None
                if args.goal_label_dataset is None
                else str(args.goal_label_dataset.resolve())
            ),
            "source_frame_confidence": (
                "NDF ensemble disagreement calibrated on training objects"
                if "frame_confidence" in predictions
                else "not provided"
            ),
            "fold": int(args.fold),
            "frame_error_deg": diagnostics,
        }
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "frame_error_deg": diagnostics}))


if __name__ == "__main__":
    main()
