#!/usr/bin/env python3
"""Replace oracle handled-mug relations with camera-only NDF relations.

The absolute desired mug pose comes from the camera T-marker goal already
attached to the dataset.  The current rotation comes from a category-specific
NDF frame ensemble.  A single centroid-to-object-origin offset is calibrated
on training objects only; deployment then needs only A/B point clouds and that
frozen constant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .augment_task_flow_with_camera_ndf_frame import (
    calibrate_current_origin_offset,
    error_summary,
    predict_current_position,
    rotation_error_deg,
)
from .functional_action_frame import frame9_rotation, pose9_rotation
from .train_ndf_functional_frame_head import resolve_object_split


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--frame-predictions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fold", type=int, default=0)
    result.add_argument("--train-ids", nargs="+", type=int, required=True)
    result.add_argument("--validation-ids", nargs="+", type=int, required=True)
    result.add_argument("--test-ids", nargs="+", type=int, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def build_camera_ndf_relation(
    payload: dict[str, np.ndarray],
    predicted_current_rotation: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    required = {
        "goal_frame9",
        "points_a",
        "current_object_pose9",
        "goal_translation_error_xyz_m",
        "goal_rotation_error_rotvec",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"dataset is missing fields: {missing}")
    predicted_rotation = np.asarray(predicted_current_rotation, dtype=np.float32)
    samples = len(payload["shoe_id"])
    if predicted_rotation.shape != (samples, 3, 3):
        raise ValueError(
            f"predicted current frames must have {(samples, 3, 3)}, "
            f"got {predicted_rotation.shape}"
        )
    current_origin_offset = calibrate_current_origin_offset(
        payload["points_a"],
        predicted_rotation,
        np.asarray(payload["current_object_pose9"][:, :3], dtype=np.float32),
        np.asarray(train_mask, dtype=bool),
    )
    predicted_current_position = predict_current_position(
        payload["points_a"], predicted_rotation, current_origin_offset
    )
    goal_frame9 = np.asarray(payload["goal_frame9"], dtype=np.float32)
    goal_rotation = frame9_rotation(goal_frame9)
    relative_translation = goal_frame9[:, :3] - predicted_current_position
    relative_rotation = goal_rotation @ np.swapaxes(predicted_rotation, -1, -2)
    relative_frame9 = np.concatenate(
        (
            relative_translation,
            relative_rotation[:, :, 0],
            relative_rotation[:, :, 1],
        ),
        axis=-1,
    ).astype(np.float32)

    true_current_rotation = pose9_rotation(payload["current_object_pose9"])
    true_relative_rotation = Rotation.from_rotvec(
        np.asarray(payload["goal_rotation_error_rotvec"], dtype=np.float64)
    ).as_matrix()
    frame_rotation_error = rotation_error_deg(
        predicted_rotation, true_current_rotation
    )
    relative_rotation_error = rotation_error_deg(
        relative_rotation, true_relative_rotation
    )
    relative_translation_error = relative_translation - np.asarray(
        payload["goal_translation_error_xyz_m"], dtype=np.float32
    )
    output = dict(payload)
    output["target_frame9"] = relative_frame9
    output["source_frame6_columns"] = np.concatenate(
        (predicted_rotation[:, :, 0], predicted_rotation[:, :, 1]), axis=-1
    ).astype(np.float32)
    output["source_frame_error_deg_label_only"] = frame_rotation_error.astype(
        np.float32
    )
    output["camera_relative_frame_error_rotation_deg_label_only"] = (
        relative_rotation_error.astype(np.float32)
    )
    diagnostics = {
        "current_origin_offset_local3": current_origin_offset,
        "relative_translation_error": relative_translation_error,
        "relative_rotation_error_deg": relative_rotation_error,
        "source_rotation_error_deg": frame_rotation_error,
    }
    return output, diagnostics


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    split = resolve_object_split(
        int(args.fold),
        train_ids=args.train_ids,
        validation_ids=args.validation_ids,
        test_ids=args.test_ids,
    )
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.frame_predictions, allow_pickle=False) as archive:
        predictions = {key: np.asarray(archive[key]) for key in archive.files}
    for key in ("shoe_id", "episode_id"):
        if key in predictions and not np.array_equal(payload[key], predictions[key]):
            raise ValueError(f"frame prediction row alignment mismatch for {key}")
    if "frames" not in predictions:
        raise KeyError("frame prediction archive is missing frames")
    shoe_id = np.asarray(payload["shoe_id"], dtype=np.int64)
    train_mask = np.isin(shoe_id, np.asarray(split["train"], dtype=np.int64))
    output, raw_diagnostics = build_camera_ndf_relation(
        payload, predictions["frames"], train_mask
    )
    samples = len(shoe_id)
    confidence = np.asarray(
        predictions.get("frame_confidence", np.ones(samples)), dtype=np.float32
    )
    if confidence.shape != (samples,) or not np.all(np.isfinite(confidence)):
        raise ValueError("frame confidence must be one finite scalar per sample")
    goal_confidence = np.asarray(
        payload.get("goal_frame_confidence", np.ones(samples)), dtype=np.float32
    )
    output["source_frame_confidence"] = confidence
    output["target_frame_confidence"] = confidence * goal_confidence
    if "frame_disagreement_deg" in predictions:
        output["source_frame_disagreement_deg"] = np.asarray(
            predictions["frame_disagreement_deg"], dtype=np.float32
        )

    diagnostics = {}
    for name, ids in split.items():
        keep = np.isin(shoe_id, np.asarray(ids, dtype=np.int64))
        diagnostics[name] = {
            "relative": error_summary(
                raw_diagnostics["relative_translation_error"][keep],
                raw_diagnostics["relative_rotation_error_deg"][keep],
            ),
            "source_rotation_deg": {
                "mean": float(raw_diagnostics["source_rotation_error_deg"][keep].mean()),
                "median": float(
                    np.median(raw_diagnostics["source_rotation_error_deg"][keep])
                ),
                "p90": float(
                    np.percentile(raw_diagnostics["source_rotation_error_deg"][keep], 90)
                ),
            },
            "confidence_acceptance": float(confidence[keep].mean()),
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
            "schema_version": max(int(metadata.get("schema_version", 1)), 9),
            "output": str(args.output.resolve()),
            "functional_frame_source": "camera-only NDF A frame plus camera T-marker goal",
            "deployable": True,
            "frame_predictions": str(args.frame_predictions.resolve()),
            "object_split": {name: list(ids) for name, ids in split.items()},
            "current_origin_offset_local3": raw_diagnostics[
                "current_origin_offset_local3"
            ].tolist(),
            "camera_relative_error": diagnostics,
        }
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "diagnostics": diagnostics}))


if __name__ == "__main__":
    main()
