#!/usr/bin/env python3
"""Attach a camera goal frame and an oracle relation ceiling for handled mugs.

The segmented target cloud provides an absolute marker frame.  The constant
marker-to-mug rotation below follows directly from the task's declared local
functional matrices; it is not fitted from evaluation poses.  Simulator goal
errors are used only for diagnostics and, in this oracle-ceiling utility, the
current-to-goal relation token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .augment_task_flow_with_oracle_frame import oracle_relative_frame
from .functional_action_frame import frame9_rotation, pose9_rotation


PLACEMENT_CLEARANCE_M = 0.03

# MARKER_FUNCTIONAL_BASE_ROTATION @ CUP_FUNCTIONAL_ROTATION.T.  Keeping this
# value local avoids importing a simulator task (and therefore CUDA/Curobo)
# during offline dataset processing.
MARKER_TO_MUG_RAW_ROTATION = np.asarray(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


def marker_frame9_to_mug_goal_frame9(
    marker_frame9: np.ndarray,
    *,
    clearance_m: float = PLACEMENT_CLEARANCE_M,
) -> np.ndarray:
    """Convert camera marker frames into desired raw mug-pose frames."""

    value = np.asarray(marker_frame9, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 9:
        raise ValueError(f"marker_frame9 must have shape [N,9], got {value.shape}")
    marker_rotation = frame9_rotation(value)
    goal_rotation = marker_rotation @ MARKER_TO_MUG_RAW_ROTATION
    goal_translation = value[:, :3] + float(clearance_m) * marker_rotation[:, :, 2]
    return np.concatenate(
        (
            goal_translation,
            goal_rotation[:, :, 0],
            goal_rotation[:, :, 1],
        ),
        axis=-1,
    ).astype(np.float32)


def camera_goal_errors(
    payload: dict[str, np.ndarray], goal_frame9: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compare camera goals with label-only intended raw object poses."""

    current_pose9 = np.asarray(payload["current_object_pose9"], dtype=np.float64)
    current_rotation = pose9_rotation(current_pose9)
    delta_rotation = Rotation.from_rotvec(
        np.asarray(payload["goal_rotation_error_rotvec"], dtype=np.float64)
    ).as_matrix()
    intended_rotation = delta_rotation @ current_rotation
    intended_translation = current_pose9[:, :3] + np.asarray(
        payload["goal_translation_error_xyz_m"], dtype=np.float64
    )
    predicted_rotation = frame9_rotation(goal_frame9)
    translation_cm = 100.0 * np.linalg.norm(
        np.asarray(goal_frame9)[:, :3] - intended_translation, axis=-1
    )
    rotation_deg = np.rad2deg(
        Rotation.from_matrix(
            predicted_rotation @ np.swapaxes(intended_rotation, -1, -2)
        ).magnitude()
    )
    return translation_cm, rotation_deg


def summarize(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(array.max()),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--train-ids", nargs="+", type=int, default=list(range(6)))
    result.add_argument("--validation-ids", nargs="+", type=int, default=[6, 7])
    result.add_argument("--test-ids", nargs="+", type=int, default=[8, 9])
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
    sets = {name: set(ids) for name, ids in split.items()}
    if any(not ids for ids in sets.values()) or any(
        sets[first] & sets[second]
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("object splits must be nonempty and pairwise disjoint")

    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    if "target_frame9" not in payload:
        raise KeyError("dataset must contain camera marker target_frame9")
    marker_frame9 = np.asarray(payload["target_frame9"], dtype=np.float32)
    goal_frame9 = marker_frame9_to_mug_goal_frame9(marker_frame9)
    translation_cm, rotation_deg = camera_goal_errors(payload, goal_frame9)
    shoe_id = np.asarray(payload["shoe_id"], dtype=np.int64)
    diagnostics = {}
    for name, ids in split.items():
        keep = np.isin(shoe_id, np.asarray(ids, dtype=np.int64))
        if not np.any(keep):
            raise ValueError(f"split {name} contains no samples")
        diagnostics[name] = {
            "samples": int(keep.sum()),
            "translation_error_cm": summarize(translation_cm[keep]),
            "rotation_error_deg": summarize(rotation_deg[keep]),
        }

    output_payload = dict(payload)
    output_payload["marker_frame9_camera"] = marker_frame9
    output_payload["goal_frame9"] = goal_frame9
    output_payload["target_frame9"] = oracle_relative_frame(payload)
    output_payload["target_frame_confidence"] = np.ones(
        len(goal_frame9), dtype=np.float32
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
            "schema_version": max(int(metadata.get("schema_version", 1)), 8),
            "output": str(args.output.resolve()),
            "goal_frame_source": "camera T-marker fit plus declared task-local transform",
            "relative_frame_source": "oracle intended-goal labels; ceiling only",
            "deployable": False,
            "oracle_ceiling_only": True,
            "object_split": {name: list(ids) for name, ids in split.items()},
            "camera_goal_error": diagnostics,
        }
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "camera_goal_error": diagnostics}))


if __name__ == "__main__":
    main()
