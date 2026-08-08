#!/usr/bin/env python3
"""Project a relative SE(3) label onto a task-observable symmetry quotient.

For axially symmetric placement, yaw about the target vertical axis is not part
of the task.  This utility retains the current-to-goal translation and replaces
the full rotation with the minimum swing that aligns the task axes, thereby
discarding the unidentifiable twist.  Local A/B point tokens remain unchanged.
The output is consumed through the ordinary ``dataset_relative`` frame
interface, so no policy architecture or action label is changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _pose9_rotation(pose9: np.ndarray) -> np.ndarray:
    value = np.asarray(pose9, dtype=np.float64)
    first = value[:, 3:6]
    second = value[:, 6:9]
    first /= np.linalg.norm(first, axis=-1, keepdims=True)
    second -= first * np.sum(first * second, axis=-1, keepdims=True)
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def _minimal_axis_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = []
    for source_axis, target_axis in zip(source, target):
        source_axis = source_axis / np.linalg.norm(source_axis)
        target_axis = target_axis / np.linalg.norm(target_axis)
        cross = np.cross(source_axis, target_axis)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(np.dot(source_axis, target_axis), -1.0, 1.0))
        if sine < 1.0e-8 and cosine > 0.0:
            matrix = np.eye(3, dtype=np.float64)
        elif sine < 1.0e-8:
            helper = np.asarray([1.0, 0.0, 0.0])
            if abs(float(np.dot(helper, source_axis))) > 0.9:
                helper = np.asarray([0.0, 1.0, 0.0])
            axis = np.cross(source_axis, helper)
            axis /= np.linalg.norm(axis)
            matrix = Rotation.from_rotvec(np.pi * axis).as_matrix()
        else:
            skew = np.asarray(
                [
                    [0.0, -cross[2], cross[1]],
                    [cross[2], 0.0, -cross[0]],
                    [-cross[1], cross[0], 0.0],
                ]
            )
            matrix = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
        result.append(matrix)
    return np.stack(result)


def symmetry_quotient_frame9(
    translation: np.ndarray,
    rotation_vector: np.ndarray,
    current_pose9: np.ndarray,
    task_axis_local: np.ndarray,
) -> np.ndarray:
    value = np.asarray(translation, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("translation must have shape [samples,3]")
    current_rotation = _pose9_rotation(current_pose9)
    full_delta = Rotation.from_rotvec(
        np.asarray(rotation_vector, dtype=np.float64)
    ).as_matrix()
    local_axis = np.asarray(task_axis_local, dtype=np.float64).reshape(3)
    local_axis /= np.linalg.norm(local_axis)
    current_axis = current_rotation @ local_axis
    desired_axis = full_delta @ current_axis[..., None]
    swing = _minimal_axis_rotation(current_axis, desired_axis[..., 0])
    return np.concatenate(
        (value, swing[..., :, 0], swing[..., :, 1]), axis=-1
    ).astype(np.float32)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--task-axis-local",
        default="0,-1,0",
        help="Operated-object local symmetry axis as x,y,z.",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    required = (
        "goal_translation_error_xyz_m",
        "goal_rotation_error_rotvec",
        "current_object_pose9",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"input dataset lacks {missing}")
    task_axis = np.asarray(
        [float(value) for value in args.task_axis_local.split(",")],
        dtype=np.float64,
    )
    if task_axis.shape != (3,) or np.linalg.norm(task_axis) < 1.0e-8:
        raise ValueError("task-axis-local must contain three nonzero values")
    payload["target_frame9"] = symmetry_quotient_frame9(
        payload["goal_translation_error_xyz_m"],
        payload["goal_rotation_error_rotvec"],
        payload["current_object_pose9"],
        task_axis,
    )
    payload["target_frame_confidence"] = np.ones(
        len(payload["target_frame9"]), dtype=np.float32
    )
    payload["relation_symmetry_mode"] = np.asarray(
        "translation_plus_axis_swing_quotient_v1"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        {
            "output": str(args.output),
            "samples": int(len(payload["target_frame9"])),
            "symmetry_mode": str(payload["relation_symmetry_mode"]),
        }
    )


if __name__ == "__main__":
    main()
