#!/usr/bin/env python3
"""Add deterministic SE(3) recovery states to a task-flow action dataset.

The operated object and the grasping end effector receive the same rigid
perturbation, preserving the grasp relation.  Labels are a smooth Cartesian
return to the nominal expert future.  Simulator poses remain supervision only;
deployment inputs are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .build_task_flow_dataset import eef_delta_action14, pose9


def sixd_to_rotation(value: np.ndarray) -> Rotation:
    columns = np.asarray(value, dtype=np.float64).reshape(3, 2)
    first = columns[:, 0]
    first /= np.linalg.norm(first)
    second = columns[:, 1] - first * np.dot(first, columns[:, 1])
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    return Rotation.from_matrix(np.column_stack((first, second, third)))


def pose7_rotation(value: np.ndarray) -> Rotation:
    pose_value = np.asarray(value, dtype=np.float64).reshape(7)
    return Rotation.from_quat(pose_value[[4, 5, 6, 3]])


def rotation_pose7(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.asarray([*position, w, x, y, z], dtype=np.float32)


def centered_transform(
    points: np.ndarray,
    center: np.ndarray,
    rotation: Rotation,
    translation: np.ndarray,
) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.shape[-1] < 3:
        raise ValueError("point features must start with XYZ")
    result = value.copy()
    result[..., :3] = (
        rotation.apply(value[..., :3] - center[None])
        + center[None]
        + translation[None]
    )
    return result.astype(np.float32)


def interpolate_pose(source: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    source_value = np.asarray(source, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    position = source_value[:3] + float(fraction) * (
        target_value[:3] - source_value[:3]
    )
    source_rotation = pose7_rotation(source_value)
    target_rotation = pose7_rotation(target_value)
    delta = target_rotation * source_rotation.inv()
    rotation = Rotation.from_rotvec(delta.as_rotvec() * float(fraction)) * source_rotation
    return rotation_pose7(position, rotation)


def state20_from_pose7(
    left_pose: np.ndarray,
    left_gripper: float,
    right_pose: np.ndarray,
    right_gripper: float,
) -> np.ndarray:
    def pose9_state(value: np.ndarray) -> np.ndarray:
        pose_value = np.asarray(value, dtype=np.float64)
        return np.concatenate(
            (pose_value[:3], pose7_rotation(pose_value).as_matrix()[:, :2].reshape(6))
        )

    return np.concatenate(
        (
            pose9_state(left_pose),
            [float(left_gripper)],
            pose9_state(right_pose),
            [float(right_gripper)],
        )
    ).astype(np.float32)


def pose9_transform(value: np.ndarray) -> tuple[np.ndarray, Rotation]:
    pose_value = np.asarray(value, dtype=np.float64).reshape(9)
    return pose_value[:3], sixd_to_rotation(pose_value[3:])


def augment_recovery_sample(
    sample: dict[str, np.ndarray | float],
    *,
    translation: np.ndarray,
    rotation: Rotation,
    goal_object_pose9: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply one grasp-preserving perturbation and construct recovery labels."""
    result = {key: np.asarray(value).copy() for key, value in sample.items()}
    object_pose9 = np.asarray(sample["current_object_pose9"], dtype=np.float64)
    object_center = object_pose9[:3]
    result["points_a"] = centered_transform(
        sample["points_a"], object_center, rotation, translation
    )
    result["current_anchors"] = centered_transform(
        sample["current_anchors"], object_center, rotation, translation
    )
    object_rotation = rotation * sixd_to_rotation(object_pose9[3:])
    x, y, z, w = object_rotation.as_quat()
    result["current_object_pose9"] = pose9(
        np.asarray([*(object_center + translation), w, x, y, z], dtype=np.float64)
    ).astype(np.float32)

    current = np.asarray(sample["current_eef_pose7"], dtype=np.float64).copy()
    future = np.asarray(sample["future_eef_pose7"], dtype=np.float64)
    action = np.asarray(sample["action"], dtype=np.float64)
    active_arm = 1 if float(sample["active_arm_right"]) >= 0.5 else 0
    active_pose = current[active_arm]
    perturbed_position = rotation.apply(active_pose[:3] - object_center) + object_center + translation
    perturbed_rotation = rotation * pose7_rotation(active_pose)
    current[active_arm] = rotation_pose7(perturbed_position, perturbed_rotation)

    horizon = int(len(future))
    if goal_object_pose9 is None:
        recovery_future = future.copy()
        recovery_targets = future[:, active_arm]
    else:
        goal_position, goal_rotation = pose9_transform(goal_object_pose9)
        augmented_object_position = object_center + translation
        object_delta_rotation = goal_rotation * object_rotation.inv()
        object_delta_translation = goal_position - object_delta_rotation.apply(
            augmented_object_position
        )
        active_goal_position = object_delta_rotation.apply(
            current[active_arm, :3]
        ) + object_delta_translation
        active_goal_rotation = object_delta_rotation * pose7_rotation(
            current[active_arm]
        )
        active_goal = rotation_pose7(active_goal_position, active_goal_rotation)
        recovery_future = np.repeat(current[None], horizon, axis=0)
        recovery_targets = np.repeat(active_goal[None], horizon, axis=0)
    for step in range(horizon):
        recovery_future[step, active_arm] = interpolate_pose(
            current[active_arm], recovery_targets[step], (step + 1) / horizon
        )

    left_gripper = float(np.asarray(sample["state"])[9])
    right_gripper = float(np.asarray(sample["state"])[19])
    recovery_actions = []
    previous = current.copy()
    for step in range(horizon):
        if goal_object_pose9 is None:
            next_left_gripper = float(action[step, 6])
            next_right_gripper = float(action[step, 13])
        else:
            next_left_gripper = left_gripper
            next_right_gripper = right_gripper
        recovery_actions.append(
            eef_delta_action14(
                previous[0],
                left_gripper,
                previous[1],
                right_gripper,
                recovery_future[step, 0],
                next_left_gripper,
                recovery_future[step, 1],
                next_right_gripper,
            )
        )
        previous = recovery_future[step].copy()
        left_gripper = next_left_gripper
        right_gripper = next_right_gripper

    result["current_eef_pose7"] = current.astype(np.float32)
    result["future_eef_pose7"] = recovery_future.astype(np.float32)
    result["action"] = np.stack(recovery_actions).astype(np.float32)
    result["state"] = state20_from_pose7(
        current[0],
        float(np.asarray(sample["state"])[9]),
        current[1],
        float(np.asarray(sample["state"])[19]),
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--copies", type=int, default=1)
    result.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of eligible nominal rows augmented per copy.",
    )
    result.add_argument("--translation-std-cm", type=float, default=1.0)
    result.add_argument("--rotation-std-deg", type=float, default=8.0)
    result.add_argument(
        "--rotation-sampling",
        choices=("gaussian", "uniform_magnitude"),
        default="gaussian",
    )
    result.add_argument("--rotation-min-deg", type=float, default=10.0)
    result.add_argument("--rotation-max-deg", type=float, default=120.0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--relation-only", action="store_true")
    result.add_argument(
        "--recovery-target",
        choices=("nominal_future", "geometry_goal"),
        default="nominal_future",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.copies < 1:
        raise ValueError("copies must be positive")
    if not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("sample-fraction must be in (0,1]")
    if not 0.0 <= args.rotation_min_deg <= args.rotation_max_deg <= 180.0:
        raise ValueError("rotation magnitude bounds must satisfy 0 <= min <= max <= 180")
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    sample_count = int(len(payload["state"]))
    sample_keys = [
        key for key, value in payload.items() if value.ndim > 0 and len(value) == sample_count
    ]
    source_indices = np.arange(sample_count, dtype=np.int64)
    if args.relation_only:
        source_indices = source_indices[
            np.isclose(payload["relation_phase"][source_indices], 1.0)
        ]

    generator = np.random.default_rng(int(args.seed))
    selected_count = max(1, int(round(len(source_indices) * float(args.sample_fraction))))
    appended = {key: [] for key in sample_keys}
    source_sample_index = []
    translation_norms = []
    rotation_degrees = []
    for _ in range(int(args.copies)):
        selected = generator.choice(source_indices, size=selected_count, replace=False)
        for index in selected:
            translation = generator.normal(
                scale=float(args.translation_std_cm) / 100.0, size=3
            )
            if args.rotation_sampling == "gaussian":
                rotvec = generator.normal(
                    scale=np.deg2rad(float(args.rotation_std_deg)), size=3
                )
            else:
                axis = generator.normal(size=3)
                axis /= max(float(np.linalg.norm(axis)), 1e-12)
                magnitude = np.deg2rad(
                    generator.uniform(
                        float(args.rotation_min_deg),
                        float(args.rotation_max_deg),
                    )
                )
                rotvec = axis * magnitude
            rotation = Rotation.from_rotvec(rotvec)
            sample = {key: payload[key][index] for key in sample_keys}
            goal_object_pose9 = None
            if args.recovery_target == "geometry_goal":
                episode = int(payload["episode_index"][index])
                goal_object_pose9 = payload["episode_pose"][episode, -1]
            augmented = augment_recovery_sample(
                sample,
                translation=translation,
                rotation=rotation,
                goal_object_pose9=goal_object_pose9,
            )
            for key in sample_keys:
                appended[key].append(augmented[key])
            source_sample_index.append(int(index))
            translation_norms.append(float(np.linalg.norm(translation)))
            rotation_degrees.append(float(np.rad2deg(np.linalg.norm(rotvec))))

    output_payload = dict(payload)
    for key in sample_keys:
        output_payload[key] = np.concatenate(
            (payload[key], np.asarray(appended[key], dtype=payload[key].dtype)), axis=0
        )
    augmented_count = int(len(source_sample_index))
    output_payload["is_recovery_augmented"] = np.concatenate(
        (
            np.zeros(sample_count, dtype=np.float32),
            np.ones(augmented_count, dtype=np.float32),
        )
    )
    output_payload["source_sample_index"] = np.concatenate(
        (np.arange(sample_count, dtype=np.int64), np.asarray(source_sample_index, dtype=np.int64))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_payload)
    metadata = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "nominal_samples": sample_count,
        "augmented_samples": augmented_count,
        "total_samples": sample_count + augmented_count,
        "copies": int(args.copies),
        "sample_fraction": float(args.sample_fraction),
        "relation_only": bool(args.relation_only),
        "recovery_target": str(args.recovery_target),
        "translation_std_cm": float(args.translation_std_cm),
        "rotation_std_deg": float(args.rotation_std_deg),
        "rotation_sampling": str(args.rotation_sampling),
        "rotation_min_deg": float(args.rotation_min_deg),
        "rotation_max_deg": float(args.rotation_max_deg),
        "actual_translation_norm_cm_mean": float(np.mean(translation_norms) * 100.0),
        "actual_rotation_magnitude_deg_mean": float(np.mean(rotation_degrees)),
        "seed": int(args.seed),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
