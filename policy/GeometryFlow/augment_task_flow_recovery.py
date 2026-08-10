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


def functional_frame9(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    matrix = rotation.as_matrix()
    return np.concatenate(
        (np.asarray(position, dtype=np.float64), matrix[:, 0], matrix[:, 1])
    ).astype(np.float32)


def functional_frame9_transform(
    value: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    frame_value = np.asarray(value, dtype=np.float64).reshape(9)
    first = frame_value[3:6]
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = frame_value[6:9] - first * float(
        np.dot(first, frame_value[6:9])
    )
    second /= max(float(np.linalg.norm(second)), 1e-12)
    matrix = np.column_stack((first, second, np.cross(first, second)))
    return frame_value[:3], Rotation.from_matrix(matrix)


def object_pose9(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(position, dtype=np.float64),
            rotation.as_matrix()[:, :2].reshape(6),
        )
    ).astype(np.float32)


def sample_failure_matched_rotation(
    generator: np.random.Generator,
    mode: dict,
    *,
    axis_jitter_deg: float,
) -> tuple[Rotation, np.ndarray, float]:
    """Sample an object perturbation that induces a requested remaining axis.

    Near an aligned donor, applying ``R`` to the current object produces the
    remaining relation ``goal * current^-1 ~= R^-1``.  The perturbation axis is
    therefore the negative of the desired remaining-rotation axis.
    """

    remaining_axis = np.asarray(mode["signed_world_axis"], dtype=np.float64)
    remaining_axis /= max(float(np.linalg.norm(remaining_axis)), 1.0e-12)
    jitter_rad = np.deg2rad(max(float(axis_jitter_deg), 0.0))
    if jitter_rad > 0.0:
        perpendicular = generator.normal(size=3)
        perpendicular -= remaining_axis * float(
            np.dot(perpendicular, remaining_axis)
        )
        perpendicular_norm = float(np.linalg.norm(perpendicular))
        if perpendicular_norm > 1.0e-12:
            perpendicular /= perpendicular_norm
            angle = float(np.clip(generator.normal(scale=jitter_rad), -2.5 * jitter_rad, 2.5 * jitter_rad))
            remaining_axis = Rotation.from_rotvec(perpendicular * angle).apply(
                remaining_axis
            )
            remaining_axis /= max(float(np.linalg.norm(remaining_axis)), 1.0e-12)
    minimum_deg = float(mode["rotation_magnitude_deg_p25"])
    maximum_deg = float(mode["rotation_magnitude_deg_p75"])
    magnitude_deg = float(generator.uniform(minimum_deg, maximum_deg))
    perturbation = Rotation.from_rotvec(
        -remaining_axis * np.deg2rad(magnitude_deg)
    )
    return perturbation, remaining_axis, magnitude_deg


def sample_uniform_magnitude_vector(
    generator: np.random.Generator,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Sample an isotropic vector without concentrating mass near zero."""

    axis = generator.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    return axis * float(generator.uniform(float(minimum), float(maximum)))


def translation_for_remaining_functional_relation(
    sample: dict[str, np.ndarray | float],
    rotation: Rotation,
    desired_remaining_translation: np.ndarray,
) -> np.ndarray:
    """Solve the shared perturbation that realizes a requested task relation.

    The camera token stores ``goal_position - current_position``. Sampling an
    object perturbation directly does not control that resulting relation when
    the donor is already offset from the goal. This inverse keeps the sampled
    recovery support identical in data generation and deployment.
    """

    if "target_frame9" not in sample or "goal_frame9" not in sample:
        raise KeyError(
            "remaining-relation translation sampling requires target_frame9 "
            "and goal_frame9"
        )
    goal_position, _ = functional_frame9_transform(sample["goal_frame9"])
    current_to_goal, _ = functional_frame9_transform(sample["target_frame9"])
    current_position = goal_position - current_to_goal
    object_center = np.asarray(sample["current_object_pose9"], dtype=np.float64)[:3]
    rotated_position = rotation.apply(current_position - object_center) + object_center
    requested_current_position = goal_position - np.asarray(
        desired_remaining_translation, dtype=np.float64
    )
    return requested_current_position - rotated_position


def intended_goal_object_pose9(sample: dict[str, np.ndarray | float]) -> np.ndarray:
    """Recover the task-defined object goal without using an expert endpoint."""

    required = ("current_object_pose9", "goal_translation_error_xyz_m", "goal_rotation_error_rotvec")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"intended goal labels are missing {missing}")
    current_position, current_rotation = pose9_transform(
        np.asarray(sample["current_object_pose9"])
    )
    goal_position = current_position + np.asarray(
        sample["goal_translation_error_xyz_m"], dtype=np.float64
    )
    goal_rotation = Rotation.from_rotvec(
        np.asarray(sample["goal_rotation_error_rotvec"], dtype=np.float64)
    ) * current_rotation
    return object_pose9(goal_position, goal_rotation)


def update_camera_functional_frame_after_perturbation(
    result: dict[str, np.ndarray],
    sample: dict[str, np.ndarray | float],
    *,
    object_center: np.ndarray,
    rotation: Rotation,
    translation: np.ndarray,
) -> None:
    """Keep a camera-derived current-to-goal token consistent with A motion."""

    if "target_frame9" not in sample or "goal_frame9" not in sample:
        return
    goal_position, goal_rotation = functional_frame9_transform(
        sample["goal_frame9"]
    )
    relative_position, relative_rotation = functional_frame9_transform(
        sample["target_frame9"]
    )
    current_position = goal_position - relative_position
    current_rotation = relative_rotation.inv() * goal_rotation
    perturbed_position = (
        rotation.apply(current_position - object_center)
        + object_center
        + translation
    )
    perturbed_rotation = rotation * current_rotation
    new_relative_rotation = goal_rotation * perturbed_rotation.inv()
    result["target_frame9"] = functional_frame9(
        goal_position - perturbed_position, new_relative_rotation
    )
    if "source_frame6_columns" in result:
        result["source_frame6_columns"] = functional_frame9(
            np.zeros(3), perturbed_rotation
        )[3:]


def augment_recovery_sample(
    sample: dict[str, np.ndarray | float],
    *,
    translation: np.ndarray,
    rotation: Rotation,
    goal_object_pose9: np.ndarray | None = None,
    release_at_goal: bool = False,
    release_immediately: bool = False,
) -> dict[str, np.ndarray]:
    """Apply one grasp-preserving perturbation and construct recovery labels."""
    if release_at_goal and release_immediately:
        raise ValueError("release_at_goal and release_immediately are mutually exclusive")
    if (release_at_goal or release_immediately) and goal_object_pose9 is None:
        raise ValueError("release supervision requires a geometry goal")
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
    update_camera_functional_frame_after_perturbation(
        result,
        sample,
        object_center=object_center,
        rotation=rotation,
        translation=translation,
    )
    if (
        "goal_translation_error_xyz_m" in sample
        and "goal_rotation_error_rotvec" in sample
    ):
        goal_position, goal_rotation = pose9_transform(
            intended_goal_object_pose9(sample)
        )
        result["goal_translation_error_xyz_m"] = (
            goal_position - result["current_object_pose9"][:3]
        ).astype(np.float32)
        result["goal_rotation_error_rotvec"] = (
            goal_rotation * object_rotation.inv()
        ).as_rotvec().astype(np.float32)
    if "goal_aligned" in result:
        result["goal_aligned"] = np.asarray(0.0, dtype=result["goal_aligned"].dtype)

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
            if release_immediately or (release_at_goal and step == horizon - 1):
                if active_arm == 0:
                    next_left_gripper = 1.0
                else:
                    next_right_gripper = 1.0
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
    result.add_argument(
        "--translation-sampling",
        choices=("gaussian", "uniform_magnitude"),
        default="gaussian",
    )
    result.add_argument(
        "--translation-objective",
        choices=("object_perturbation", "remaining_relation"),
        default="object_perturbation",
        help=(
            "Interpret the sampled vector as either a physical object "
            "perturbation or the desired goal-minus-current task relation."
        ),
    )
    result.add_argument("--translation-min-cm", type=float, default=3.0)
    result.add_argument("--translation-max-cm", type=float, default=8.0)
    result.add_argument("--rotation-std-deg", type=float, default=8.0)
    result.add_argument(
        "--rotation-sampling",
        choices=("gaussian", "uniform_magnitude"),
        default="gaussian",
    )
    result.add_argument("--rotation-min-deg", type=float, default=10.0)
    result.add_argument("--rotation-max-deg", type=float, default=120.0)
    result.add_argument(
        "--rotation-mode-json",
        type=Path,
        default=None,
        help=(
            "Failure-trace summary containing an aggregate signed remaining-"
            "rotation mode. Raw held-out observations are never consumed."
        ),
    )
    result.add_argument(
        "--rotation-mode-key",
        default="stalled_recovery_direction_mode",
    )
    result.add_argument("--rotation-axis-jitter-deg", type=float, default=15.0)
    result.add_argument(
        "--gate-positive",
        action="store_true",
        help=(
            "Mark every newly augmented row as a positive example for the "
            "learned recovery gate. Failure-mode rotations are always marked; "
            "use this flag for deliberately broad axis-coverage augmentation."
        ),
    )
    result.add_argument(
        "--translation-recovery-positive",
        action="store_true",
        help=(
            "Mark appended rows as supervision for the low-rotation translation "
            "decoder independently of the broader recovery-gate label."
        ),
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--relation-only", action="store_true")
    result.add_argument(
        "--allowed-shoes",
        default="",
        help="Optional comma-separated object IDs eligible for augmentation.",
    )
    result.add_argument(
        "--recovery-target",
        choices=("nominal_future", "geometry_goal"),
        default="nominal_future",
    )
    result.add_argument(
        "--release-at-goal",
        action="store_true",
        help=(
            "For geometry-goal recovery, label only the final active-gripper "
            "action as open. This teaches release after completing the learned "
            "recovery chunk without adding a deployment-time threshold."
        ),
    )
    result.add_argument(
        "--release-immediately",
        action="store_true",
        help=(
            "For geometry-goal samples already restricted to the terminal "
            "success neighborhood, label the active gripper open at every "
            "step. Use small perturbations and strict remaining-error filters."
        ),
    )
    result.add_argument(
        "--remaining-translation-max-cm",
        type=float,
        default=None,
        help="Optionally restrict donors to near-goal remaining translation.",
    )
    result.add_argument(
        "--remaining-rotation-max-deg",
        type=float,
        default=None,
        help="Optionally restrict donors to near-goal remaining rotation.",
    )
    result.add_argument(
        "--active-gripper-max",
        type=float,
        default=None,
        help=(
            "Optionally require the currently active gripper state to be at "
            "or below this value. Terminal release augmentation should use a "
            "small value (for example 0.1) so supervision reaches the learned "
            "closed-gripper retention branch rather than the already-open head."
        ),
    )
    result.add_argument(
        "--active-arm",
        choices=("any", "left", "right"),
        default="any",
        help="Optionally balance recovery augmentation toward one grasping arm.",
    )
    result.add_argument(
        "--balance-active-arms",
        action="store_true",
        help="Select exactly equal numbers of closed left- and right-arm donors.",
    )
    result.add_argument(
        "--nominal-donors-only",
        action="store_true",
        help="Exclude pre-existing synthetic recovery rows from donor selection.",
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
    if not 0.0 <= args.translation_min_cm <= args.translation_max_cm:
        raise ValueError(
            "translation magnitude bounds must satisfy 0 <= min <= max"
        )
    if args.release_at_goal and args.release_immediately:
        raise ValueError("release supervision modes are mutually exclusive")
    if (
        (args.release_at_goal or args.release_immediately)
        and args.recovery_target != "geometry_goal"
    ):
        raise ValueError("release supervision requires --recovery-target geometry_goal")
    if (
        args.remaining_translation_max_cm is not None
        and float(args.remaining_translation_max_cm) <= 0.0
    ):
        raise ValueError("remaining-translation-max-cm must be positive")
    if (
        args.remaining_rotation_max_deg is not None
        and float(args.remaining_rotation_max_deg) <= 0.0
    ):
        raise ValueError("remaining-rotation-max-deg must be positive")
    if args.active_gripper_max is not None and not (
        0.0 <= float(args.active_gripper_max) <= 1.0
    ):
        raise ValueError("active-gripper-max must be in [0,1]")
    if args.balance_active_arms and args.active_arm != "any":
        raise ValueError("balance-active-arms requires --active-arm any")
    if float(args.rotation_axis_jitter_deg) < 0.0:
        raise ValueError("rotation-axis-jitter-deg must be non-negative")
    rotation_mode = None
    if args.rotation_mode_json is not None:
        mode_payload = json.loads(args.rotation_mode_json.read_text(encoding="utf-8"))
        rotation_mode = mode_payload.get(str(args.rotation_mode_key))
        if not isinstance(rotation_mode, dict):
            raise KeyError(
                f"rotation mode {args.rotation_mode_key!r} is absent from "
                f"{args.rotation_mode_json}"
            )
        required_mode_keys = {
            "signed_world_axis",
            "rotation_magnitude_deg_p25",
            "rotation_magnitude_deg_p75",
        }
        missing_mode_keys = required_mode_keys - set(rotation_mode)
        if missing_mode_keys:
            raise KeyError(f"rotation mode is missing {sorted(missing_mode_keys)}")
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    sample_count = int(len(payload["state"]))
    sample_keys = [
        key for key, value in payload.items() if value.ndim > 0 and len(value) == sample_count
    ]
    source_indices = np.arange(sample_count, dtype=np.int64)
    active_right = (
        None
        if "active_arm_right" not in payload
        else np.asarray(payload["active_arm_right"]).reshape(-1) >= 0.5
    )
    allowed_shoes = {
        int(item) for item in str(args.allowed_shoes).split(",") if item.strip()
    }
    if allowed_shoes:
        source_indices = source_indices[
            np.isin(payload["shoe_id"][source_indices], sorted(allowed_shoes))
        ]
    if args.relation_only:
        source_indices = source_indices[
            np.isclose(payload["relation_phase"][source_indices], 1.0)
        ]
    if args.nominal_donors_only and "is_recovery_augmented" in payload:
        source_indices = source_indices[
            np.asarray(payload["is_recovery_augmented"])[source_indices] < 0.5
        ]
    if args.active_arm != "any":
        if active_right is None:
            raise KeyError("active-arm filtering requires active_arm_right")
        desired_right = args.active_arm == "right"
        source_indices = source_indices[
            active_right[source_indices] == desired_right
        ]
    if args.active_gripper_max is not None:
        if active_right is None or "state" not in payload:
            raise KeyError(
                "active gripper filtering requires active_arm_right and state"
            )
        active_gripper = np.where(
            active_right,
            np.asarray(payload["state"])[:, 19],
            np.asarray(payload["state"])[:, 9],
        )
        source_indices = source_indices[
            active_gripper[source_indices] <= float(args.active_gripper_max)
        ]
    if args.remaining_translation_max_cm is not None:
        if "target_frame9" not in payload:
            raise KeyError("remaining translation filtering requires target_frame9")
        remaining_translation_cm = 100.0 * np.linalg.norm(
            np.asarray(payload["target_frame9"][:, :3], dtype=np.float64), axis=-1
        )
        source_indices = source_indices[
            remaining_translation_cm[source_indices]
            <= float(args.remaining_translation_max_cm)
        ]
    if args.remaining_rotation_max_deg is not None:
        if "target_frame9" not in payload:
            raise KeyError("remaining rotation filtering requires target_frame9")
        remaining_rotation_deg = np.asarray(
            [
                np.rad2deg(functional_frame9_transform(frame)[1].magnitude())
                for frame in payload["target_frame9"]
            ],
            dtype=np.float64,
        )
        source_indices = source_indices[
            remaining_rotation_deg[source_indices]
            <= float(args.remaining_rotation_max_deg)
        ]
    if len(source_indices) == 0:
        raise ValueError("recovery donor filters selected no samples")

    generator = np.random.default_rng(int(args.seed))
    selected_count = max(1, int(round(len(source_indices) * float(args.sample_fraction))))
    if args.balance_active_arms:
        if active_right is None:
            raise KeyError("balance-active-arms requires active_arm_right")
        left_indices = source_indices[~active_right[source_indices]]
        right_indices = source_indices[active_right[source_indices]]
        per_arm_count = min(
            max(1, int(round(len(left_indices) * float(args.sample_fraction)))),
            max(1, int(round(len(right_indices) * float(args.sample_fraction)))),
        )
        if not len(left_indices) or not len(right_indices):
            raise ValueError("balanced donor selection requires both active arms")
    else:
        left_indices = right_indices = np.empty(0, dtype=np.int64)
        per_arm_count = 0
    appended = {key: [] for key in sample_keys}
    source_sample_index = []
    translation_norms = []
    remaining_translation_norms = []
    rotation_degrees = []
    desired_remaining_axes = []
    for _ in range(int(args.copies)):
        if args.balance_active_arms:
            selected = np.concatenate(
                (
                    generator.choice(left_indices, size=per_arm_count, replace=False),
                    generator.choice(right_indices, size=per_arm_count, replace=False),
                )
            )
            generator.shuffle(selected)
        else:
            selected = generator.choice(source_indices, size=selected_count, replace=False)
        for index in selected:
            sample = {key: payload[key][index] for key in sample_keys}
            if args.translation_sampling == "gaussian":
                sampled_translation = generator.normal(
                    scale=float(args.translation_std_cm) / 100.0, size=3
                )
            else:
                sampled_translation = sample_uniform_magnitude_vector(
                    generator,
                    float(args.translation_min_cm) / 100.0,
                    float(args.translation_max_cm) / 100.0,
                )
            if rotation_mode is not None:
                rotation, desired_axis, sampled_magnitude_deg = (
                    sample_failure_matched_rotation(
                        generator,
                        rotation_mode,
                        axis_jitter_deg=float(args.rotation_axis_jitter_deg),
                    )
                )
                rotvec = rotation.as_rotvec()
                desired_remaining_axes.append(desired_axis)
            elif args.rotation_sampling == "gaussian":
                rotvec = generator.normal(
                    scale=np.deg2rad(float(args.rotation_std_deg)), size=3
                )
                rotation = Rotation.from_rotvec(rotvec)
                sampled_magnitude_deg = float(np.rad2deg(np.linalg.norm(rotvec)))
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
                sampled_magnitude_deg = float(np.rad2deg(magnitude))
            if args.translation_objective == "remaining_relation":
                translation = translation_for_remaining_functional_relation(
                    sample, rotation, sampled_translation
                )
                remaining_translation_norms.append(
                    float(np.linalg.norm(sampled_translation))
                )
            else:
                translation = sampled_translation
            goal_object_pose9 = None
            if args.recovery_target == "geometry_goal":
                if {
                    "goal_translation_error_xyz_m",
                    "goal_rotation_error_rotvec",
                }.issubset(sample):
                    goal_object_pose9 = intended_goal_object_pose9(sample)
                else:
                    episode = int(payload["episode_index"][index])
                    goal_object_pose9 = payload["episode_pose"][episode, -1]
            augmented = augment_recovery_sample(
                sample,
                translation=translation,
                rotation=rotation,
                goal_object_pose9=goal_object_pose9,
                release_at_goal=bool(args.release_at_goal),
                release_immediately=bool(args.release_immediately),
            )
            for key in sample_keys:
                appended[key].append(augmented[key])
            source_sample_index.append(int(index))
            translation_norms.append(float(np.linalg.norm(translation)))
            rotation_degrees.append(sampled_magnitude_deg)

    output_payload = dict(payload)
    for key in sample_keys:
        output_payload[key] = np.concatenate(
            (payload[key], np.asarray(appended[key], dtype=payload[key].dtype)), axis=0
        )
    augmented_count = int(len(source_sample_index))
    existing_recovery = np.asarray(
        payload.get("is_recovery_augmented", np.zeros(sample_count)),
        dtype=np.float32,
    ).reshape(-1)
    output_payload["is_recovery_augmented"] = np.concatenate(
        (
            existing_recovery,
            np.ones(augmented_count, dtype=np.float32),
        )
    )
    existing_failure_matched = np.asarray(
        payload.get("is_failure_matched_recovery", np.zeros(sample_count)),
        dtype=np.float32,
    ).reshape(-1)
    output_payload["is_failure_matched_recovery"] = np.concatenate(
        (
            existing_failure_matched,
            np.full(
                augmented_count,
                1.0 if rotation_mode is not None else 0.0,
                dtype=np.float32,
            ),
        )
    )
    existing_gate_positive = np.asarray(
        payload.get(
            "is_recovery_gate_positive",
            existing_failure_matched,
        ),
        dtype=np.float32,
    ).reshape(-1)
    output_payload["is_recovery_gate_positive"] = np.concatenate(
        (
            existing_gate_positive,
            np.full(
                augmented_count,
                1.0 if (rotation_mode is not None or args.gate_positive) else 0.0,
                dtype=np.float32,
            ),
        )
    )
    existing_translation_positive = np.asarray(
        payload.get("is_translation_recovery_positive", np.zeros(sample_count)),
        dtype=np.float32,
    ).reshape(-1)
    output_payload["is_translation_recovery_positive"] = np.concatenate(
        (
            existing_translation_positive,
            np.full(
                augmented_count,
                1.0 if args.translation_recovery_positive else 0.0,
                dtype=np.float32,
            ),
        )
    )
    output_payload["source_sample_index"] = np.concatenate(
        (np.arange(sample_count, dtype=np.int64), np.asarray(source_sample_index, dtype=np.int64))
    )
    # Each synthetic recovery row is an independent off-trajectory state, not
    # another frame in its donor demonstration.  Preserve the semantic donor
    # episode fields for auditing, while giving the causal dataset builder a
    # separate history identity.  Otherwise rows with duplicated donor
    # frame_index values become accidental temporal neighbors.
    nominal_history_episode = np.asarray(
        payload.get(
            "history_episode_index",
            payload.get(
                "episode_index",
                payload.get("episode_id", np.arange(sample_count, dtype=np.int64)),
            ),
        ),
        dtype=np.int64,
    ).reshape(-1)
    nominal_history_frame = np.asarray(
        payload.get(
            "history_frame_index",
            payload.get("frame_index", np.arange(sample_count, dtype=np.int64)),
        ),
        dtype=np.int64,
    ).reshape(-1)
    synthetic_history_episode = -1 - np.arange(augmented_count, dtype=np.int64)
    output_payload["history_episode_index"] = np.concatenate(
        (nominal_history_episode, synthetic_history_episode)
    )
    output_payload["history_frame_index"] = np.concatenate(
        (nominal_history_frame, np.zeros(augmented_count, dtype=np.int64))
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
        "eligible_samples": int(len(source_indices)),
        "remaining_translation_max_cm": (
            None
            if args.remaining_translation_max_cm is None
            else float(args.remaining_translation_max_cm)
        ),
        "remaining_rotation_max_deg": (
            None
            if args.remaining_rotation_max_deg is None
            else float(args.remaining_rotation_max_deg)
        ),
        "active_gripper_max": (
            None
            if args.active_gripper_max is None
            else float(args.active_gripper_max)
        ),
        "active_arm": str(args.active_arm),
        "balance_active_arms": bool(args.balance_active_arms),
        "balanced_donors_per_arm_per_copy": int(per_arm_count),
        "nominal_donors_only": bool(args.nominal_donors_only),
        "allowed_shoes": sorted(allowed_shoes),
        "recovery_target": str(args.recovery_target),
        "release_at_goal": bool(args.release_at_goal),
        "release_immediately": bool(args.release_immediately),
        "translation_std_cm": float(args.translation_std_cm),
        "translation_sampling": str(args.translation_sampling),
        "translation_objective": str(args.translation_objective),
        "translation_min_cm": float(args.translation_min_cm),
        "translation_max_cm": float(args.translation_max_cm),
        "rotation_std_deg": float(args.rotation_std_deg),
        "rotation_sampling": (
            "failure_matched_mode"
            if rotation_mode is not None
            else str(args.rotation_sampling)
        ),
        "rotation_mode_json": (
            None
            if args.rotation_mode_json is None
            else str(args.rotation_mode_json.resolve())
        ),
        "rotation_mode_key": (
            None if rotation_mode is None else str(args.rotation_mode_key)
        ),
        "rotation_axis_jitter_deg": float(args.rotation_axis_jitter_deg),
        "gate_positive": bool(args.gate_positive or rotation_mode is not None),
        "translation_recovery_positive": bool(
            args.translation_recovery_positive
        ),
        "desired_remaining_axis_mean": (
            None
            if not desired_remaining_axes
            else (
                np.mean(np.asarray(desired_remaining_axes), axis=0)
                / max(
                    float(
                        np.linalg.norm(
                            np.mean(np.asarray(desired_remaining_axes), axis=0)
                        )
                    ),
                    1.0e-12,
                )
            ).astype(float).tolist()
        ),
        "rotation_min_deg": float(args.rotation_min_deg),
        "rotation_max_deg": float(args.rotation_max_deg),
        "actual_translation_norm_cm_mean": float(np.mean(translation_norms) * 100.0),
        "actual_remaining_translation_norm_cm_mean": (
            None
            if not remaining_translation_norms
            else float(np.mean(remaining_translation_norms) * 100.0)
        ),
        "actual_rotation_magnitude_deg_mean": float(np.mean(rotation_degrees)),
        "seed": int(args.seed),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
