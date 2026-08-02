#!/usr/bin/env python3
"""Build a compact oracle object-task-flow benchmark from RoboTwin demos.

Simulator poses are used only to create supervision.  They are never intended
to be policy inputs at deployment.  The dataset keeps one task flow per
episode and references it from frame-level inverse-dynamics samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .build_grasp_dataset import deterministic_farthest_points, valid_xyz
from .target_frame import estimate_geometry_marker_frame


DEFAULT_DATA_DIR = Path(
    "/shared2/sz/robotwin_data/data/place_shoe_geometry_marker/"
    "demo_clean_3d_object_pc_geometry_marker/data"
)


def pose7_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(7)
    w, x, y, z = value[3:] / np.linalg.norm(value[3:])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    transform[:3, 3] = value[:3]
    return transform


def pose7_wxyz_to_rotation(pose: np.ndarray) -> Rotation:
    value = np.asarray(pose, dtype=np.float64).reshape(7)
    w, x, y, z = value[3:] / np.linalg.norm(value[3:])
    return Rotation.from_quat([x, y, z, w])


def rotation_to_sixd(rotation: Rotation) -> np.ndarray:
    return rotation.as_matrix()[:, :2].reshape(6)


def pose9(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(7)
    return np.concatenate((value[:3], rotation_to_sixd(pose7_wxyz_to_rotation(value))))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return value @ transform[:3, :3].T + transform[:3, 3]


def object_goal_error_labels(
    object_pose_a: np.ndarray,
    object_pose_b: np.ndarray,
    goal_t_a_from_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return privileged current-to-goal labels for recovery supervision.

    ``goal_t_a_from_b`` follows the task convention used by RoboTwin; the
    desired world pose of A is ``world_from_b @ inv(goal_t_a_from_b)``.
    These values are dataset labels only and are never deployment inputs.
    """
    world_from_a = pose7_wxyz_to_matrix(object_pose_a)
    world_from_b = pose7_wxyz_to_matrix(object_pose_b)
    goal = np.asarray(goal_t_a_from_b, dtype=np.float64).reshape(4, 4)
    desired_world_from_a = world_from_b @ np.linalg.inv(goal)
    translation = desired_world_from_a[:3, 3] - world_from_a[:3, 3]
    rotation = Rotation.from_matrix(
        desired_world_from_a[:3, :3] @ world_from_a[:3, :3].T
    ).as_rotvec()
    rotation_tolerance = 2.0 * np.arccos(0.995)
    aligned = bool(
        np.all(np.abs(translation) <= np.asarray((0.02, 0.02, 0.025)))
        and np.linalg.norm(rotation) <= rotation_tolerance
    )
    return (
        translation.astype(np.float32),
        rotation.astype(np.float32),
        float(aligned),
    )


def eef_state20(
    left_pose: np.ndarray,
    left_gripper: float,
    right_pose: np.ndarray,
    right_gripper: float,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(left_pose)[:3],
            rotation_to_sixd(pose7_wxyz_to_rotation(left_pose)),
            [left_gripper],
            np.asarray(right_pose)[:3],
            rotation_to_sixd(pose7_wxyz_to_rotation(right_pose)),
            [right_gripper],
        )
    ).astype(np.float32)


def eef_delta_action14(
    left_pose: np.ndarray,
    left_gripper: float,
    right_pose: np.ndarray,
    right_gripper: float,
    next_left_pose: np.ndarray,
    next_left_gripper: float,
    next_right_pose: np.ndarray,
    next_right_gripper: float,
) -> np.ndarray:
    left_delta_rotation = (
        pose7_wxyz_to_rotation(next_left_pose) * pose7_wxyz_to_rotation(left_pose).inv()
    )
    right_delta_rotation = (
        pose7_wxyz_to_rotation(next_right_pose) * pose7_wxyz_to_rotation(right_pose).inv()
    )
    return np.concatenate(
        (
            np.asarray(next_left_pose)[:3] - np.asarray(left_pose)[:3],
            left_delta_rotation.as_rotvec(),
            [next_left_gripper],
            np.asarray(next_right_pose)[:3] - np.asarray(right_pose)[:3],
            right_delta_rotation.as_rotvec(),
            [next_right_gripper],
        )
    ).astype(np.float32)


def deterministic_sample(
    points: np.ndarray,
    count: int,
    seed: int,
    point_channels: int = 3,
    priority_color: bool = False,
) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < int(point_channels):
        raise ValueError(
            f"point cloud must provide {point_channels} channels, got {raw.shape}"
        )
    keep = np.all(np.isfinite(raw[:, : int(point_channels)]), axis=1)
    value = raw[keep, : int(point_channels)]
    if len(value) == 0:
        raise ValueError("cannot sample an empty point cloud")
    generator = np.random.default_rng(int(seed))
    priority_indices = np.empty((0,), dtype=np.int64)
    if priority_color:
        if value.shape[1] < 6:
            raise ValueError("priority_color requires XYZRGB points")
        color = value[:, 3:6]
        priority_indices = np.flatnonzero(
            color[:, 0] + color[:, 1] - 2.0 * color[:, 2] > 0.65
        )
        if len(priority_indices) > int(count):
            priority_indices = generator.choice(
                priority_indices, size=int(count), replace=False
            )
    remaining = int(count) - len(priority_indices)
    if remaining > 0:
        available = np.setdiff1d(
            np.arange(len(value), dtype=np.int64),
            priority_indices,
            assume_unique=False,
        )
        background = generator.choice(
            available,
            size=remaining,
            replace=len(available) < remaining,
        )
        indices = np.concatenate((priority_indices, background))
        generator.shuffle(indices)
    else:
        indices = priority_indices
    return value[indices].astype(np.float32)


def resample_indices(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    return np.rint(np.linspace(0, length - 1, int(count))).astype(np.int64)


def identify_active_arm(left_gripper: np.ndarray, right_gripper: np.ndarray) -> str:
    closing = []
    if float(np.min(left_gripper)) < 0.5:
        closing.append("left")
    if float(np.min(right_gripper)) < 0.5:
        closing.append("right")
    if len(closing) != 1:
        raise ValueError(f"expected one active arm, got {closing}")
    return closing[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--observation-points", type=int, default=64)
    parser.add_argument("--point-channels", type=int, choices=(3, 6), default=3)
    parser.add_argument("--target-color-priority", action="store_true")
    parser.add_argument("--cache-target-observation", action="store_true")
    parser.add_argument("--target-frame-token", action="store_true")
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")

    frame_arrays: dict[str, list[np.ndarray | float | int]] = {
        "points_a": [],
        "points_b": [],
        "state": [],
        "action": [],
        "current_anchors": [],
        "current_object_pose9": [],
        "current_eef_pose7": [],
        "future_eef_pose7": [],
        "active_arm_right": [],
        "episode_index": [],
        "episode_id": [],
        "shoe_id": [],
        "frame_index": [],
        "relation_phase": [],
        "goal_translation_error_xyz_m": [],
        "goal_rotation_error_rotvec": [],
        "goal_aligned": [],
        "steps_to_release": [],
        "stop_phase": [],
    }
    if args.target_frame_token:
        frame_arrays["target_frame9"] = []
    episode_flow = []
    episode_pose = []
    episode_shoe_id = []
    diagnostics = []

    for episode in range(int(args.episodes)):
        path = args.data_dir / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as root:
            object_pose = np.asarray(root["task_state/object_pose_A"], dtype=np.float64)
            target_pose = np.asarray(root["task_state/object_pose_B"], dtype=np.float64)
            goal_t_a_from_b = np.asarray(
                root["task_state/goal_T_A_from_B_oracle"], dtype=np.float64
            ).reshape(-1, 4, 4)
            frame_count = int(len(object_pose))
            left_pose = np.asarray(root["endpose/left_endpose"], dtype=np.float64)
            right_pose = np.asarray(root["endpose/right_endpose"], dtype=np.float64)
            left_gripper = np.asarray(root["endpose/left_gripper"], dtype=np.float64)
            right_gripper = np.asarray(root["endpose/right_gripper"], dtype=np.float64)
            phase = np.asarray(root["task_state/relation_phase"], dtype=np.float32).reshape(-1)
            shoe_id = int(np.asarray(root["task_state/shoe_id"][0]).reshape(-1)[0])
            active_arm = identify_active_arm(left_gripper, right_gripper)
            active_gripper = left_gripper if active_arm == "left" else right_gripper
            target_frame9 = None
            if args.target_frame_token:
                target_frame, _ = estimate_geometry_marker_frame(
                    root["object_pointcloud/{B}"][0]
                )
                target_frame9 = np.concatenate(
                    (
                        target_frame[:3, 3],
                        target_frame[:3, 0],
                        target_frame[:3, 1],
                    )
                ).astype(np.float32)

            initial_visible = valid_xyz(root["object_pointcloud/{A}"][0])
            initial_anchors_world = deterministic_farthest_points(
                initial_visible, int(args.anchors)
            ).astype(np.float64)
            initial_transform = pose7_wxyz_to_matrix(object_pose[0])
            anchors_local = transform_points(
                np.linalg.inv(initial_transform), initial_anchors_world
            )
            task_indices = resample_indices(frame_count, int(args.flow_steps))
            task_flow = np.stack(
                [
                    transform_points(pose7_wxyz_to_matrix(object_pose[index]), anchors_local)
                    for index in task_indices
                ],
                axis=1,
            ).astype(np.float32)
            task_pose = np.stack([pose9(object_pose[index]) for index in task_indices]).astype(
                np.float32
            )
            episode_flow.append(task_flow)
            episode_pose.append(task_pose)
            episode_shoe_id.append(shoe_id)

            sample_count_before = len(frame_arrays["episode_id"])
            for frame in range(0, frame_count - 1, int(args.frame_stride)):
                sequence_frames = [
                    min(frame + int(args.frame_stride) * step, frame_count - 1)
                    for step in range(int(args.action_horizon) + 1)
                ]
                actions = []
                future_poses = []
                for step in range(int(args.action_horizon)):
                    source = sequence_frames[step]
                    target = sequence_frames[step + 1]
                    actions.append(
                        eef_delta_action14(
                            left_pose[source],
                            left_gripper[source],
                            right_pose[source],
                            right_gripper[source],
                            left_pose[target],
                            left_gripper[target],
                            right_pose[target],
                            right_gripper[target],
                        )
                    )
                    future_poses.append(np.stack((left_pose[target], right_pose[target])))

                current_transform = pose7_wxyz_to_matrix(object_pose[frame])
                frame_arrays["points_a"].append(
                    deterministic_sample(
                        root["object_pointcloud/{A}"][frame],
                        int(args.observation_points),
                        seed=episode * 100003 + frame * 17 + 1,
                        point_channels=int(args.point_channels),
                    )
                )
                frame_arrays["points_b"].append(
                    deterministic_sample(
                        root["object_pointcloud/{B}"][
                            0 if args.cache_target_observation else frame
                        ],
                        int(args.observation_points),
                        seed=(
                            episode * 100003
                            + (0 if args.cache_target_observation else frame * 17)
                            + 2
                        ),
                        point_channels=int(args.point_channels),
                        priority_color=bool(args.target_color_priority),
                    )
                )
                frame_arrays["state"].append(
                    eef_state20(
                        left_pose[frame],
                        left_gripper[frame],
                        right_pose[frame],
                        right_gripper[frame],
                    )
                )
                frame_arrays["action"].append(np.stack(actions))
                frame_arrays["current_anchors"].append(
                    transform_points(current_transform, anchors_local).astype(np.float32)
                )
                frame_arrays["current_object_pose9"].append(pose9(object_pose[frame]))
                frame_arrays["current_eef_pose7"].append(
                    np.stack((left_pose[frame], right_pose[frame])).astype(np.float32)
                )
                frame_arrays["future_eef_pose7"].append(
                    np.stack(future_poses).astype(np.float32)
                )
                frame_arrays["active_arm_right"].append(float(active_arm == "right"))
                frame_arrays["episode_index"].append(episode)
                frame_arrays["episode_id"].append(episode)
                frame_arrays["shoe_id"].append(shoe_id)
                frame_arrays["frame_index"].append(frame)
                frame_arrays["relation_phase"].append(float(phase[frame]))
                (
                    goal_translation,
                    goal_rotation,
                    goal_aligned,
                ) = object_goal_error_labels(
                    object_pose[frame],
                    target_pose[frame],
                    goal_t_a_from_b[frame],
                )
                frame_arrays["goal_translation_error_xyz_m"].append(
                    goal_translation
                )
                frame_arrays["goal_rotation_error_rotvec"].append(goal_rotation)
                frame_arrays["goal_aligned"].append(goal_aligned)
                future_open = np.flatnonzero(active_gripper[frame:] >= 0.5)
                frame_arrays["steps_to_release"].append(
                    float(future_open[0]) if len(future_open) else float(frame_count - frame)
                )
                active_offset = 7 if active_arm == "right" else 0
                releases_now = float(actions[0][active_offset + 6]) >= 0.5
                frame_arrays["stop_phase"].append(
                    float(2 if releases_now else 1 if goal_aligned >= 0.5 else 0)
                )
                if target_frame9 is not None:
                    frame_arrays["target_frame9"].append(target_frame9)

            diagnostics.append(
                {
                    "episode": episode,
                    "shoe_id": shoe_id,
                    "frames": frame_count,
                    "samples": len(frame_arrays["episode_id"]) - sample_count_before,
                    "active_arm": active_arm,
                    "task_indices": task_indices.tolist(),
                }
            )
            print(json.dumps(diagnostics[-1]), flush=True)

    integer_keys = {"episode_index", "episode_id", "shoe_id", "frame_index"}
    payload = {
        key: np.asarray(values, dtype=np.int64 if key in integer_keys else np.float32)
        for key, values in frame_arrays.items()
    }
    payload.update(
        {
            "episode_flow": np.asarray(episode_flow, dtype=np.float32),
            "episode_pose": np.asarray(episode_pose, dtype=np.float32),
            "episode_shoe_id": np.asarray(episode_shoe_id, dtype=np.int64),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = {
        "schema_version": 1,
        "data_dir": str(args.data_dir.resolve()),
        "output": str(output),
        "episodes": int(args.episodes),
        "samples": int(len(payload["episode_id"])),
        "observation_points_per_object": int(args.observation_points),
        "point_channels": int(args.point_channels),
        "target_color_priority": bool(args.target_color_priority),
        "cache_target_observation": bool(args.cache_target_observation),
        "target_frame_token": bool(args.target_frame_token),
        "anchors": int(args.anchors),
        "flow_steps": int(args.flow_steps),
        "action_horizon": int(args.action_horizon),
        "frame_stride": int(args.frame_stride),
        "privileged_training_labels": [
            "simulator object pose used to construct oracle task flow",
            "simulator end-effector pose used to construct action labels",
            "simulator A/B poses and goal transform used for stop/release labels",
        ],
        "deployment_inputs_for_future_model": [
            "camera point clouds A and B",
            "robot proprioception",
            "predicted object task flow",
        ],
        "diagnostics": diagnostics,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: metadata[key] for key in ("output", "samples")}, indent=2))


if __name__ == "__main__":
    main()
