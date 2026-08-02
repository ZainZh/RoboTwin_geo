#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import GRIPPER_KEYPOINTS_METERS, pose7_wxyz_to_matrix, transform_points
from .ndf_adapter import NdfPointwiseAdapter


DEFAULT_DATA_DIR = Path(
    "/shared2/sz/robotwin_data/data/place_shoe_geometry_marker/"
    "demo_clean_3d_object_pc_geometry_marker/data"
)


def valid_xyz(point_cloud: np.ndarray) -> np.ndarray:
    value = np.asarray(point_cloud, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"point cloud must have shape (N,C>=3), got {value.shape}")
    xyz = value[:, :3]
    return xyz[~np.isclose(xyz, 0.0).all(axis=1)]


def deterministic_farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if len(value) == 0:
        raise ValueError("cannot sample an empty point cloud")
    if len(value) < count:
        repetitions = int(np.ceil(float(count) / float(len(value))))
        return np.tile(value, (repetitions, 1))[:count]
    if len(value) == count:
        return value.copy()
    selected = np.zeros((count,), dtype=np.int64)
    minimum_distance = np.full((len(value),), np.inf, dtype=np.float32)
    center = value.mean(axis=0)
    farthest = int(np.argmax(np.sum((value - center) ** 2, axis=1)))
    for index in range(count):
        selected[index] = farthest
        distance = np.sum((value - value[farthest]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        farthest = int(np.argmax(minimum_distance))
    return value[selected]


def identify_active_arm(root: h5py.File) -> tuple[str, np.ndarray]:
    left = np.asarray(root["endpose/left_gripper"], dtype=np.float64).reshape(-1)
    right = np.asarray(root["endpose/right_gripper"], dtype=np.float64).reshape(-1)
    closing = [name for name, values in (("left", left), ("right", right)) if values.min() < 0.5]
    if len(closing) != 1:
        raise ValueError(f"expected exactly one active gripper, found {closing}")
    arm = closing[0]
    return arm, left if arm == "left" else right


def closed_frame(gripper: np.ndarray, threshold: float) -> int:
    candidates = np.flatnonzero(np.asarray(gripper) <= float(threshold))
    if not len(candidates):
        raise ValueError(f"gripper never reaches closed threshold {threshold}")
    return int(candidates[0])


def closing_pose_stability(
    poses: np.ndarray, gripper: np.ndarray, target_frame: int
) -> tuple[float, float]:
    starts = np.flatnonzero(np.asarray(gripper) < 0.99)
    start = int(starts[0]) if len(starts) else int(target_frame)
    segment = np.asarray(poses[start : target_frame + 1], dtype=np.float64)
    reference = np.asarray(poses[target_frame], dtype=np.float64)
    translation = float(np.linalg.norm(segment[:, :3] - reference[:3], axis=1).max())
    quaternions = segment[:, 3:] / np.linalg.norm(segment[:, 3:], axis=1, keepdims=True)
    reference_q = reference[3:] / np.linalg.norm(reference[3:])
    dots = np.abs(quaternions @ reference_q)
    rotation = float(np.rad2deg(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))).max())
    return translation, rotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract camera-only object inputs and expert grasp-frame labels."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--observation-frame", type=int, default=0)
    parser.add_argument("--closed-threshold", type=float, default=0.02)
    parser.add_argument("--ndf-checkpoint", type=Path)
    parser.add_argument("--ndf-device", default="cpu")
    parser.add_argument(
        "--ndf-scalar-only",
        action="store_true",
        help="Drop the checkpoint's learned 3D direction for a controlled ablation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    ndf = (
        NdfPointwiseAdapter(
            args.ndf_checkpoint,
            device=args.ndf_device,
            include_vector_direction=not args.ndf_scalar_only,
        )
        if args.ndf_checkpoint is not None
        else None
    )
    arrays: dict[str, list] = {
        "points_world": [],
        "target_transform_world": [],
        "target_keypoints_world": [],
        "object_transform_world_label_only": [],
        "active_arm_right": [],
        "shoe_id_label_only": [],
        "episode_id": [],
        "observation_frame": [],
        "grasp_frame": [],
    }
    if ndf is not None:
        arrays["ndf_features"] = []
    diagnostics = []
    for episode in range(int(args.episodes)):
        path = data_dir / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as root:
            arm, gripper = identify_active_arm(root)
            grasp_frame = closed_frame(gripper, args.closed_threshold)
            observation_frame = int(args.observation_frame)
            if not 0 <= observation_frame < grasp_frame:
                raise ValueError(
                    f"episode {episode}: observation frame {observation_frame} is not pre-grasp"
                )
            all_points = valid_xyz(root["object_pointcloud/{A}"][observation_frame])
            points = deterministic_farthest_points(all_points, int(args.num_points))
            target_transform = pose7_wxyz_to_matrix(
                root[f"endpose/{arm}_endpose"][grasp_frame]
            )
            target_keypoints = transform_points(
                target_transform, GRIPPER_KEYPOINTS_METERS
            )
            object_transform = pose7_wxyz_to_matrix(
                root["task_state/object_pose_A"][observation_frame]
            )
            arrays["points_world"].append(points.astype(np.float32))
            arrays["target_transform_world"].append(target_transform.astype(np.float32))
            arrays["target_keypoints_world"].append(target_keypoints.astype(np.float32))
            arrays["object_transform_world_label_only"].append(
                object_transform.astype(np.float32)
            )
            arrays["active_arm_right"].append(float(arm == "right"))
            arrays["shoe_id_label_only"].append(
                int(np.asarray(root["task_state/shoe_id"][0]).reshape(-1)[0])
            )
            arrays["episode_id"].append(episode)
            arrays["observation_frame"].append(observation_frame)
            arrays["grasp_frame"].append(grasp_frame)
            if ndf is not None:
                arrays["ndf_features"].append(ndf.encode(all_points, points))
            pose_path = np.asarray(root[f"endpose/{arm}_endpose"])
            translation_drift, rotation_drift = closing_pose_stability(
                pose_path, gripper, grasp_frame
            )
            diagnostics.append(
                {
                    "episode": episode,
                    "active_arm": arm,
                    "shoe_id": arrays["shoe_id_label_only"][-1],
                    "observation_frame": observation_frame,
                    "grasp_frame": grasp_frame,
                    "closing_translation_drift_m": translation_drift,
                    "closing_rotation_drift_deg": rotation_drift,
                }
            )
            print(
                json.dumps(
                    {
                        "episode": episode,
                        "shoe_id": arrays["shoe_id_label_only"][-1],
                        "arm": arm,
                        "grasp_frame": grasp_frame,
                    }
                ),
                flush=True,
            )
    integer_keys = {
        "shoe_id_label_only",
        "episode_id",
        "observation_frame",
        "grasp_frame",
    }
    serialized = {
        key: np.asarray(value, dtype=np.int64 if key in integer_keys else np.float32)
        for key, value in arrays.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **serialized)
    metadata = {
        "schema_version": 1,
        "output": str(output),
        "data_dir": str(data_dir),
        "episodes": int(args.episodes),
        "samples": int(len(serialized["episode_id"])),
        "num_points": int(args.num_points),
        "ndf_checkpoint": str(args.ndf_checkpoint.resolve()) if args.ndf_checkpoint else None,
        "ndf_feature_dim": int(ndf.output_dim) if ndf is not None else 0,
        "ndf_vector_direction": bool(ndf.include_vector_direction) if ndf is not None else False,
        "policy_inputs": [
            "segmented operated-object camera point cloud in world coordinates",
            "active arm identity",
        ] + (["frozen pointwise NDF descriptors"] if ndf is not None else []),
        "label_or_evaluator_only": [
            "expert end-effector grasp pose",
            "simulator object pose",
            "shoe ID",
        ],
        "diagnostics": diagnostics,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: metadata[key] for key in ("output", "samples", "ndf_feature_dim")}, indent=2))


if __name__ == "__main__":
    main()
