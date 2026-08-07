#!/usr/bin/env python3
"""Express geometry, robot state, and EEF actions in a camera-estimated target frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .functional_action_frame import (
    frame9_transform,
    relative_frame9_to_target_frame,
    transform_point_features_to_target_frame,
)


def rotation_to_sixd(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=np.float64)[:, :2].reshape(6)


def sixd_to_rotation(value: np.ndarray) -> np.ndarray:
    columns = np.asarray(value, dtype=np.float64).reshape(3, 2)
    first = columns[:, 0] / max(np.linalg.norm(columns[:, 0]), 1e-12)
    second = columns[:, 1] - first * np.dot(first, columns[:, 1])
    second = second / max(np.linalg.norm(second), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    w, x, y, z = value[3:] / np.linalg.norm(value[3:])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat((x, y, z, w)).as_matrix()
    result[:3, 3] = value[:3]
    return result


def matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quaternion = np.asarray((w, x, y, z))
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return np.concatenate((matrix[:3, 3], quaternion)).astype(np.float32)


def transform_xyz(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - frame[:3, 3]) @ frame[:3, :3]


def transform_pose7(pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return matrix_to_pose7(np.linalg.inv(frame) @ pose7_to_matrix(pose))


def transform_pose9(pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    translation = transform_xyz(value[:3], frame)
    rotation = frame[:3, :3].T @ sixd_to_rotation(value[3:9])
    return np.concatenate((translation, rotation_to_sixd(rotation))).astype(np.float32)


def transform_action(action: np.ndarray, frame: np.ndarray) -> np.ndarray:
    result = np.asarray(action, dtype=np.float64).copy()
    world_from_target = frame[:3, :3]
    for offset in (0, 7):
        result[..., offset : offset + 3] = (
            result[..., offset : offset + 3] @ world_from_target
        )
        result[..., offset + 3 : offset + 6] = (
            result[..., offset + 3 : offset + 6] @ world_from_target
        )
    return result.astype(np.float32)


def eef_state_from_pose7(poses: np.ndarray, grippers: tuple[float, float]) -> np.ndarray:
    values = []
    for pose, gripper in zip(poses, grippers):
        matrix = pose7_to_matrix(pose)
        values.extend(matrix[:3, 3])
        values.extend(rotation_to_sixd(matrix[:3, :3]))
        values.append(float(gripper))
    return np.asarray(values, dtype=np.float32)


def canonicalize(payload: dict[str, np.ndarray], frames: np.ndarray) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value).copy() for key, value in payload.items()}
    episode_indices = payload["episode_index"].astype(np.int64)
    for sample, episode in enumerate(episode_indices):
        frame = frames[episode]
        frame9 = np.concatenate(
            (frame[:3, 3], frame[:3, 0], frame[:3, 1])
        ).astype(np.float32)
        result["points_a"][sample] = transform_point_features_to_target_frame(
            payload["points_a"][sample], frame9
        )
        result["points_b"][sample] = transform_point_features_to_target_frame(
            payload["points_b"][sample], frame9
        )
        result["current_anchors"][sample] = transform_xyz(
            payload["current_anchors"][sample], frame
        )
        result["current_object_pose9"][sample] = transform_pose9(
            payload["current_object_pose9"][sample], frame
        )
        current = np.stack(
            [transform_pose7(pose, frame) for pose in payload["current_eef_pose7"][sample]]
        )
        future = np.stack(
            [
                np.stack([transform_pose7(pose, frame) for pose in step])
                for step in payload["future_eef_pose7"][sample]
            ]
        )
        result["current_eef_pose7"][sample] = current
        result["future_eef_pose7"][sample] = future
        result["action"][sample] = transform_action(payload["action"][sample], frame)
        result["state"][sample] = eef_state_from_pose7(
            current, (payload["state"][sample, 9], payload["state"][sample, 19])
        )
        if "target_frame9" in payload:
            result["target_frame9"][sample] = relative_frame9_to_target_frame(
                payload["target_frame9"][sample], frame9
            )
        if "goal_frame9" in payload:
            result["goal_frame9"][sample] = np.asarray(
                (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                dtype=np.float32,
            )
        if "goal_translation_error_xyz_m" in payload:
            result["goal_translation_error_xyz_m"][sample] = (
                np.asarray(payload["goal_translation_error_xyz_m"][sample])
                @ frame[:3, :3]
            )
        if "goal_rotation_error_rotvec" in payload:
            world_rotation = Rotation.from_rotvec(
                payload["goal_rotation_error_rotvec"][sample]
            ).as_matrix()
            local_rotation = frame[:3, :3].T @ world_rotation @ frame[:3, :3]
            result["goal_rotation_error_rotvec"][sample] = Rotation.from_matrix(
                local_rotation
            ).as_rotvec()

    for episode, frame in enumerate(frames):
        result["episode_flow"][episode] = transform_xyz(
            payload["episode_flow"][episode], frame
        )
        result["episode_pose"][episode] = np.stack(
            [transform_pose9(pose, frame) for pose in payload["episode_pose"][episode]]
        )
    result["target_frame_camera"] = np.asarray(frames, dtype=np.float32)
    return result


def episode_frames_from_sample_frame9(
    payload: dict[str, np.ndarray], key: str = "goal_frame9"
) -> np.ndarray:
    """Recover one stable world target frame per episode from sample rows."""
    if key not in payload:
        raise KeyError(f"payload is missing {key}")
    episode_index = np.asarray(payload["episode_index"], dtype=np.int64)
    if episode_index.ndim != 1 or len(episode_index) != len(payload[key]):
        raise ValueError(f"{key} must align one-to-one with episode_index")
    episodes = np.unique(episode_index)
    if not np.array_equal(episodes, np.arange(len(episodes))):
        raise ValueError("episode_index must densely cover [0, episode_count)")
    frames = []
    for episode in episodes:
        rows = np.flatnonzero(episode_index == int(episode))
        sample_frames = frame9_transform(np.asarray(payload[key][rows]))
        reference = sample_frames[0]
        if not np.allclose(sample_frames, reference[None], atol=2e-5):
            position_error = float(
                np.max(np.linalg.norm(sample_frames[:, :3, 3] - reference[:3, 3], axis=-1))
            )
            rotation_error = float(
                np.max(
                    Rotation.from_matrix(
                        sample_frames[:, :3, :3]
                        @ reference[:3, :3].T
                    ).magnitude()
                )
            )
            raise ValueError(
                f"{key} is not stable in episode {episode}: "
                f"position_error={position_error}, rotation_error={rotation_error}"
            )
        frames.append(reference)
    return np.asarray(frames, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frame-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    with np.load(args.frame_dataset, allow_pickle=False) as archive:
        frame_ids = archive["episode_id"].astype(np.int64)
        frames_unordered = archive["target_frame_camera"].astype(np.float64)
    if set(frame_ids.tolist()) != set(range(len(payload["episode_flow"]))):
        raise ValueError("frame dataset must contain exactly one frame per episode")
    frames = np.empty_like(frames_unordered)
    frames[frame_ids] = frames_unordered
    result = canonicalize(payload, frames)

    # Rigid coordinate changes must preserve endpoint supervision exactly.
    original_step_norm = np.linalg.norm(payload["action"][..., :3], axis=-1)
    canonical_step_norm = np.linalg.norm(result["action"][..., :3], axis=-1)
    invariant_error = float(np.max(np.abs(original_step_norm - canonical_step_norm)))
    if invariant_error > 2e-6:
        raise RuntimeError(f"translation norm invariance failed: {invariant_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    metadata = {
        "input": str(args.input.resolve()),
        "frame_dataset": str(args.frame_dataset.resolve()),
        "output": str(args.output.resolve()),
        "samples": int(len(payload["action"])),
        "episodes": int(len(frames)),
        "frame_source": "camera-only segmented target XYZRGB geometry-marker fit",
        "canonicalized": [
            "operated and target object points",
            "object task flow",
            "EEF state and pose labels",
            "EEF delta translations and rotations",
        ],
        "translation_norm_invariance_max_m": invariant_error,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
