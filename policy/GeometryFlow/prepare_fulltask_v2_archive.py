#!/usr/bin/env python3
"""Turn a camera NDF relation archive into temporal full-task policy inputs.

Only fields available to the deployed policy are used: segmented A/B camera
clouds, camera-estimated NDF/target frames and robot proprioception.  Existing
simulator pose fields in the source archive are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .functional_action_frame import frame9_rotation


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--history", type=int, default=3)
    result.add_argument("--history-stride", type=int, default=2)
    result.add_argument("--scene-points", type=int, default=48)
    result.add_argument("--object-points", type=int, default=32)
    result.add_argument("--anchors", type=int, default=16)
    result.add_argument("--sample-stride", type=int, default=1)
    result.add_argument("--overwrite", action="store_true")
    return result


def fixed_sample(points: np.ndarray, count: int, offset: int = 0) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3 or len(value) == 0:
        raise ValueError(f"invalid point cloud {value.shape}")
    indices = (np.linspace(0, len(value) - 1, int(count)).round().astype(np.int64) + int(offset)) % len(value)
    return value[indices]


def episode_rows(episode_id: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(episode): np.flatnonzero(episode_id == episode)
        for episode in np.unique(episode_id)
    }


def history_lookup(
    episode_id: np.ndarray, history: int, history_stride: int
) -> np.ndarray:
    lookup = np.empty((len(episode_id), int(history)), dtype=np.int64)
    for rows in episode_rows(episode_id).values():
        for position, row in enumerate(rows):
            lookup[row] = [
                rows[max(0, position - (int(history) - 1 - step) * int(history_stride))]
                for step in range(int(history))
            ]
    return lookup


def local_relation_tokens(
    points_a: np.ndarray,
    state: np.ndarray,
    relative_frame9: np.ndarray,
    goal_frame9: np.ndarray,
    count: int,
) -> np.ndarray:
    anchors = fixed_sample(points_a, int(count))[:, :3]
    relative_rotation = frame9_rotation(relative_frame9)
    goal_position = np.asarray(goal_frame9[:3], dtype=np.float32)
    current_position = goal_position - np.asarray(relative_frame9[:3], dtype=np.float32)
    desired = (
        (anchors - current_position[None]) @ relative_rotation.T
        + goal_position[None]
    )
    flow = desired - anchors
    left_offset = anchors - np.asarray(state[:3], dtype=np.float32)[None]
    right_offset = anchors - np.asarray(state[10:13], dtype=np.float32)[None]
    return np.concatenate(
        (anchors - current_position[None], left_offset, right_offset, flow), axis=-1
    ).astype(np.float32)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "points_a",
        "points_b",
        "state",
        "action",
        "episode_id",
        "frame_index",
        "shoe_id",
        "target_frame9",
        "goal_frame9",
        "source_frame_confidence",
        "goal_frame_confidence",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"input archive lacks {missing}")
    episode_id = np.asarray(payload["episode_id"], dtype=np.int64)
    lookup = history_lookup(episode_id, args.history, args.history_stride)
    selected = np.arange(0, len(episode_id), int(args.sample_stride), dtype=np.int64)
    history_rows = lookup[selected]
    sample_count = len(selected)
    output = {
        "scene": np.empty(
            (sample_count, args.history, args.scene_points, 6), dtype=np.float32
        ),
        "points_a": np.empty(
            (sample_count, args.history, args.object_points, 6), dtype=np.float32
        ),
        "points_b": np.empty(
            (sample_count, args.history, args.object_points, 6), dtype=np.float32
        ),
        "local": np.empty(
            (sample_count, args.history, args.anchors, 12), dtype=np.float32
        ),
        "global": np.empty((sample_count, args.history, 10), dtype=np.float32),
        "state": np.asarray(payload["state"][history_rows], dtype=np.float32),
        "action": np.asarray(payload["action"][selected], dtype=np.float32),
        "episode_id": episode_id[selected],
        "shoe_id": np.asarray(payload["shoe_id"][selected], dtype=np.int64),
        "frame_index": np.asarray(payload["frame_index"][selected], dtype=np.int64),
        "source_index": selected,
    }
    for sample, rows in enumerate(history_rows):
        for time_index, row in enumerate(rows):
            points_a = fixed_sample(payload["points_a"][row], args.object_points)
            points_b = fixed_sample(payload["points_b"][row], args.object_points, offset=1)
            combined = np.concatenate(
                (payload["points_a"][row], payload["points_b"][row]), axis=0
            )
            output["scene"][sample, time_index] = fixed_sample(
                combined, args.scene_points, offset=time_index
            )[:, :6]
            output["points_a"][sample, time_index] = points_a[:, :6]
            output["points_b"][sample, time_index] = points_b[:, :6]
            output["local"][sample, time_index] = local_relation_tokens(
                payload["points_a"][row],
                payload["state"][row],
                payload["target_frame9"][row],
                payload["goal_frame9"][row],
                args.anchors,
            )
            confidence = float(payload["source_frame_confidence"][row]) * float(
                payload["goal_frame_confidence"][row]
            )
            output["global"][sample, time_index] = np.concatenate(
                (payload["target_frame9"][row], [confidence])
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "samples": int(sample_count),
        "episodes": int(len(np.unique(output["episode_id"]))),
        "history": int(args.history),
        "history_stride": int(args.history_stride),
        "action_horizon": int(output["action"].shape[1]),
        "scene_points": int(args.scene_points),
        "object_points": int(args.object_points),
        "anchors": int(args.anchors),
        "deployment_inputs": [
            "camera A/B point clouds",
            "NDF source functional frame and observable confidence",
            "camera target frame",
            "bimanual EEF and gripper state",
        ],
        "simulator_pose_inputs": [],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
