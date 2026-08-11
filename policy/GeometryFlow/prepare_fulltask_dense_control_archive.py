#!/usr/bin/env python3
"""Build full-task Transformer inputs with replay-equivalent dense actions.

Every policy label is an exact low-level command block from the demonstration:
12 arm position/velocity channels plus two grippers are never reconstructed
from observed EEF states and never passed through an online motion planner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .build_task_flow_dataset import eef_state20
from .prepare_fulltask_v2_archive import fixed_sample, local_relation_tokens


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--episodes", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--relation-archive", type=Path)
    result.add_argument("--history", type=int, default=3)
    result.add_argument("--history-stride", type=int, default=2)
    result.add_argument("--action-horizon", type=int, default=15)
    result.add_argument("--scene-points", type=int, default=48)
    result.add_argument("--object-points", type=int, default=32)
    result.add_argument("--anchors", type=int, default=16)
    result.add_argument("--overwrite", action="store_true")
    return result


def relation_lookup(path: Path | None) -> tuple[dict, dict[tuple[int, int], int]]:
    if path is None:
        return {}, {}
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "episode_id",
        "frame_index",
        "target_frame9",
        "goal_frame9",
        "source_frame_confidence",
        "goal_frame_confidence",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"relation archive lacks {missing}")
    lookup: dict[tuple[int, int], int] = {}
    for row, (episode, frame) in enumerate(
        zip(payload["episode_id"], payload["frame_index"])
    ):
        key = (int(episode), int(frame))
        if key in lookup:
            raise ValueError(f"duplicate relation row {key}")
        lookup[key] = row
    return payload, lookup


def policy_observation_indices(last_step: np.ndarray, horizon: int) -> np.ndarray:
    """Greedily select observations separated by one executed dense chunk."""
    value = np.asarray(last_step, dtype=np.int64).reshape(-1)
    if len(value) < 2 or value[0] != -1 or np.any(np.diff(value) < 0):
        raise ValueError("invalid observation-to-control index stream")
    selected = [0]
    while True:
        desired = int(value[selected[-1]]) + int(horizon)
        candidates = np.flatnonzero(value >= desired)
        candidates = candidates[candidates > selected[-1]]
        if not len(candidates):
            break
        selected.append(int(candidates[0]))
    return np.asarray(selected, dtype=np.int64)


def dense_action26(
    position: np.ndarray,
    arm_velocity: np.ndarray,
    start: int,
    horizon: int,
) -> np.ndarray:
    stop = min(int(start) + int(horizon), len(position))
    pos = np.asarray(position[start:stop], dtype=np.float32)
    vel = np.asarray(arm_velocity[start:stop], dtype=np.float32)
    if not len(pos):
        raise ValueError("dense action begins after the command trace")
    if len(pos) < int(horizon):
        pad = int(horizon) - len(pos)
        pos = np.concatenate((pos, np.repeat(pos[-1:], pad, axis=0)), axis=0)
        vel = np.concatenate((vel, np.repeat(vel[-1:], pad, axis=0)), axis=0)
    return np.concatenate(
        (
            pos[:, :6],
            vel[:, :6],
            pos[:, 6:7],
            pos[:, 7:13],
            vel[:, 6:12],
            pos[:, 13:14],
        ),
        axis=-1,
    ).astype(np.float32)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    relation, relation_rows = relation_lookup(args.relation_archive)
    frames: list[dict[str, np.ndarray | int]] = []
    diagnostics = []
    for episode in range(int(args.episodes)):
        path = args.data_dir / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as archive:
            last_step = np.asarray(
                archive["dense_control/observation_last_step_index"], dtype=np.int64
            ).reshape(-1)
            position = np.asarray(archive["dense_control/position"], dtype=np.float32)
            velocity = np.asarray(
                archive["dense_control/arm_velocity"], dtype=np.float32
            )
            selected = policy_observation_indices(last_step, args.action_horizon)
            shoe_id = int(np.asarray(archive["task_state/shoe_id"][0]).reshape(-1)[0])
            for observation_frame in selected:
                command_start = int(last_step[observation_frame]) + 1
                if command_start >= len(position):
                    continue
                points_a_full = np.asarray(
                    archive["object_pointcloud/{A}"][observation_frame],
                    dtype=np.float32,
                )
                points_b_full = np.asarray(
                    archive["object_pointcloud/{B}"][observation_frame],
                    dtype=np.float32,
                )
                points_a = fixed_sample(points_a_full, args.object_points)
                points_b = fixed_sample(points_b_full, args.object_points, offset=1)
                scene = fixed_sample(
                    np.concatenate((points_a_full, points_b_full), axis=0),
                    args.scene_points,
                )
                state = eef_state20(
                    archive["endpose/left_endpose"][observation_frame],
                    archive["endpose/left_gripper"][observation_frame],
                    archive["endpose/right_endpose"][observation_frame],
                    archive["endpose/right_gripper"][observation_frame],
                )
                relation_row = relation_rows.get((episode, int(observation_frame)))
                if args.relation_archive is not None and relation_row is None:
                    raise KeyError(
                        f"no relation token for episode/frame {(episode, int(observation_frame))}"
                    )
                if relation_row is None:
                    local = np.zeros((args.anchors, 12), dtype=np.float32)
                    global_relation = np.zeros(10, dtype=np.float32)
                else:
                    relative = relation["target_frame9"][relation_row]
                    goal = relation["goal_frame9"][relation_row]
                    local = local_relation_tokens(
                        points_a_full, state, relative, goal, args.anchors
                    )
                    confidence = float(
                        relation["source_frame_confidence"][relation_row]
                    ) * float(relation["goal_frame_confidence"][relation_row])
                    global_relation = np.concatenate(
                        (relative, [confidence])
                    ).astype(np.float32)
                frames.append(
                    {
                        "scene": scene[:, :6],
                        "points_a": points_a[:, :6],
                        "points_b": points_b[:, :6],
                        "local": local,
                        "global": global_relation,
                        "state": state,
                        "action": dense_action26(
                            position,
                            velocity,
                            command_start,
                            args.action_horizon,
                        ),
                        "episode_id": episode,
                        "shoe_id": shoe_id,
                        "frame_index": int(observation_frame),
                        "control_start": command_start,
                    }
                )
            diagnostics.append(
                {
                    "episode": episode,
                    "shoe_id": shoe_id,
                    "observations": int(len(last_step)),
                    "dense_steps": int(len(position)),
                    "policy_samples": int(sum(row["episode_id"] == episode for row in frames)),
                }
            )

    # Add causal temporal histories without crossing episode boundaries.
    histories = []
    for row, frame in enumerate(frames):
        episode_rows = [
            index
            for index in range(row + 1)
            if frames[index]["episode_id"] == frame["episode_id"]
        ]
        position_in_episode = len(episode_rows) - 1
        histories.append(
            [
                episode_rows[
                    max(
                        0,
                        position_in_episode
                        - (int(args.history) - 1 - step) * int(args.history_stride),
                    )
                ]
                for step in range(int(args.history))
            ]
        )
    history_rows = np.asarray(histories, dtype=np.int64)
    temporal_keys = ("scene", "points_a", "points_b", "local", "global", "state")
    output = {
        key: np.asarray([frame[key] for frame in frames], dtype=np.float32)[history_rows]
        for key in temporal_keys
    }
    for key, dtype in (
        ("action", np.float32),
        ("episode_id", np.int64),
        ("shoe_id", np.int64),
        ("frame_index", np.int64),
        ("control_start", np.int64),
    ):
        output[key] = np.asarray([frame[key] for frame in frames], dtype=dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        "schema_version": 1,
        "action_representation": "dense_joint26",
        "data_dir": str(args.data_dir.resolve()),
        "relation_archive": (
            None if args.relation_archive is None else str(args.relation_archive.resolve())
        ),
        "output": str(args.output.resolve()),
        "episodes": int(args.episodes),
        "samples": int(len(frames)),
        "history": int(args.history),
        "history_stride": int(args.history_stride),
        "action_horizon": int(args.action_horizon),
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
