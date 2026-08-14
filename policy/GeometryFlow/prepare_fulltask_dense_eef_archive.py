#!/usr/bin/env python3
"""Build dense camera/TAGRT histories with observation-aligned EEF deltas.

The source demonstrations contain camera and EEF observations every configured
dense-control interval.  This builder labels each selected observation with
the actual EEF displacement to the next selected observation, matching the
camera-only receding-horizon deployment interface used by the command-EEF
policy.  Low-level joint controls and simulator object poses are not policy
inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from policy.DP3.scripts.eef_delta_action_utils import (
    eef_delta_action14,
    eef_state20,
)

from .prepare_fulltask_dense_control_archive import (
    policy_observation_indices,
    relation_lookup,
)
from .prepare_fulltask_v2_archive import fixed_sample, local_relation_tokens


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--episodes", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--relation-archive", type=Path, required=True)
    result.add_argument("--history", type=int, default=3)
    result.add_argument("--history-stride", type=int, default=1)
    result.add_argument("--action-horizon", type=int, default=6)
    result.add_argument(
        "--control-stride",
        type=int,
        default=15,
        help="Minimum low-level control steps between selected observations.",
    )
    result.add_argument("--scene-points", type=int, default=48)
    result.add_argument("--object-points", type=int, default=32)
    result.add_argument("--anchors", type=int, default=16)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--drop-unmatched-terminal",
        action="store_true",
        help=(
            "Drop only a final selected observation when the relation archive "
            "ends one frame earlier. Task-flow archives label transitions and "
            "therefore intentionally omit the final self-transition frame."
        ),
    )
    return result


def observed_eef_action(
    archive: h5py.File, current: int, following: int
) -> np.ndarray:
    """Return the world-frame EEF delta between two camera observations."""
    return eef_delta_action14(
        archive["endpose/left_endpose"][current],
        archive["endpose/left_gripper"][current],
        archive["endpose/right_endpose"][current],
        archive["endpose/right_gripper"][current],
        archive["endpose/left_endpose"][following],
        archive["endpose/left_gripper"][following],
        archive["endpose/right_endpose"][following],
        archive["endpose/right_gripper"][following],
    )


def padded_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 14 or len(value) == 0:
        raise ValueError(f"actions must be non-empty [T,14], got {value.shape}")
    chunks = []
    for row in range(len(value)):
        chunk = value[row : row + int(horizon)]
        if len(chunk) < int(horizon):
            chunk = np.concatenate(
                (chunk, np.repeat(value[-1:], int(horizon) - len(chunk), axis=0)),
                axis=0,
            )
        chunks.append(chunk)
    return np.asarray(chunks, dtype=np.float32)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    if int(args.control_stride) <= 0:
        raise ValueError("control stride must be positive")
    relation, relation_rows = relation_lookup(args.relation_archive)
    frames: list[dict[str, np.ndarray | int]] = []
    diagnostics = []

    for episode in range(int(args.episodes)):
        path = args.data_dir / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as archive:
            last_step = np.asarray(
                archive["dense_control/observation_last_step_index"], dtype=np.int64
            ).reshape(-1)
            selected = policy_observation_indices(last_step, args.control_stride)
            terminal_key = (episode, int(selected[-1]))
            if (
                args.drop_unmatched_terminal
                and terminal_key not in relation_rows
                and int(selected[-1]) == len(last_step) - 1
            ):
                selected = selected[:-1]
            if len(selected) < 2:
                raise ValueError(f"episode {episode} has fewer than two dense observations")
            shoe_id = int(np.asarray(archive["task_state/shoe_id"][0]).reshape(-1)[0])

            actions = [
                observed_eef_action(archive, int(current), int(following))
                for current, following in zip(selected[:-1], selected[1:])
            ]
            # The final observation is a learned stop state, not a repeated
            # incoming motion.  Preserve its observed gripper values.
            actions.append(
                observed_eef_action(archive, int(selected[-1]), int(selected[-1]))
            )
            action_chunks = padded_action_chunks(
                np.asarray(actions, dtype=np.float32), args.action_horizon
            )

            for position, observation_frame in enumerate(selected):
                observation_frame = int(observation_frame)
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
                relation_row = relation_rows.get((episode, observation_frame))
                if relation_row is None:
                    raise KeyError(
                        f"no relation token for episode/frame {(episode, observation_frame)}"
                    )
                relative = relation["target_frame9"][relation_row]
                goal = relation["goal_frame9"][relation_row]
                local = local_relation_tokens(
                    points_a_full, state, relative, goal, args.anchors
                )
                confidence = float(
                    relation["source_frame_confidence"][relation_row]
                ) * float(relation["goal_frame_confidence"][relation_row])
                frames.append(
                    {
                        "scene": scene[:, :6],
                        "points_a": points_a[:, :6],
                        "points_b": points_b[:, :6],
                        "local": local,
                        "global": np.concatenate((relative, [confidence])).astype(
                            np.float32
                        ),
                        "state": state,
                        "action": action_chunks[position],
                        "episode_id": episode,
                        "shoe_id": shoe_id,
                        "frame_index": observation_frame,
                        "control_start": int(last_step[observation_frame]) + 1,
                        "is_terminal": bool(position == len(selected) - 1),
                    }
                )

            diagnostics.append(
                {
                    "episode": episode,
                    "shoe_id": shoe_id,
                    "recorded_observations": int(len(last_step)),
                    "selected_observations": int(len(selected)),
                    "first_control_step": int(last_step[selected[0]]) + 1,
                    "last_control_step": int(last_step[selected[-1]]),
                }
            )

    histories = []
    episode_rows: dict[int, list[int]] = {}
    for row, frame in enumerate(frames):
        episode = int(frame["episode_id"])
        rows = episode_rows.setdefault(episode, [])
        rows.append(row)
        position = len(rows) - 1
        histories.append(
            [
                rows[
                    max(
                        0,
                        position
                        - (int(args.history) - 1 - step) * int(args.history_stride),
                    )
                ]
                for step in range(int(args.history))
            ]
        )
    history_rows = np.asarray(histories, dtype=np.int64)
    temporal = ("scene", "points_a", "points_b", "local", "global", "state")
    output = {
        key: np.asarray([frame[key] for frame in frames], dtype=np.float32)[history_rows]
        for key in temporal
    }
    for key, dtype in (
        ("action", np.float32),
        ("episode_id", np.int64),
        ("shoe_id", np.int64),
        ("frame_index", np.int64),
        ("control_start", np.int64),
        ("is_terminal", np.bool_),
    ):
        output[key] = np.asarray([frame[key] for frame in frames], dtype=dtype)
    output["action_frame_ndf_source"] = np.asarray(0, dtype=np.int8)
    output["local_left_eef_query"] = np.asarray(0, dtype=np.int8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        "schema_version": 1,
        "action_representation": "dense_eef_delta14",
        "deployment_action_representation": "eef_delta14",
        "data_dir": str(args.data_dir.resolve()),
        "relation_archive": str(args.relation_archive.resolve()),
        "output": str(args.output.resolve()),
        "episodes": int(args.episodes),
        "samples": int(len(frames)),
        "history": int(args.history),
        "history_stride": int(args.history_stride),
        "action_horizon": int(args.action_horizon),
        "control_stride": int(args.control_stride),
        "simulator_pose_inputs": [],
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: metadata[key] for key in (
        "output", "episodes", "samples", "history", "action_horizon", "control_stride"
    )}), flush=True)


if __name__ == "__main__":
    main()
