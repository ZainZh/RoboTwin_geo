from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from incremental_objpc_zarr import append_episode_to_buffer, open_or_reset_replay_buffer
from ndf_feature_utils import summarize_modes
from object_pointcloud_utils import (
    default_placeholder_order,
    extract_placeholder_point_cloud,
    load_hdf5,
    load_scene_info,
    merge_object_point_clouds,
    parse_placeholder_list,
    parse_target_extents,
    valid_xyz_centroid,
)
from se3_relation_token_utils import (
    RELATION_TOKEN_DIM,
    RELATION_TOKEN_KEY,
    SUPPORTED_RELATION_ROUTES,
    build_relation_token_from_task_state,
)


TASK_STATE_KEYS = (
    "object_pose_A",
    "object_pose_B",
    "goal_T_A_from_B_oracle",
    "shoe_id",
    "relation_phase",
)


def extract_task_state_frame(episode: dict, frame_idx: int) -> dict:
    task_state = episode.get("task_state")
    if not isinstance(task_state, dict):
        raise RuntimeError(
            "SE(3) relation preprocessing requires task_state pose metadata; recollect the dataset "
            "with the updated place_shoe_rotating_block task."
        )
    missing = [key for key in TASK_STATE_KEYS if key not in task_state]
    if missing:
        raise RuntimeError(
            "SE(3) relation task_state is incomplete; recollect the dataset. Missing keys: "
            + ", ".join(missing)
        )
    return {key: np.asarray(task_state[key])[int(frame_idx)] for key in TASK_STATE_KEYS}


def relation_token_for_frame(*, route: str, task_state: dict, goal_table: dict | None) -> np.ndarray:
    return build_relation_token_from_task_state(
        route=route, task_state=task_state, goal_table=goal_table
    )


def should_keep_frame(task_state: dict, *, placement_only: bool) -> bool:
    if not placement_only:
        return True
    phase = float(np.asarray(task_state["relation_phase"]).reshape(-1)[0])
    return phase > 0.0


def load_goal_table(path: str, route: str) -> dict | None:
    if route in {"baseline", "oracle"}:
        return None
    if not path:
        raise ValueError(f"--goal_table is required for route={route}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a DP3 object-PC dataset with a phase-gated explicit SE(3) relation token."
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--route", choices=SUPPORTED_RELATION_ROUTES, required=True)
    parser.add_argument("--goal_table", default="")
    parser.add_argument("--object_placeholders", default="{A},{B}")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--cluster_eps", type=float, default=0.04)
    parser.add_argument("--min_cluster_points", type=int, default=24)
    parser.add_argument("--table_quantile", type=float, default=0.08)
    parser.add_argument("--table_margin", type=float, default=0.01)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    route = str(args.route)
    output_suffix = args.output_suffix or "-objpc-se3-relation-" + route.replace("_", "-")
    load_dir = os.path.join("../../data", args.task_name, args.task_config)
    save_dir = f"./data/{args.task_name}-{args.task_config}-{args.expert_data_num}{output_suffix}.zarr"
    meta_path = f"./data/{args.task_name}-{args.task_config}-{args.expert_data_num}{output_suffix}_meta.json"

    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    first_episode = load_hdf5(os.path.join(load_dir, "data/episode0.hdf5"))
    extract_task_state_frame(first_episode, 0)
    placeholders = parse_placeholder_list(args.object_placeholders)
    if not placeholders:
        placeholders = default_placeholder_order(scene_info, first_episode)
    if "{A}" not in placeholders or "{B}" not in placeholders:
        raise ValueError("SE(3) relation comparison requires {A} (shoe) and {B} (ramp).")
    goal_table = load_goal_table(args.goal_table, route)

    replay_buffer, start_episode = open_or_reset_replay_buffer(save_dir)
    existing_meta = {}
    if Path(meta_path).is_file():
        existing_meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        existing_route = existing_meta.get("relation_route")
        if existing_route not in {None, route}:
            raise RuntimeError(
                f"Existing zarr route is {existing_route!r}, requested {route!r}; use a different suffix."
            )
    episode_stats = list(existing_meta.get("episodes", []))[:start_episode]

    for current_ep in range(start_episode, int(args.expert_data_num)):
        print(f"processing SE(3) relation episode: {current_ep + 1} / {args.expert_data_num}", end="\r")
        episode = load_hdf5(os.path.join(load_dir, f"data/episode{current_ep}.hdf5"))
        vector_all = np.asarray(episode["vector"])
        extract_task_state_frame(episode, 0)
        prev_centroids = {placeholder: None for placeholder in placeholders}
        local_modes = {placeholder: Counter() for placeholder in placeholders}
        asset_specs = {}
        for placeholder in placeholders:
            _, asset_spec = parse_target_extents(scene_info, current_ep, placeholder)
            asset_specs[placeholder] = asset_spec

        point_clouds = []
        states = []
        actions = []
        relation_tokens = []
        active_frames = 0
        for frame_idx in range(vector_all.shape[0]):
            object_clouds = []
            for placeholder in placeholders:
                target_extents, _ = parse_target_extents(scene_info, current_ep, placeholder)
                object_pc, extract_meta = extract_placeholder_point_cloud(
                    episode,
                    frame_idx=frame_idx,
                    placeholder=placeholder,
                    target_num_points=int(args.target_num_points),
                    target_extents=target_extents,
                    prev_centroid=prev_centroids[placeholder],
                    cluster_eps=float(args.cluster_eps),
                    min_cluster_points=int(args.min_cluster_points),
                    table_quantile=float(args.table_quantile),
                    table_margin=float(args.table_margin),
                )
                object_clouds.append(object_pc)
                local_modes[placeholder][str(extract_meta.get("mode", "unknown"))] += 1
                centroid = valid_xyz_centroid(object_pc)
                if centroid is not None:
                    prev_centroids[placeholder] = centroid

            if frame_idx != vector_all.shape[0] - 1:
                point_clouds.append(
                    merge_object_point_clouds(
                        object_clouds,
                        target_num_points=int(args.target_num_points),
                    ).astype(np.float32)
                )
                states.append(vector_all[frame_idx].astype(np.float32))
                token = relation_token_for_frame(
                    route=route,
                    task_state=extract_task_state_frame(episode, frame_idx),
                    goal_table=goal_table,
                )
                relation_tokens.append(token)
                active_frames += int(float(token[-1]) > 0.0)
            if frame_idx != 0:
                actions.append(vector_all[frame_idx].astype(np.float32))

        episode_data = {
            "point_cloud": np.asarray(point_clouds, dtype=np.float32),
            "state": np.asarray(states, dtype=np.float32),
            "action": np.asarray(actions, dtype=np.float32),
            RELATION_TOKEN_KEY: np.asarray(relation_tokens, dtype=np.float32),
        }
        append_episode_to_buffer(replay_buffer, episode_data)
        episode_stats.append(
            {
                "episode": current_ep,
                "target_assets": asset_specs,
                "active_relation_fraction": float(active_frames / max(len(relation_tokens), 1)),
                "modes": {
                    placeholder: summarize_modes(local_modes[placeholder])
                    for placeholder in placeholders
                },
            }
        )
        meta = {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "expert_data_num": int(args.expert_data_num),
            "output_zarr": str(Path(save_dir).resolve()),
            "relation_route": route,
            "relation_token_key": RELATION_TOKEN_KEY,
            "relation_token_dim": RELATION_TOKEN_DIM,
            "relation_token_schema": "translation3_rotation6d_energy_valid",
            "relation_phase": "active_gripper_closed_and_shoe_lifted_3cm",
            "goal_table": str(Path(args.goal_table).resolve()) if args.goal_table else None,
            "object_placeholders": placeholders,
            "target_num_points": int(args.target_num_points),
            "episodes": episode_stats,
        }
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()


if __name__ == "__main__":
    main()
