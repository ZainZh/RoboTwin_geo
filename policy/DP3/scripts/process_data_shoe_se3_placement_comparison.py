from __future__ import annotations

from data_path_utils import dp3_data_path, raw_task_data_dir

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from incremental_objpc_zarr import append_episode_to_buffer, open_or_reset_replay_buffer
from geometry_relation_estimator import create_estimator_from_spec
from object_pointcloud_utils import (
    default_placeholder_order,
    extract_placeholder_point_cloud,
    load_hdf5,
    load_scene_info,
    merge_object_point_clouds,
    parse_placeholder_list,
    parse_target_extents,
    summarize_modes,
    valid_xyz_centroid,
)
from process_data_se3_relation_hybrid import (
    extract_task_state_frame,
    load_goal_table,
    relation_token_for_frame,
    should_keep_frame,
)
from se3_relation_token_utils import (
    OBSERVATION_RELATION_ROUTES,
    RELATION_TOKEN_DIM,
    RELATION_TOKEN_KEY,
    SUPPORTED_RELATION_ROUTES,
    build_se3_relation_token,
)


COMPARISON_ROUTES = ("baseline",) + SUPPORTED_RELATION_ROUTES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build matched placement-only baseline/oracle/NDF SE(3) comparison datasets."
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--route", choices=COMPARISON_ROUTES, required=True)
    parser.add_argument("--goal_table", default="")
    parser.add_argument("--geometry_estimator_spec", default="")
    parser.add_argument("--geometry_device", default="")
    parser.add_argument("--object_placeholders", default="{A},{B}")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--cluster_eps", type=float, default=0.04)
    parser.add_argument("--min_cluster_points", type=int, default=24)
    parser.add_argument("--table_quantile", type=float, default=0.08)
    parser.add_argument("--table_margin", type=float, default=0.01)
    return parser


def default_output_suffix(route: str) -> str:
    if route == "baseline":
        return "-objpc-placement-only-baseline"
    return "-objpc-placement-only-se3-relation-" + route.replace("_", "-")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    route = str(args.route)
    output_suffix = args.output_suffix or default_output_suffix(route)
    load_dir = str(raw_task_data_dir(args.task_name, args.task_config))
    save_dir = str(dp3_data_path(f"{args.task_name}-{args.task_config}-{args.expert_data_num}{output_suffix}.zarr"))
    meta_path = str(dp3_data_path(f"{args.task_name}-{args.task_config}-{args.expert_data_num}{output_suffix}_meta.json"))

    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    first_episode = load_hdf5(os.path.join(load_dir, "data/episode0.hdf5"))
    extract_task_state_frame(first_episode, 0)
    placeholders = parse_placeholder_list(args.object_placeholders)
    if not placeholders:
        placeholders = default_placeholder_order(scene_info, first_episode)
    if "{A}" not in placeholders or "{B}" not in placeholders:
        raise ValueError("Placement comparison requires {A} (shoe) and {B} (ramp).")
    geometry_estimator = None
    if route in OBSERVATION_RELATION_ROUTES:
        if not args.geometry_estimator_spec:
            raise ValueError(f"--geometry_estimator_spec is required for route={route}")
        geometry_estimator = create_estimator_from_spec(
            args.geometry_estimator_spec,
            device_override=args.geometry_device or None,
        )
        goal_table = None
    else:
        goal_table = load_goal_table(args.goal_table, route)

    replay_buffer, start_episode = open_or_reset_replay_buffer(save_dir)
    existing_meta = {}
    if Path(meta_path).is_file():
        existing_meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        existing_route = existing_meta.get("comparison_route")
        if existing_route not in {None, route}:
            raise RuntimeError(
                f"Existing zarr route is {existing_route!r}, requested {route!r}; use a different suffix."
            )
    episode_stats = list(existing_meta.get("episodes", []))[:start_episode]

    for current_ep in range(start_episode, int(args.expert_data_num)):
        print(
            f"processing placement comparison episode: {current_ep + 1} / {args.expert_data_num}",
            end="\r",
        )
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
        source_frame_indices = []
        for frame_idx in range(vector_all.shape[0] - 1):
            task_state = extract_task_state_frame(episode, frame_idx)
            if not should_keep_frame(task_state, placement_only=True):
                continue

            object_clouds = []
            object_cloud_by_placeholder = {}
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
                object_cloud_by_placeholder[placeholder] = object_pc
                local_modes[placeholder][str(extract_meta.get("mode", "unknown"))] += 1
                centroid = valid_xyz_centroid(object_pc)
                if centroid is not None:
                    prev_centroids[placeholder] = centroid

            point_clouds.append(
                merge_object_point_clouds(
                    object_clouds,
                    target_num_points=int(args.target_num_points),
                ).astype(np.float32)
            )
            states.append(vector_all[frame_idx].astype(np.float32))
            actions.append(vector_all[frame_idx + 1].astype(np.float32))
            source_frame_indices.append(frame_idx)
            if route != "baseline":
                if route in OBSERVATION_RELATION_ROUTES:
                    prediction = geometry_estimator.estimate_goal(
                        object_cloud_by_placeholder["{A}"],
                        object_cloud_by_placeholder["{B}"],
                    )
                    relation_tokens.append(
                        build_se3_relation_token(
                            object_pose_a=task_state["object_pose_A"],
                            object_pose_b=task_state["object_pose_B"],
                            goal_a_from_b=prediction.goal_t_a_from_b,
                            phase_gate=float(
                                np.asarray(task_state["relation_phase"]).reshape(-1)[0]
                            ),
                            solver_energy=prediction.solver_energy,
                            confidence=prediction.confidence,
                        )
                    )
                else:
                    relation_tokens.append(
                        relation_token_for_frame(
                            route=route,
                            task_state=task_state,
                            goal_table=goal_table,
                        )
                    )

        if not point_clouds:
            raise RuntimeError(
                f"episode {current_ep} has no placement-phase frames; verify relation_phase collection."
            )
        episode_data = {
            "point_cloud": np.asarray(point_clouds, dtype=np.float32),
            "state": np.asarray(states, dtype=np.float32),
            "action": np.asarray(actions, dtype=np.float32),
        }
        if route != "baseline":
            episode_data[RELATION_TOKEN_KEY] = np.asarray(relation_tokens, dtype=np.float32)
        append_episode_to_buffer(replay_buffer, episode_data)

        episode_stats.append(
            {
                "episode": current_ep,
                "target_assets": asset_specs,
                "source_frame_indices": source_frame_indices,
                "kept_frames": len(source_frame_indices),
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
            "comparison_route": route,
            "placement_only": True,
            "policy_start_phase": "expert_grasp_and_lift_then_policy_place",
            "relation_token_key": RELATION_TOKEN_KEY if route != "baseline" else None,
            "relation_token_dim": RELATION_TOKEN_DIM if route != "baseline" else 0,
            "relation_token_schema": "translation3_rotation6d_energy_valid",
            "goal_table": str(Path(args.goal_table).resolve()) if args.goal_table else None,
            "geometry_estimator_spec": (
                str(Path(args.geometry_estimator_spec).resolve())
                if args.geometry_estimator_spec
                else None
            ),
            "object_placeholders": placeholders,
            "target_num_points": int(args.target_num_points),
            "episodes": episode_stats,
        }
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()


if __name__ == "__main__":
    main()
