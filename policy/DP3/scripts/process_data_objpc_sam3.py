import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from incremental_objpc_zarr import append_episode_to_buffer, open_or_reset_replay_buffer
from sam3_pointcloud_utils import (
    SAM3ProjectiveTracker,
    Sam3TrackingState,
    build_placeholder_prompt_map,
    extract_placeholder_point_cloud_sam3,
    load_hdf5_with_cameras,
    load_scene_info,
    merge_object_point_clouds,
    parse_camera_list,
    parse_prompt_map,
    parse_target_extents,
)
from object_pointcloud_utils import default_placeholder_order, parse_placeholder_list
from ndf_feature_utils import summarize_modes


def infer_placeholder_order(scene_info: dict, first_episode: dict) -> list[str]:
    placeholders = []
    seen = set()

    def append_placeholder(value):
        value = str(value)
        if value not in seen:
            seen.add(value)
            placeholders.append(value)

    for key in default_placeholder_order(scene_info, first_episode):
        append_placeholder(key)

    if isinstance(scene_info, dict):
        episode_keys = sorted(
            [key for key in scene_info.keys() if isinstance(key, str) and key.startswith("episode_")]
        )
        for episode_key in episode_keys:
            episode_info = scene_info.get(episode_key, {})
            if not isinstance(episode_info, dict):
                continue
            info_dict = episode_info.get("info", {})
            if isinstance(info_dict, dict):
                for placeholder in info_dict.keys():
                    if str(placeholder).startswith("{"):
                        append_placeholder(placeholder)

    return placeholders


def load_or_init_meta(meta_path: str, *, task_name: str, task_config: str, expert_data_num: int):
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.setdefault("episodes", [])
        return meta
    return {
        "task_name": task_name,
        "task_config": task_config,
        "expert_data_num": int(expert_data_num),
        "episodes": [],
    }


def write_meta(meta_path: str, meta: dict):
    tmp_path = f"{meta_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process RoboTwin episodes into DP3 zarr using SAM3 masks projected onto scene point clouds."
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--object_placeholders", type=str, default="")
    parser.add_argument("--camera_names", type=str, default="head_camera,front_camera")
    parser.add_argument("--sam3_model", type=str, default="/home/zheng/Datasets/sam3/sam3.pt")
    parser.add_argument("--sam3_conf", type=float, default=0.6)
    parser.add_argument("--sam3_prompt_map", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="-objpc-sam3")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--min_mask_points", type=int, default=16)
    parser.add_argument("--text_refresh_every", type=int, default=15)
    parser.add_argument("--save_placeholder_point_clouds", action="store_true")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = int(args.expert_data_num)
    load_dir = os.path.join("../../data", str(task_name), str(task_config))
    save_dir = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}.zarr"
    final_meta_path = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}_meta.json"
    legacy_tmp_dir = f"{save_dir}.tmp"
    legacy_tmp_meta = f"{final_meta_path}.tmp"

    if os.path.exists(legacy_tmp_dir) and not os.path.exists(save_dir):
        try:
            shutil.rmtree(legacy_tmp_dir)
        except FileNotFoundError:
            pass
    if os.path.exists(legacy_tmp_meta):
        try:
            os.remove(legacy_tmp_meta)
        except FileNotFoundError:
            pass

    camera_names = parse_camera_list(args.camera_names)
    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    first_episode = load_hdf5_with_cameras(
        os.path.join(load_dir, "data/episode0.hdf5"),
        camera_names=camera_names,
    )
    placeholders = parse_placeholder_list(args.object_placeholders)
    if len(placeholders) == 0:
        placeholders = infer_placeholder_order(scene_info, first_episode)
    if len(placeholders) == 0:
        raise RuntimeError("No object placeholders were found for SAM3 object-pointcloud preprocessing.")

    prompt_overrides = parse_prompt_map(args.sam3_prompt_map)
    tracker = SAM3ProjectiveTracker(
        model_path=args.sam3_model,
        conf=float(args.sam3_conf),
        verbose=False,
    )

    replay_buffer, start_episode = open_or_reset_replay_buffer(save_dir)
    meta = load_or_init_meta(
        final_meta_path,
        task_name=task_name,
        task_config=task_config,
        expert_data_num=num,
    )
    meta.update(
        {
            "output_zarr": str(Path(save_dir).resolve()),
            "object_placeholders": placeholders,
            "camera_names": camera_names,
            "sam3_model": str(args.sam3_model),
            "sam3_conf": float(args.sam3_conf),
            "sam3_prompt_overrides": prompt_overrides,
            "text_refresh_every": int(args.text_refresh_every),
            "min_mask_points": int(args.min_mask_points),
            "save_placeholder_point_clouds": bool(args.save_placeholder_point_clouds),
        }
    )
    episode_stats = list(meta.get("episodes", []))
    if len(episode_stats) > start_episode:
        episode_stats = episode_stats[:start_episode]
    while len(episode_stats) < start_episode:
        episode_stats.append(
            {
                "episode": len(episode_stats),
                "recovered_without_stats": True,
            }
        )

    print(f"Resuming SAM3 preprocessing from episode {start_episode + 1} / {num}")

    for current_ep in range(start_episode, num):
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        episode = load_hdf5_with_cameras(load_path, camera_names=camera_names)
        vector_all = episode["vector"]

        prompt_map = build_placeholder_prompt_map(
            scene_info=scene_info,
            episode_idx=current_ep,
            placeholders=placeholders,
            prompt_overrides=prompt_overrides,
        )
        local_modes = {placeholder: Counter() for placeholder in placeholders}
        tracking_state = {
            placeholder: {
                camera_name: Sam3TrackingState()
                for camera_name in camera_names
            }
            for placeholder in placeholders
        }
        asset_specs = {}

        for placeholder in placeholders:
            _, asset_spec = parse_target_extents(scene_info, current_ep, placeholder)
            asset_specs[placeholder] = asset_spec

        episode_point_cloud_arrays = []
        episode_placeholder_point_cloud_arrays = {placeholder: [] for placeholder in placeholders}
        episode_state_arrays = []
        episode_joint_action_arrays = []

        for frame_idx in range(vector_all.shape[0]):
            per_placeholder_point_clouds = []
            for placeholder in placeholders:
                target_extents, _ = parse_target_extents(scene_info, current_ep, placeholder)
                object_pc, extract_meta = extract_placeholder_point_cloud_sam3(
                    episode,
                    frame_idx=frame_idx,
                    placeholder=placeholder,
                    prompt=prompt_map[placeholder],
                    camera_names=camera_names,
                    tracker=tracker,
                    tracking_state_by_camera=tracking_state[placeholder],
                    target_num_points=int(args.target_num_points),
                    target_extents=target_extents,
                    min_mask_points=int(args.min_mask_points),
                    text_refresh_every=int(args.text_refresh_every),
                )
                per_placeholder_point_clouds.append(object_pc)
                local_modes[placeholder][str(extract_meta.get("mode", "unknown"))] += 1
                if frame_idx != vector_all.shape[0] - 1 and args.save_placeholder_point_clouds:
                    episode_placeholder_point_cloud_arrays[placeholder].append(object_pc.astype(np.float32))

            merged_object_pc = merge_object_point_clouds(
                per_placeholder_point_clouds,
                target_num_points=int(args.target_num_points),
            )

            if frame_idx != vector_all.shape[0] - 1:
                episode_point_cloud_arrays.append(merged_object_pc.astype(np.float32))
                episode_state_arrays.append(vector_all[frame_idx])
            if frame_idx != 0:
                episode_joint_action_arrays.append(vector_all[frame_idx])

        episode_data = {
            "point_cloud": np.asarray(episode_point_cloud_arrays, dtype=np.float32),
            "state": np.asarray(episode_state_arrays, dtype=np.float32),
            "action": np.asarray(episode_joint_action_arrays, dtype=np.float32),
        }
        if args.save_placeholder_point_clouds:
            for placeholder, arrays in episode_placeholder_point_cloud_arrays.items():
                episode_data[f"object_point_cloud_{placeholder.strip('{}')}"] = np.asarray(arrays, dtype=np.float32)

        append_episode_to_buffer(replay_buffer, episode_data)

        episode_record = {
            "episode": current_ep,
            "placeholders": placeholders,
            "camera_names": camera_names,
            "prompts": prompt_map,
            "target_assets": asset_specs,
            "modes": {
                placeholder: summarize_modes(local_modes[placeholder])
                for placeholder in placeholders
            },
        }
        if len(episode_stats) == current_ep:
            episode_stats.append(episode_record)
        else:
            episode_stats[current_ep] = episode_record

        meta["episodes"] = episode_stats
        meta["completed_episodes"] = int(replay_buffer.n_episodes)
        write_meta(final_meta_path, meta)
        print(
            f"finished episode {current_ep + 1} / {num}, "
            f"persisted_steps={int(replay_buffer.n_steps)}, persisted_episodes={int(replay_buffer.n_episodes)}"
        )

    print()
    meta["episodes"] = episode_stats
    meta["completed_episodes"] = int(replay_buffer.n_episodes)
    write_meta(final_meta_path, meta)


if __name__ == "__main__":
    main()
