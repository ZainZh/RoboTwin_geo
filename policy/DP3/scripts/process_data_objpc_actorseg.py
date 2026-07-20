from data_path_utils import dp3_data_path, raw_task_data_dir
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from actorseg_pointcloud_utils import (
    extract_placeholder_point_cloud_actorseg,
    load_hdf5_with_actorseg,
    parse_camera_list,
    parse_episode_actor_id_map,
)
from incremental_objpc_zarr import append_episode_to_buffer, open_or_reset_replay_buffer
from ndf_feature_utils import load_scene_info, summarize_modes
from object_pointcloud_utils import default_placeholder_order, merge_object_point_clouds, parse_placeholder_list


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
            target_info = episode_info.get("object_pointcloud", {}).get("targets", {})
            if isinstance(target_info, dict):
                for placeholder in target_info.keys():
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


def require_actor_id_targets(scene_info: dict, placeholders: list[str]):
    missing = []
    if not isinstance(scene_info, dict):
        raise RuntimeError("scene_info.json is missing or invalid for actor-segmentation preprocessing.")
    for episode_key, episode_info in scene_info.items():
        if not isinstance(episode_key, str) or not episode_key.startswith("episode_"):
            continue
        targets = episode_info.get("object_pointcloud", {}).get("targets", {}) if isinstance(episode_info, dict) else {}
        for placeholder in placeholders:
            actor_ids = targets.get(placeholder, {}).get("actor_ids", [])
            if len(actor_ids) == 0:
                missing.append(f"{episode_key}:{placeholder}")
        if len(missing) > 0:
            break
    if missing:
        joined = ", ".join(missing[:5])
        raise RuntimeError(
            "Actor-segmentation preprocessing requires placeholder->actor_ids metadata in scene_info.json, "
            f"but missing entries were found: {joined}. Collect data with an actor-segmentation task config "
            "that preserves target metadata."
        )


def require_actorseg_cameras(episode: dict, camera_names: list[str], segmentation_key: str = "actor_segmentation"):
    missing = []
    cameras = episode.get("cameras", {})
    for camera_name in camera_names:
        camera_info = cameras.get(camera_name)
        if camera_info is None:
            missing.append(f"{camera_name}:missing_camera")
            continue
        if camera_info.get(segmentation_key) is None:
            missing.append(f"{camera_name}:missing_{segmentation_key}")
        if camera_info.get("intrinsic_cv") is None:
            missing.append(f"{camera_name}:missing_intrinsic_cv")
        if camera_info.get("extrinsic_cv") is None:
            missing.append(f"{camera_name}:missing_extrinsic_cv")
    if missing:
        raise RuntimeError(
            "Actor-segmentation preprocessing requires the requested cameras to provide segmentation and camera "
            f"matrices, but missing entries were found: {', '.join(missing)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Process RoboTwin episodes into DP3 zarr using simulator actor segmentation projected onto scene point clouds."
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--object_placeholders", type=str, default="")
    parser.add_argument("--camera_names", type=str, default="head_camera,front_camera")
    parser.add_argument("--output_suffix", type=str, default="-objpc-actorseg")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--save_placeholder_point_clouds", action="store_true")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = int(args.expert_data_num)
    load_dir = str(raw_task_data_dir(task_name, task_config))
    save_dir = str(dp3_data_path(f"{task_name}-{task_config}-{num}{args.output_suffix}.zarr"))
    final_meta_path = str(dp3_data_path(f"{task_name}-{task_config}-{num}{args.output_suffix}_meta.json"))

    camera_names = parse_camera_list(args.camera_names)
    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    first_episode = load_hdf5_with_actorseg(
        os.path.join(load_dir, "data/episode0.hdf5"),
        camera_names=camera_names,
    )
    placeholders = parse_placeholder_list(args.object_placeholders)
    if len(placeholders) == 0:
        placeholders = infer_placeholder_order(scene_info, first_episode)
    if len(placeholders) == 0:
        raise RuntimeError("No object placeholders were found for actor-segmentation object-pointcloud preprocessing.")
    require_actor_id_targets(scene_info, placeholders)
    require_actorseg_cameras(first_episode, camera_names, segmentation_key="actor_segmentation")

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
            "segmentation_key": "actor_segmentation",
            "save_placeholder_point_clouds": bool(args.save_placeholder_point_clouds),
        }
    )
    episode_stats = list(meta.get("episodes", []))
    if len(episode_stats) > start_episode:
        episode_stats = episode_stats[:start_episode]
    while len(episode_stats) < start_episode:
        episode_stats.append({"episode": len(episode_stats), "recovered_without_stats": True})

    print(f"Resuming actor-segmentation preprocessing from episode {start_episode + 1} / {num}")

    for current_ep in range(start_episode, num):
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        episode = load_hdf5_with_actorseg(load_path, camera_names=camera_names)
        vector_all = episode["vector"]
        actor_id_map = parse_episode_actor_id_map(scene_info, current_ep, placeholders)
        local_modes = {placeholder: Counter() for placeholder in placeholders}

        episode_point_cloud_arrays = []
        episode_placeholder_point_cloud_arrays = {placeholder: [] for placeholder in placeholders}
        episode_state_arrays = []
        episode_joint_action_arrays = []

        for frame_idx in range(vector_all.shape[0]):
            per_placeholder_point_clouds = []
            for placeholder in placeholders:
                object_pc, extract_meta = extract_placeholder_point_cloud_actorseg(
                    episode,
                    frame_idx=frame_idx,
                    placeholder=placeholder,
                    actor_ids=actor_id_map.get(placeholder, []),
                    camera_names=camera_names,
                    target_num_points=int(args.target_num_points),
                    segmentation_key="actor_segmentation",
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

        target_record = {
            placeholder: {
                "actor_ids": actor_id_map.get(placeholder, []),
            }
            for placeholder in placeholders
        }
        episode_record = {
            "episode": current_ep,
            "placeholders": placeholders,
            "camera_names": camera_names,
            "segmentation_key": "actor_segmentation",
            "targets": target_record,
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
        meta["completed_episodes"] = current_ep + 1
        write_meta(final_meta_path, meta)

    print()


if __name__ == "__main__":
    main()
