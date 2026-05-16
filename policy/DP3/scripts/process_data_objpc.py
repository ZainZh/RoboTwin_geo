import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

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
from eef_action_utils import add_eef_preprocess_args, eef_arrays_for_episode, validate_eef_dataset_frame
from ndf_feature_utils import summarize_modes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Process RoboTwin episodes into DP3 zarr using only object point clouds.")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--object_placeholders", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="-objpc")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--cluster_eps", type=float, default=0.04)
    parser.add_argument("--min_cluster_points", type=int, default=24)
    parser.add_argument("--table_quantile", type=float, default=0.08)
    parser.add_argument("--table_margin", type=float, default=0.01)
    parser.add_argument("--save_placeholder_point_clouds", action="store_true")
    add_eef_preprocess_args(parser)
    args = parser.parse_args(argv)

    task_name = args.task_name
    task_config = args.task_config
    num = int(args.expert_data_num)
    load_dir = os.path.join("../../data", str(task_name), str(task_config))
    validate_eef_dataset_frame(
        action_mode=args.action_mode,
        eef_frame_mode=args.eef_frame_mode,
        load_dir=load_dir,
    )
    save_dir = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}.zarr"
    meta_path = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}_meta.json"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    first_episode = load_hdf5(os.path.join(load_dir, "data/episode0.hdf5"))
    placeholders = parse_placeholder_list(args.object_placeholders)
    if len(placeholders) == 0:
        placeholders = default_placeholder_order(scene_info, first_episode)
    if len(placeholders) == 0:
        raise RuntimeError("No object_pointcloud placeholders were found in the collected data.")

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    placeholder_point_cloud_arrays = {placeholder: [] for placeholder in placeholders}
    state_arrays = []
    joint_action_arrays = []
    episode_ends_arrays = []
    total_count = 0
    episode_stats = []

    for current_ep in range(num):
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        episode = load_hdf5(load_path)
        vector_all = episode["vector"]
        eef_arrays = eef_arrays_for_episode(args, episode)
        local_modes = {placeholder: Counter() for placeholder in placeholders}
        prev_centroids = {placeholder: None for placeholder in placeholders}
        asset_specs = {}

        for placeholder in placeholders:
            _, asset_spec = parse_target_extents(scene_info, current_ep, placeholder)
            asset_specs[placeholder] = asset_spec

        for frame_idx in range(vector_all.shape[0]):
            per_placeholder_point_clouds = []
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
                per_placeholder_point_clouds.append(object_pc)
                local_modes[placeholder][str(extract_meta.get("mode", "unknown"))] += 1
                centroid = valid_xyz_centroid(object_pc)
                if centroid is not None:
                    prev_centroids[placeholder] = centroid

                if frame_idx != vector_all.shape[0] - 1 and args.save_placeholder_point_clouds:
                    placeholder_point_cloud_arrays[placeholder].append(object_pc)

            merged_object_pc = merge_object_point_clouds(
                per_placeholder_point_clouds,
                target_num_points=int(args.target_num_points),
            )

            if frame_idx != vector_all.shape[0] - 1:
                point_cloud_arrays.append(merged_object_pc.astype(np.float32))
                state_arrays.append(eef_arrays[0][frame_idx] if eef_arrays is not None else vector_all[frame_idx])
            if frame_idx != 0:
                joint_action_arrays.append(eef_arrays[1][frame_idx - 1] if eef_arrays is not None else vector_all[frame_idx])

        total_count += vector_all.shape[0] - 1
        episode_ends_arrays.append(total_count)
        episode_stats.append(
            {
                "episode": current_ep,
                "placeholders": placeholders,
                "target_assets": asset_specs,
                "modes": {
                    placeholder: summarize_modes(local_modes[placeholder])
                    for placeholder in placeholders
                },
            }
        )

    print()
    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)
    point_cloud_arrays = np.asarray(point_cloud_arrays, dtype=np.float32)
    state_arrays = np.asarray(state_arrays, dtype=np.float32)
    joint_action_arrays = np.asarray(joint_action_arrays, dtype=np.float32)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud_arrays,
        chunks=(100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2]),
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=(100, joint_action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    if args.save_placeholder_point_clouds:
        for placeholder, arrays in placeholder_point_cloud_arrays.items():
            if len(arrays) == 0:
                continue
            key = f"object_point_cloud_{placeholder.strip('{}')}"
            zarr_data.create_dataset(
                key,
                data=np.asarray(arrays, dtype=np.float32),
                chunks=(100, args.target_num_points, 6),
                dtype="float32",
                overwrite=True,
                compressor=compressor,
            )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    meta = {
        "task_name": task_name,
        "task_config": task_config,
        "expert_data_num": num,
        "output_zarr": str(Path(save_dir).resolve()),
        "object_placeholders": placeholders,
        "episodes": episode_stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
