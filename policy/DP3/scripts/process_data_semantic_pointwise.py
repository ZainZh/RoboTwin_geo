import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import zarr

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
from pointwise_context_utils import build_context_point_cloud, resolve_context_placeholders
from semantic_feature_utils import compute_semantic_pointwise_cloud, load_semantic_model


def parse_placeholder_model_args(values):
    result = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Expected PLACEHOLDER=CHECKPOINT, got {item!r}")
        placeholder, checkpoint = item.split("=", 1)
        placeholder = placeholder.strip()
        checkpoint = checkpoint.strip()
        if not placeholder:
            raise ValueError(f"Missing placeholder in {item!r}")
        if not checkpoint:
            continue
        result[placeholder] = checkpoint
    return result


def placeholder_pointcloud_key(placeholder: str) -> str:
    return f"semantic_point_cloud_{placeholder.strip('{}')}"


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
            object_pc_info = episode_info.get("object_pointcloud", {})
            targets = object_pc_info.get("targets", {}) if isinstance(object_pc_info, dict) else {}
            if isinstance(targets, dict):
                for placeholder in targets.keys():
                    append_placeholder(placeholder)

    object_pointcloud = first_episode.get("object_pointcloud", {})
    if isinstance(object_pointcloud, dict):
        for placeholder in sorted(object_pointcloud.keys()):
            append_placeholder(placeholder)

    return placeholders


def build_parser():
    parser = argparse.ArgumentParser(
        description="Process RoboTwin episodes into DP3 zarr using context object point clouds and point-wise semantic embeddings."
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--object_placeholders", type=str, default="")
    parser.add_argument("--semantic_model", action="append", default=[], help="Repeated PLACEHOLDER=CHECKPOINT mapping.")
    parser.add_argument("--semantic_device", type=str, default="cuda:0")
    parser.add_argument("--semantic_num_points", type=int, default=128)
    parser.add_argument("--output_suffix", type=str, default="-objpc-semantic-pointwise")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--cluster_eps", type=float, default=0.04)
    parser.add_argument("--min_cluster_points", type=int, default=24)
    parser.add_argument("--table_quantile", type=float, default=0.08)
    parser.add_argument("--table_margin", type=float, default=0.01)
    parser.add_argument("--save_placeholder_point_clouds", action="store_true")
    parser.add_argument("--keep_feature_placeholders_in_context", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    task_name = args.task_name
    task_config = args.task_config
    num = int(args.expert_data_num)
    load_dir = os.path.join("../../data", str(task_name), str(task_config))
    save_dir = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}.zarr"
    meta_path = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}_meta.json"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    first_episode = load_hdf5(os.path.join(load_dir, "data/episode0.hdf5"))
    placeholders = parse_placeholder_list(args.object_placeholders)
    if len(placeholders) == 0:
        placeholders = infer_placeholder_order(scene_info, first_episode)
    if len(placeholders) == 0:
        raise RuntimeError("No object_pointcloud placeholders were found in the collected data.")

    model_paths = parse_placeholder_model_args(args.semantic_model)
    if len(model_paths) == 0:
        raise RuntimeError("Semantic pointwise preprocessing requires at least one --semantic_model PLACEHOLDER=CHECKPOINT.")

    device = torch.device(args.semantic_device if torch.cuda.is_available() else "cpu")
    model_by_placeholder = {}
    for placeholder, checkpoint in model_paths.items():
        model_by_placeholder[placeholder] = load_semantic_model(
            checkpoint=checkpoint,
            device=device,
        )
    semantic_feat_dims = {
        placeholder: int(artifacts["sem_embedding_dim"])
        for placeholder, artifacts in model_by_placeholder.items()
    }
    unique_semantic_dims = sorted(set(semantic_feat_dims.values()))
    if len(unique_semantic_dims) != 1:
        raise RuntimeError(
            f"Semantic pointwise preprocessing requires consistent semantic embedding dimensions, got {semantic_feat_dims!r}."
        )
    semantic_feat_dim = int(unique_semantic_dims[0])

    feature_placeholders = [placeholder for placeholder in placeholders if placeholder in model_by_placeholder]
    context_placeholders = resolve_context_placeholders(
        placeholders,
        feature_placeholders,
        keep_feature_placeholders_in_context=bool(args.keep_feature_placeholders_in_context),
    )
    if len(feature_placeholders) == 0:
        raise RuntimeError("None of the requested object placeholders has a semantic model configured.")

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    semantic_arrays = {placeholder: [] for placeholder in feature_placeholders}
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
        prev_centroids = {placeholder: None for placeholder in placeholders}
        local_modes = {placeholder: Counter() for placeholder in placeholders}
        asset_specs = {}
        has_exact = {}

        for placeholder in placeholders:
            _, asset_spec = parse_target_extents(scene_info, current_ep, placeholder)
            asset_specs[placeholder] = asset_spec
            has_exact[placeholder] = placeholder in episode["object_pointcloud"]

        for frame_idx in range(vector_all.shape[0]):
            per_placeholder_point_clouds = {}
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
                per_placeholder_point_clouds[placeholder] = object_pc
                local_modes[placeholder][str(extract_meta.get("mode", "unknown"))] += 1
                centroid = valid_xyz_centroid(object_pc)
                if centroid is not None:
                    prev_centroids[placeholder] = centroid

            if frame_idx != vector_all.shape[0] - 1:
                context_point_cloud, _ = build_context_point_cloud(
                    per_placeholder_point_clouds,
                    placeholders=placeholders,
                    feature_placeholders=feature_placeholders,
                    target_num_points=int(args.target_num_points),
                    keep_feature_placeholders_in_context=bool(args.keep_feature_placeholders_in_context),
                )
                point_cloud_arrays.append(context_point_cloud.astype(np.float32))
                state_arrays.append(vector_all[frame_idx])

                for placeholder in placeholders:
                    object_pc = per_placeholder_point_clouds[placeholder]
                    if args.save_placeholder_point_clouds:
                        placeholder_point_cloud_arrays[placeholder].append(object_pc.astype(np.float32))

                for placeholder in feature_placeholders:
                    semantic_arrays[placeholder].append(
                        compute_semantic_pointwise_cloud(
                            artifacts=model_by_placeholder[placeholder],
                            object_point_cloud=per_placeholder_point_clouds[placeholder],
                            target_num_points=int(args.semantic_num_points),
                        ).astype(np.float32)
                    )

            if frame_idx != 0:
                joint_action_arrays.append(vector_all[frame_idx])

        total_count += vector_all.shape[0] - 1
        episode_ends_arrays.append(total_count)
        episode_stats.append(
            {
                "episode": current_ep,
                "placeholders": placeholders,
                "context_placeholders": context_placeholders,
                "feature_placeholders": feature_placeholders,
                "target_assets": asset_specs,
                "has_exact_object_pointcloud": has_exact,
                "has_semantic_model": {
                    placeholder: placeholder in model_by_placeholder
                    for placeholder in placeholders
                },
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

    for placeholder in feature_placeholders:
        point_cloud_key = placeholder_pointcloud_key(placeholder)
        point_cloud_arr = np.asarray(semantic_arrays[placeholder], dtype=np.float32)
        zarr_data.create_dataset(
            point_cloud_key,
            data=point_cloud_arr,
            chunks=(100, point_cloud_arr.shape[1], point_cloud_arr.shape[2]),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )

    if args.save_placeholder_point_clouds:
        for placeholder in placeholders:
            point_cloud_key = f"object_point_cloud_{placeholder.strip('{}')}"
            zarr_data.create_dataset(
                point_cloud_key,
                data=np.asarray(placeholder_point_cloud_arrays[placeholder], dtype=np.float32),
                chunks=(100, int(args.target_num_points), 6),
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
        "context_placeholders": context_placeholders,
        "feature_placeholders": feature_placeholders,
        "semantic_models": model_paths,
        "semantic_num_points": int(args.semantic_num_points),
        "semantic_feat_dim": semantic_feat_dim,
        "semantic_feat_dims": semantic_feat_dims,
        "keep_feature_placeholders_in_context": bool(args.keep_feature_placeholders_in_context),
        "episodes": episode_stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
