import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from actorseg_pointcloud_utils import (
    extract_placeholder_point_cloud_actorseg,
    load_hdf5_with_actorseg,
    parse_camera_list,
    parse_episode_actor_id_map,
)
from incremental_objpc_zarr import append_episode_to_buffer, open_or_reset_replay_buffer
from ndf_feature_utils import load_scene_info, summarize_modes
from object_pointcloud_utils import parse_placeholder_list
from pointwise_context_utils import build_context_point_cloud, resolve_context_placeholders
from process_data_objpc_actorseg import (
    infer_placeholder_order,
    load_or_init_meta,
    require_actor_id_targets,
    require_actorseg_cameras,
    write_meta,
)
from process_data_semantic_pointwise import parse_placeholder_model_args, placeholder_pointcloud_key
from semantic_feature_utils import compute_semantic_pointwise_cloud, load_semantic_model


def main():
    parser = argparse.ArgumentParser(
        description="Process RoboTwin episodes into DP3 zarr using actor segmentation for raw object point clouds and point-wise semantic embeddings."
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--object_placeholders", type=str, default="")
    parser.add_argument("--camera_names", type=str, default="head_camera,front_camera")
    parser.add_argument("--semantic_model", action="append", default=[], help="Repeated PLACEHOLDER=CHECKPOINT mapping.")
    parser.add_argument("--semantic_device", type=str, default="cuda:0")
    parser.add_argument("--semantic_num_points", type=int, default=128)
    parser.add_argument(
        "--semantic_input_color_mode",
        choices=["debug_placeholder", "stored_scaled", "stored"],
        default="debug_placeholder",
    )
    parser.add_argument("--semantic_forward_mode", choices=["reference", "dp3"], default="reference")
    parser.add_argument("--output_suffix", type=str, default="-objpc-actorseg-semantic-pointwise-hybrid")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--save_placeholder_point_clouds", action="store_true")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = int(args.expert_data_num)
    load_dir = os.path.join("../../data", str(task_name), str(task_config))
    save_dir = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}.zarr"
    final_meta_path = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}_meta.json"

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
        raise RuntimeError("No object placeholders were found for actorseg semantic hybrid preprocessing.")
    require_actor_id_targets(scene_info, placeholders)
    require_actorseg_cameras(first_episode, camera_names, segmentation_key="actor_segmentation")

    model_paths = parse_placeholder_model_args(args.semantic_model)
    if len(model_paths) == 0:
        raise RuntimeError(
            "Actorseg semantic hybrid preprocessing requires at least one --semantic_model PLACEHOLDER=CHECKPOINT."
        )

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
            f"Actorseg semantic hybrid preprocessing requires consistent semantic embedding dimensions, got {semantic_feat_dims!r}."
        )
    semantic_feat_dim = int(unique_semantic_dims[0])

    feature_placeholders = [placeholder for placeholder in placeholders if placeholder in model_by_placeholder]
    context_placeholders = resolve_context_placeholders(
        placeholders,
        feature_placeholders,
        keep_feature_placeholders_in_context=True,
    )
    if len(feature_placeholders) == 0:
        raise RuntimeError("None of the requested object placeholders has a semantic model configured.")

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
            "context_placeholders": context_placeholders,
            "feature_placeholders": feature_placeholders,
            "camera_names": camera_names,
            "segmentation_key": "actor_segmentation",
            "semantic_models": model_paths,
            "semantic_num_points": int(args.semantic_num_points),
            "semantic_feat_dim": semantic_feat_dim,
            "semantic_feat_dims": semantic_feat_dims,
            "semantic_input_color_mode": str(args.semantic_input_color_mode),
            "semantic_forward_mode": str(args.semantic_forward_mode),
            "keep_feature_placeholders_in_context": True,
            "save_placeholder_point_clouds": bool(args.save_placeholder_point_clouds),
        }
    )
    episode_stats = list(meta.get("episodes", []))
    if len(episode_stats) > start_episode:
        episode_stats = episode_stats[:start_episode]
    while len(episode_stats) < start_episode:
        episode_stats.append({"episode": len(episode_stats), "recovered_without_stats": True})

    print(f"Resuming actorseg semantic hybrid preprocessing from episode {start_episode + 1} / {num}")

    for current_ep in range(start_episode, num):
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        episode = load_hdf5_with_actorseg(load_path, camera_names=camera_names)
        vector_all = episode["vector"]
        actor_id_map = parse_episode_actor_id_map(scene_info, current_ep, placeholders)
        local_modes = {placeholder: Counter() for placeholder in placeholders}

        episode_point_cloud_arrays = []
        episode_feature_arrays = {placeholder: [] for placeholder in feature_placeholders}
        episode_placeholder_point_cloud_arrays = {placeholder: [] for placeholder in placeholders}
        episode_state_arrays = []
        episode_joint_action_arrays = []

        for frame_idx in range(vector_all.shape[0]):
            per_placeholder_point_clouds = {}
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
                per_placeholder_point_clouds[placeholder] = object_pc
                local_modes[placeholder][str(extract_meta.get("mode", "unknown"))] += 1
                if frame_idx != vector_all.shape[0] - 1 and args.save_placeholder_point_clouds:
                    episode_placeholder_point_cloud_arrays[placeholder].append(object_pc.astype(np.float32))

            if frame_idx != vector_all.shape[0] - 1:
                context_point_cloud, _ = build_context_point_cloud(
                    per_placeholder_point_clouds,
                    placeholders=placeholders,
                    feature_placeholders=feature_placeholders,
                    target_num_points=int(args.target_num_points),
                    keep_feature_placeholders_in_context=True,
                )
                episode_point_cloud_arrays.append(context_point_cloud.astype(np.float32))
                episode_state_arrays.append(vector_all[frame_idx])
                for placeholder in feature_placeholders:
                    episode_feature_arrays[placeholder].append(
                        compute_semantic_pointwise_cloud(
                            artifacts=model_by_placeholder[placeholder],
                            object_point_cloud=per_placeholder_point_clouds[placeholder],
                            target_num_points=int(args.semantic_num_points),
                            placeholder=placeholder,
                            semantic_input_color_mode=str(args.semantic_input_color_mode),
                            semantic_forward_mode=str(args.semantic_forward_mode),
                        ).astype(np.float32)
                    )
            if frame_idx != 0:
                episode_joint_action_arrays.append(vector_all[frame_idx])

        episode_data = {
            "point_cloud": np.asarray(episode_point_cloud_arrays, dtype=np.float32),
            "state": np.asarray(episode_state_arrays, dtype=np.float32),
            "action": np.asarray(episode_joint_action_arrays, dtype=np.float32),
        }
        for placeholder in feature_placeholders:
            episode_data[placeholder_pointcloud_key(placeholder)] = np.asarray(
                episode_feature_arrays[placeholder], dtype=np.float32
            )
        if args.save_placeholder_point_clouds:
            for placeholder, arrays in episode_placeholder_point_cloud_arrays.items():
                episode_data[f"object_point_cloud_{placeholder.strip('{}')}"] = np.asarray(arrays, dtype=np.float32)

        append_episode_to_buffer(replay_buffer, episode_data)

        target_record = {
            placeholder: {"actor_ids": actor_id_map.get(placeholder, [])}
            for placeholder in placeholders
        }
        episode_record = {
            "episode": current_ep,
            "placeholders": placeholders,
            "context_placeholders": context_placeholders,
            "feature_placeholders": feature_placeholders,
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
