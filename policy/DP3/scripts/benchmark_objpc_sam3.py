import argparse
import os
import time

import numpy as np

from object_pointcloud_utils import parse_placeholder_list
from sam3_pointcloud_utils import (
    SAM3ProjectiveTracker,
    Sam3TrackingState,
    build_placeholder_prompt_map,
    extract_placeholder_point_cloud_sam3,
    load_hdf5_with_cameras,
    load_scene_info,
    parse_camera_list,
    parse_prompt_map,
    parse_target_extents,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark SAM3 projected object point-cloud extraction.")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode_idx", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--object_placeholders", type=str, default="{A},{B}")
    parser.add_argument("--camera_names", type=str, default="head_camera,front_camera")
    parser.add_argument("--sam3_model", type=str, default="/home/zheng/Datasets/sam3/sam3.pt")
    parser.add_argument("--sam3_prompt_map", type=str, default="")
    parser.add_argument("--sam3_conf", type=float, default=0.50)
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--text_refresh_every", type=int, default=15)
    parser.add_argument("--min_mask_points", type=int, default=16)
    args = parser.parse_args()

    load_dir = os.path.join("../../data", str(args.task_name), str(args.task_config))
    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    camera_names = parse_camera_list(args.camera_names)
    episode = load_hdf5_with_cameras(
        os.path.join(load_dir, f"data/episode{int(args.episode_idx)}.hdf5"),
        camera_names=camera_names,
    )

    placeholders = parse_placeholder_list(args.object_placeholders)
    prompt_map = build_placeholder_prompt_map(
        scene_info=scene_info,
        episode_idx=int(args.episode_idx),
        placeholders=placeholders,
        prompt_overrides=parse_prompt_map(args.sam3_prompt_map),
    )

    tracker = SAM3ProjectiveTracker(
        model_path=args.sam3_model,
        conf=float(args.sam3_conf),
        verbose=False,
    )
    tracking_state = {
        placeholder: {
            camera_name: Sam3TrackingState()
            for camera_name in camera_names
        }
        for placeholder in placeholders
    }

    frame_count = min(int(args.num_frames), int(episode["vector"].shape[0] - 1))
    if frame_count <= 0:
        raise RuntimeError("No frames available for benchmark.")

    frame_times = []
    placeholder_times = {placeholder: [] for placeholder in placeholders}

    for frame_idx in range(frame_count):
        t0 = time.perf_counter()
        for placeholder in placeholders:
            ph_t0 = time.perf_counter()
            target_extents, _ = parse_target_extents(scene_info, int(args.episode_idx), placeholder)
            _cloud, _meta = extract_placeholder_point_cloud_sam3(
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
            placeholder_times[placeholder].append(time.perf_counter() - ph_t0)
        frame_times.append(time.perf_counter() - t0)

    frame_times = np.asarray(frame_times, dtype=np.float64)
    print(f"task={args.task_name}, task_config={args.task_config}, episode={args.episode_idx}")
    print(f"camera_names={camera_names}, placeholders={placeholders}, prompts={prompt_map}")
    print(f"frames_benchmarked={frame_count}")
    print(f"frame_total_ms: first={frame_times[0]*1000:.2f}, mean={frame_times.mean()*1000:.2f}, p95={np.quantile(frame_times, 0.95)*1000:.2f}")
    if frame_count > 1:
        steady = frame_times[1:]
        print(f"frame_total_ms_steady: mean={steady.mean()*1000:.2f}, p95={np.quantile(steady, 0.95)*1000:.2f}")
    for placeholder in placeholders:
        values = np.asarray(placeholder_times[placeholder], dtype=np.float64)
        print(
            f"{placeholder}_ms: first={values[0]*1000:.2f}, "
            f"mean={values.mean()*1000:.2f}, p95={np.quantile(values, 0.95)*1000:.2f}"
        )


if __name__ == "__main__":
    main()
