import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr


DP3_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "DP3" / "scripts"
if str(DP3_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_SCRIPTS_ROOT))

from eef_action_utils import add_eef_preprocess_args, eef_arrays_for_episode  # noqa: E402


RAW_TO_ZARR_CAMERA_KEYS = {
    "global": "head_camera",
    "head": "head_camera",
    "head_camera": "head_camera",
    "left": "left_camera",
    "left_camera": "left_camera",
    "right": "right_camera",
    "right_camera": "right_camera",
}


def read_json(path):
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def default_meta_path(task_name, task_config):
    return Path("../../data") / str(task_name) / str(task_config) / "real_zed_sam2_objpc_meta.json"


def load_rgb(raw_episode_dir, rel_path, resize_hw=None):
    with np.load(raw_episode_dir / rel_path, allow_pickle=False) as data:
        image = np.asarray(data["rgb"], dtype=np.uint8)
    if resize_hw is not None:
        height, width = resize_hw
        image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return image


def load_robot_episode_vectors(hdf5_path):
    with h5py.File(hdf5_path, "r") as root:
        vector = np.asarray(root["/joint_action/vector"][()], dtype=np.float32)
        control = np.asarray(root["/joint_action/control"][()], dtype=np.float32) if "/joint_action/control" in root else None
        eef_pose_base = np.asarray(root["/eef_action/base_pose6"][()], dtype=np.float32) if "/eef_action/base_pose6" in root else None
    return vector, control, eef_pose_base


def parse_resize(value):
    if value is None or str(value).strip() == "":
        return None
    parts = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--resize_hw must be formatted as H,W, for example 360,640")
    return int(parts[0]), int(parts[1])


def parse_camera_labels(value):
    if value is None:
        return ["global"]
    if isinstance(value, (list, tuple)):
        labels = [str(item).strip() for item in value if str(item).strip()]
    else:
        labels = [item.strip() for item in str(value).split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one camera label is required.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate camera labels are not supported: {labels}")
    return labels


def camera_label_setting(camera_labels):
    return "_".join(str(label) for label in camera_labels)


def zarr_camera_keys_for_labels(camera_labels):
    if len(camera_labels) == 1:
        return {str(camera_labels[0]): "head_camera"}

    zarr_keys = []
    for idx, label in enumerate(camera_labels):
        zarr_key = RAW_TO_ZARR_CAMERA_KEYS.get(str(label), None)
        if zarr_key is None:
            if idx == 0:
                zarr_key = "head_camera"
            elif idx == 1:
                zarr_key = "left_camera"
            elif idx == 2:
                zarr_key = "right_camera"
            else:
                raise ValueError(
                    f"Cannot map camera label {label!r} to a DP zarr key. "
                    "Use at most three cameras or one of: global/head, left, right."
                )
        zarr_keys.append(zarr_key)
    if len(set(zarr_keys)) != len(zarr_keys):
        raise ValueError(f"Camera labels map to duplicate DP zarr keys: {dict(zip(camera_labels, zarr_keys))}")
    return dict(zip(camera_labels, zarr_keys))


def build_dp_real_zed_zarr(
    *,
    task_name,
    task_config,
    expert_data_num,
    camera_label=None,
    camera_labels=None,
    meta_path,
    output_zarr,
    resize_hw=None,
    action_args=None,
):
    if camera_labels is None:
        camera_labels = [camera_label or "global"]
    camera_labels = parse_camera_labels(camera_labels)
    camera_key_map = zarr_camera_keys_for_labels(camera_labels)

    meta = read_json(meta_path)
    processed = meta.get("processed", [])
    if len(processed) < int(expert_data_num):
        raise ValueError(f"Requested {expert_data_num} episodes, but meta only lists {len(processed)} processed episodes.")

    camera_arrays = {zarr_key: [] for zarr_key in camera_key_map.values()}
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0

    for episode_idx, item in enumerate(processed[: int(expert_data_num)]):
        raw_episode_dir = Path(item["raw_episode_dir"]).expanduser().resolve()
        hdf5_path = Path(item["hdf5_path"]).expanduser().resolve()
        manifest = read_json(raw_episode_dir / "manifest.json")
        frames = manifest.get("frames", [])
        if not frames:
            raise ValueError(f"No frames in raw manifest: {raw_episode_dir / 'manifest.json'}")
        joint_vector, control_vector, eef_pose_base = load_robot_episode_vectors(hdf5_path)
        frame_count = min(len(frames), joint_vector.shape[0])
        if frame_count < 2:
            raise ValueError(f"Episode {episode_idx} has fewer than 2 aligned frames.")
        eef_arrays = None
        if action_args is not None and str(getattr(action_args, "action_mode", "joint")) == "eef_absolute6d":
            episode = {"vector": joint_vector[:frame_count]}
            if control_vector is not None:
                episode["control"] = control_vector[:frame_count]
            if eef_pose_base is not None:
                episode["eef_pose_base"] = eef_pose_base[:frame_count]
            eef_arrays = eef_arrays_for_episode(action_args, episode)

        print(f"processing real-zed episode: {episode_idx + 1} / {expert_data_num}", end="\r")
        for frame_idx in range(frame_count):
            frame = frames[frame_idx]
            cameras = frame.get("cameras", {})
            missing_cameras = [label for label in camera_labels if str(label) not in cameras]
            if missing_cameras:
                raise KeyError(f"Cameras {missing_cameras!r} not found in raw episode {raw_episode_dir}")
            if frame_idx != frame_count - 1:
                for camera_label_i, zarr_key in camera_key_map.items():
                    rgb = load_rgb(raw_episode_dir, str(cameras[str(camera_label_i)]), resize_hw=resize_hw)
                    camera_arrays[zarr_key].append(np.moveaxis(rgb, -1, 0))
                state_arrays.append(eef_arrays[0][frame_idx] if eef_arrays is not None else joint_vector[frame_idx])
            if frame_idx != 0:
                action_arrays.append(eef_arrays[1][frame_idx - 1] if eef_arrays is not None else joint_vector[frame_idx])

        total_count += frame_count - 1
        episode_ends_arrays.append(total_count)

    print()
    camera_lengths = {key: len(value) for key, value in camera_arrays.items()}
    if any(length != len(action_arrays) for length in camera_lengths.values()) or len(state_arrays) != len(action_arrays):
        raise RuntimeError(
            "Mismatched DP arrays: "
            f"images={camera_lengths}, states={len(state_arrays)}, actions={len(action_arrays)}"
        )

    output_zarr = Path(output_zarr).expanduser()
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    camera_arrays = {key: np.asarray(value, dtype=np.uint8) for key, value in camera_arrays.items()}
    state_arrays = np.asarray(state_arrays, dtype=np.float32)
    action_arrays = np.asarray(action_arrays, dtype=np.float32)
    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)

    zarr_root = zarr.group(str(output_zarr))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    zarr_root.attrs["source"] = {
        "task_name": str(task_name),
        "task_config": str(task_config),
        "camera_label": str(camera_labels[0]),
        "camera_labels": [str(item) for item in camera_labels],
        "camera_key_map": {str(raw): str(mapped) for raw, mapped in camera_key_map.items()},
        "meta_path": str(Path(meta_path).expanduser().name),
        "portable": True,
        "expert_data_num": int(expert_data_num),
        "action_mode": str(getattr(action_args, "action_mode", "joint")) if action_args is not None else "joint",
        "eef_frame_mode": str(getattr(action_args, "eef_frame_mode", "")) if action_args is not None else "",
    }

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    for zarr_key, camera_data in camera_arrays.items():
        zarr_data.create_dataset(
            zarr_key,
            data=camera_data,
            chunks=(min(100, max(1, camera_data.shape[0])), *camera_data.shape[1:]),
            overwrite=True,
            compressor=compressor,
        )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(min(100, max(1, state_arrays.shape[0])), state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=(min(100, max(1, action_arrays.shape[0])), action_arrays.shape[1]),
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
    print(f"wrote {output_zarr}")


def main():
    parser = argparse.ArgumentParser(description="Convert real-ZED postprocess metadata/raw frames to policy/DP image baseline zarr.")
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--camera_label", default="global")
    parser.add_argument("--camera_labels", default="")
    parser.add_argument("--meta_path", default="")
    parser.add_argument("--output_zarr", default="")
    parser.add_argument("--resize_hw", default="")
    add_eef_preprocess_args(parser)
    args = parser.parse_args()

    camera_labels = parse_camera_labels(args.camera_labels if args.camera_labels else args.camera_label)
    train_setting = f"{args.task_config}-dp-{camera_label_setting(camera_labels)}"
    meta_path = Path(args.meta_path).expanduser() if args.meta_path else default_meta_path(args.task_name, args.task_config)
    output_zarr = (
        Path(args.output_zarr).expanduser()
        if args.output_zarr
        else Path("./data") / f"{args.task_name}-{train_setting}-{args.expert_data_num}.zarr"
    )
    build_dp_real_zed_zarr(
        task_name=args.task_name,
        task_config=args.task_config,
        expert_data_num=args.expert_data_num,
        camera_labels=camera_labels,
        meta_path=meta_path,
        output_zarr=output_zarr,
        resize_hw=parse_resize(args.resize_hw),
        action_args=args,
    )


if __name__ == "__main__":
    main()
