#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from script.real_zed_collection.real_zed_utils import (
    depth_rgb_to_point_cloud,
    deterministic_resample,
    ensure_dir,
    load_three_zed_calibration,
    merge_point_clouds,
    read_json,
    transform_point_cloud,
    write_json,
)
from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds, apply_workspace_crop_to_camera_frame


def parse_object_prompts(text: str) -> dict[str, str]:
    if not text:
        return {}
    result: dict[str, str] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid object prompt item {item!r}; expected '{{A}}:mug'.")
        key, value = item.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _load_mask(mask_root: Path, placeholder: str, camera_label: str, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
    clean = placeholder.strip()
    candidates = [
        mask_root / clean / camera_label / f"mask_{frame_index:06d}.png",
        mask_root / clean.strip("{}") / camera_label / f"mask_{frame_index:06d}.png",
        mask_root / camera_label / clean / f"mask_{frame_index:06d}.png",
        mask_root / camera_label / clean.strip("{}") / f"mask_{frame_index:06d}.png",
    ]
    for path in candidates:
        if path.exists():
            if cv2 is None:
                from imageio.v2 import imread

                raw = imread(path)
            else:
                raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"Failed to load mask: {path}")
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            mask = np.asarray(raw) > 0
            if mask.shape != shape:
                if cv2 is None:
                    from imageio.v2 import imread

                    mask = np.asarray(imread(path)) > 0
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                else:
                    mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            return mask
    return np.zeros(shape, dtype=bool)


def _load_frame_npz(raw_episode_dir: Path, rel_path: str) -> dict[str, np.ndarray]:
    path = raw_episode_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Raw frame file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _resolve_calibration_path(raw_episode_dir: Path, manifest: dict, override: str | Path, frame_mode: str = "reference_camera") -> Path:
    if str(override).strip():
        calib_path = Path(override).expanduser().resolve()
    else:
        if str(frame_mode) == "workspace":
            snapshot_rel = str(
                manifest.get("workspace_calibration_snapshot_path", "")
                or manifest.get("calibration_snapshot_path", "")
            ).strip()
            source_path = str(
                manifest.get("workspace_calibration_path", "")
                or manifest.get("calibration_path", "")
            ).strip()
        else:
            snapshot_rel = str(manifest.get("calibration_snapshot_path", "")).strip()
            source_path = str(manifest.get("calibration_path", "")).strip()
        if snapshot_rel:
            calib_path = (raw_episode_dir / snapshot_rel).resolve()
        else:
            if not source_path:
                raise ValueError("Missing calibration path. Pass --calibration_path or include it in manifest.json.")
            calib_path = Path(source_path).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {calib_path}")
    return calib_path


def _camera_frame_to_world_pc(
    *,
    camera_frame: Mapping[str, np.ndarray],
    camera_matrix: np.ndarray,
    t_world_from_cam: np.ndarray,
    mask: np.ndarray | None,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    rgb = np.asarray(camera_frame["rgb"])
    if "depth_m" in camera_frame:
        depth_m = np.asarray(camera_frame["depth_m"], dtype=np.float32)
    elif "depth_mm" in camera_frame:
        depth_m = np.asarray(camera_frame["depth_mm"], dtype=np.float32) / 1000.0
    else:
        raise KeyError("Camera frame must contain depth_m or depth_mm.")
    if rgb.shape[:2] != depth_m.shape:
        if cv2 is None:
            raise ValueError(f"rgb/depth shape mismatch without cv2 available: rgb={rgb.shape[:2]}, depth={depth_m.shape}")
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_LINEAR)
    pc_cam = depth_rgb_to_point_cloud(
        depth_m=depth_m,
        rgb=rgb,
        camera_matrix=camera_matrix,
        mask=mask,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    return transform_point_cloud(pc_cam, t_world_from_cam)


def _crop_point_cloud_by_bounds(point_cloud: np.ndarray, bounds: WorkspaceBounds | None) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if bounds is None or pc.size == 0:
        return pc
    mask = (
        (pc[:, 0] >= float(bounds.x_min))
        & (pc[:, 0] <= float(bounds.x_max))
        & (pc[:, 1] >= float(bounds.y_min))
        & (pc[:, 1] <= float(bounds.y_max))
        & (pc[:, 2] >= float(bounds.z_min))
        & (pc[:, 2] <= float(bounds.z_max))
    )
    return pc[mask]


def _rgb_to_depth_shape(rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    if rgb_arr.shape[:2] == depth_shape:
        return rgb_arr
    if cv2 is None:
        raise ValueError(f"rgb/depth shape mismatch without cv2 available: rgb={rgb_arr.shape[:2]}, depth={depth_shape}")
    return cv2.resize(rgb_arr, (int(depth_shape[1]), int(depth_shape[0])), interpolation=cv2.INTER_LINEAR)


def postprocess_episode(
    *,
    raw_episode_dir: str | Path,
    output_dir: str | Path,
    episode_index: int,
    calibration_path: str | Path,
    camera_labels: list[str],
    object_prompts: Mapping[str, str],
    mask_root: str | Path | None,
    scene_point_num: int = 1024,
    object_point_num: int = 1024,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
    frame_mode: str = "reference_camera",
    workspace_crop_bounds: WorkspaceBounds | None = None,
    workspace_crop_margin_px: int = 0,
) -> Path:
    raw_episode_dir = Path(raw_episode_dir).expanduser().resolve()
    output_dir = ensure_dir(output_dir)
    manifest = read_json(raw_episode_dir / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Raw episode manifest has no frames: {raw_episode_dir / 'manifest.json'}")

    calib_path = _resolve_calibration_path(raw_episode_dir, manifest, calibration_path, frame_mode=frame_mode)
    calib = load_three_zed_calibration(calib_path, frame_mode=frame_mode)
    labels = camera_labels or list(manifest.get("camera_labels", [])) or list(calib.keys())
    labels = [str(label) for label in labels]
    missing = [label for label in labels if label not in calib]
    if missing:
        raise ValueError(f"Camera labels missing from calibration: {missing}")

    mask_root_path = None if mask_root is None or str(mask_root) == "" else Path(mask_root).expanduser().resolve()
    placeholders = list(object_prompts.keys())

    joint_vectors = []
    scene_point_clouds = []
    object_point_clouds = {placeholder: [] for placeholder in placeholders}
    intrinsic_by_camera: dict[str, np.ndarray] = {}
    observation_by_camera: dict[str, dict[str, list[np.ndarray]]] = {
        label: {"rgb": [], "depth": []}
        for label in labels
    }

    for frame in frames:
        frame_index = int(frame.get("frame_index", len(joint_vectors)))
        robot = _load_frame_npz(raw_episode_dir, str(frame["robot"]))
        joint_vector = np.asarray(robot.get("joint_vector", robot.get("joint_positions")), dtype=np.float32)
        if joint_vector.shape != (14,):
            raise ValueError(f"Expected robot joint vector shape (14,), got {joint_vector.shape}")
        joint_vectors.append(joint_vector)

        scene_chunks = []
        object_chunks_by_placeholder = {placeholder: [] for placeholder in placeholders}
        cameras = frame.get("cameras", {})
        if not isinstance(cameras, dict):
            raise ValueError(f"Frame cameras must be a dict at frame {frame_index}")

        for label in labels:
            camera_frame = _load_frame_npz(raw_episode_dir, str(cameras[label]))
            rgb = np.asarray(camera_frame["rgb"])
            depth_m = (
                np.asarray(camera_frame["depth_m"], dtype=np.float32)
                if "depth_m" in camera_frame
                else np.asarray(camera_frame["depth_mm"], dtype=np.float32) / 1000.0
            )
            camera_matrix = (
                np.asarray(camera_frame["camera_matrix"], dtype=np.float32).reshape(3, 3)
                if "camera_matrix" in camera_frame
                else calib[label].camera_matrix.astype(np.float32)
            )
            intrinsic_by_camera[label] = camera_matrix.astype(np.float32)
            obs_rgb = rgb.astype(np.uint8)
            obs_depth = depth_m.astype(np.float32)
            obs_camera_matrix = camera_matrix.astype(np.float32)
            if workspace_crop_bounds is not None:
                rgb_for_crop = _rgb_to_depth_shape(obs_rgb, obs_depth.shape)
                cropped_obs = apply_workspace_crop_to_camera_frame(
                    rgb=rgb_for_crop,
                    depth_m=obs_depth,
                    camera_matrix=obs_camera_matrix,
                    t_workspace_from_cam=calib[label].t_world_from_cam,
                    bounds=workspace_crop_bounds,
                    margin_px=int(workspace_crop_margin_px),
                )
                obs_rgb = np.asarray(cropped_obs["rgb"], dtype=np.uint8)
                obs_depth = np.asarray(cropped_obs["depth_m"], dtype=np.float32)
                obs_camera_matrix = np.asarray(cropped_obs["camera_matrix"], dtype=np.float32)
            intrinsic_by_camera[label] = obs_camera_matrix.astype(np.float32)
            observation_by_camera[label]["rgb"].append(obs_rgb.astype(np.uint8))
            observation_by_camera[label]["depth"].append(obs_depth.astype(np.float32))

            scene_chunks.append(
                _camera_frame_to_world_pc(
                    camera_frame=camera_frame,
                    camera_matrix=camera_matrix,
                    t_world_from_cam=calib[label].t_world_from_cam,
                    mask=None,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
            )

            for placeholder in placeholders:
                if mask_root_path is None:
                    mask = np.ones(depth_m.shape, dtype=bool)
                else:
                    mask = _load_mask(mask_root_path, placeholder, label, frame_index, depth_m.shape)
                object_chunks_by_placeholder[placeholder].append(
                    _camera_frame_to_world_pc(
                        camera_frame=camera_frame,
                        camera_matrix=camera_matrix,
                        t_world_from_cam=calib[label].t_world_from_cam,
                        mask=mask,
                        min_depth_m=min_depth_m,
                        max_depth_m=max_depth_m,
                    )
                )

        scene_point_clouds.append(
            deterministic_resample(
                _crop_point_cloud_by_bounds(merge_point_clouds(scene_chunks), workspace_crop_bounds),
                int(scene_point_num),
            )
        )
        for placeholder in placeholders:
            object_point_clouds[placeholder].append(
                deterministic_resample(
                    _crop_point_cloud_by_bounds(
                        merge_point_clouds(object_chunks_by_placeholder[placeholder]),
                        workspace_crop_bounds,
                    ),
                    int(object_point_num),
                )
            )

    data_dir = ensure_dir(output_dir / "data")
    hdf5_path = data_dir / f"episode{int(episode_index)}.hdf5"
    with h5py.File(hdf5_path, "w") as root:
        joint_group = root.create_group("joint_action")
        vector = np.asarray(joint_vectors, dtype=np.float32)
        joint_group.create_dataset("vector", data=vector)
        joint_group.create_dataset("left_arm", data=vector[:, :6])
        joint_group.create_dataset("left_gripper", data=vector[:, 6])
        joint_group.create_dataset("right_arm", data=vector[:, 7:13])
        joint_group.create_dataset("right_gripper", data=vector[:, 13])
        root.create_dataset("pointcloud", data=np.asarray(scene_point_clouds, dtype=np.float32))

        obs_group = root.create_group("observation")
        for label in labels:
            cam_group = obs_group.create_group(label)
            cam_group.create_dataset("rgb", data=np.asarray(observation_by_camera[label]["rgb"], dtype=np.uint8))
            cam_group.create_dataset("depth", data=np.asarray(observation_by_camera[label]["depth"], dtype=np.float32))
            cam_group.create_dataset(
                "intrinsic_cv",
                data=np.asarray(intrinsic_by_camera.get(label, calib[label].camera_matrix), dtype=np.float32),
            )
            cam_group.create_dataset("extrinsic_cv", data=calib[label].t_world_from_cam.astype(np.float32))
            cam_group.create_dataset("cam2world_gl", data=calib[label].t_world_from_cam.astype(np.float32))

        if placeholders:
            obj_group = root.create_group("object_pointcloud")
            for placeholder, chunks in object_point_clouds.items():
                obj_group.create_dataset(str(placeholder), data=np.asarray(chunks, dtype=np.float32))

    scene_info_path = output_dir / "scene_info.json"
    scene_info = {
        f"episode_{int(episode_index)}": {
            "info": {placeholder: prompt for placeholder, prompt in object_prompts.items()},
            "object_pointcloud": {
                "target_source": "real_sam_mask",
                "point_num": int(object_point_num),
                "targets": {
                    placeholder: {
                        "prompt": prompt,
                        "source": "real_mask",
                    }
                    for placeholder, prompt in object_prompts.items()
                },
            },
            "raw_episode": str(raw_episode_dir),
            "calibration_path": str(calib_path),
            "frame_mode": str(frame_mode),
            "workspace_crop_bounds_m": {} if workspace_crop_bounds is None else workspace_crop_bounds.as_dict(),
        }
    }
    if scene_info_path.exists():
        existing = read_json(scene_info_path)
        existing.update(scene_info)
        scene_info = existing
    write_json(scene_info_path, scene_info)
    return hdf5_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real three-ZED raw episode to RoboTwin-compatible HDF5.")
    parser.add_argument("--raw_episode_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--calibration_path", default="")
    parser.add_argument("--camera_labels", type=str, default="")
    parser.add_argument("--object_prompts", type=str, default="")
    parser.add_argument("--mask_root", type=str, default="")
    parser.add_argument("--scene_point_num", type=int, default=1024)
    parser.add_argument("--object_point_num", type=int, default=1024)
    parser.add_argument("--min_depth_m", type=float, default=0.05)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--frame_mode", type=str, default="reference_camera")
    parser.add_argument("--workspace_crop_x_min", type=float, default=None)
    parser.add_argument("--workspace_crop_x_max", type=float, default=None)
    parser.add_argument("--workspace_crop_y_min", type=float, default=None)
    parser.add_argument("--workspace_crop_y_max", type=float, default=None)
    parser.add_argument("--workspace_crop_z_min", type=float, default=None)
    parser.add_argument("--workspace_crop_z_max", type=float, default=None)
    parser.add_argument("--workspace_crop_margin_px", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.camera_labels.split(",") if item.strip()]
    prompts = parse_object_prompts(args.object_prompts)
    crop_values = [
        args.workspace_crop_x_min,
        args.workspace_crop_x_max,
        args.workspace_crop_y_min,
        args.workspace_crop_y_max,
        args.workspace_crop_z_min,
        args.workspace_crop_z_max,
    ]
    if any(value is not None for value in crop_values):
        if any(value is None for value in crop_values):
            raise ValueError("All six workspace crop bounds must be provided together.")
        workspace_crop_bounds = WorkspaceBounds(
            x_min=float(args.workspace_crop_x_min),
            x_max=float(args.workspace_crop_x_max),
            y_min=float(args.workspace_crop_y_min),
            y_max=float(args.workspace_crop_y_max),
            z_min=float(args.workspace_crop_z_min),
            z_max=float(args.workspace_crop_z_max),
        )
    else:
        workspace_crop_bounds = None
    hdf5_path = postprocess_episode(
        raw_episode_dir=args.raw_episode_dir,
        output_dir=args.output_dir,
        episode_index=args.episode_index,
        calibration_path=args.calibration_path,
        camera_labels=labels,
        object_prompts=prompts,
        mask_root=args.mask_root,
        scene_point_num=args.scene_point_num,
        object_point_num=args.object_point_num,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        frame_mode=args.frame_mode,
        workspace_crop_bounds=workspace_crop_bounds,
        workspace_crop_margin_px=args.workspace_crop_margin_px,
    )
    print(hdf5_path)


if __name__ == "__main__":
    main()
