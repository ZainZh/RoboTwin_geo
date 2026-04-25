#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from script.real_zed_collection.real_zed_utils import (
    deterministic_resample,
    load_three_zed_calibration,
    merge_point_clouds,
    read_json,
    transform_point_cloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one merged point cloud from a raw real_zed_collection episode.")
    parser.add_argument("--raw_episode_dir", required=True, type=str)
    parser.add_argument("--frame_index", default=0, type=int)
    parser.add_argument("--calibration_path", default="", type=str)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--min_depth_m", default=0.05, type=float)
    parser.add_argument("--max_depth_m", default=3.0, type=float)
    parser.add_argument("--point_num", default=800000, type=int, help="<=0 keeps all points.")
    parser.add_argument("--x_min", default=None, type=float)
    parser.add_argument("--x_max", default=None, type=float)
    parser.add_argument("--y_min", default=None, type=float)
    parser.add_argument("--y_max", default=None, type=float)
    parser.add_argument("--z_min", default=None, type=float)
    parser.add_argument("--z_max", default=None, type=float)
    return parser.parse_args()


def _load_frame_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _rgb_to_depth_shape(rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == depth_shape:
        return rgb
    if cv2 is None:
        raise RuntimeError("cv2 is required when rgb/depth shapes differ.")
    return cv2.resize(rgb, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)


def _frame_to_point_cloud(
    camera_frame: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth_m = (
        np.asarray(camera_frame["depth_m"], dtype=np.float32)
        if "depth_m" in camera_frame
        else np.asarray(camera_frame["depth_mm"], dtype=np.float32) / 1000.0
    )
    rgb = _rgb_to_depth_shape(np.asarray(camera_frame["rgb"], dtype=np.uint8), depth_m.shape)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    valid = np.isfinite(depth_m) & (depth_m > float(min_depth_m)) & (depth_m < float(max_depth_m))
    ys, xs = np.where(valid)
    if xs.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
    y = (ys.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[ys, xs, :3].astype(np.float32) / 255.0
    return np.concatenate([xyz, colors], axis=1).astype(np.float32)


def _apply_workspace_crop(point_cloud: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    mask = np.ones((len(pc),), dtype=bool)
    if args.x_min is not None:
        mask &= pc[:, 0] >= float(args.x_min)
    if args.x_max is not None:
        mask &= pc[:, 0] <= float(args.x_max)
    if args.y_min is not None:
        mask &= pc[:, 1] >= float(args.y_min)
    if args.y_max is not None:
        mask &= pc[:, 1] <= float(args.y_max)
    if args.z_min is not None:
        mask &= pc[:, 2] >= float(args.z_min)
    if args.z_max is not None:
        mask &= pc[:, 2] <= float(args.z_max)
    return pc[mask]


def _write_ply(path: Path, point_cloud: np.ndarray) -> None:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 6:
        raise ValueError(f"Expected Nx6 point cloud, got {pc.shape}")

    xyz = pc[:, :3].astype(np.float32)
    rgb = np.clip(np.round(pc[:, 3:6] * 255.0), 0, 255).astype(np.uint8)

    vertex = np.empty(
        len(pc),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(vertex.tobytes())


def _resolve_calibration_path(raw_episode_dir: Path, manifest: dict, override: str) -> Path:
    if str(override).strip():
        calib_path = Path(override).expanduser().resolve()
    else:
        snapshot_rel = str(manifest.get("calibration_snapshot_path", "")).strip()
        if snapshot_rel:
            calib_path = (raw_episode_dir / snapshot_rel).resolve()
        else:
            source_path = str(manifest.get("calibration_path", "")).strip()
            if not source_path:
                raise ValueError("Missing calibration path. Pass --calibration_path or include it in manifest.json.")
            calib_path = Path(source_path).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {calib_path}")
    return calib_path


def main() -> None:
    args = parse_args()
    raw_episode_dir = Path(args.raw_episode_dir).expanduser().resolve()
    manifest = read_json(raw_episode_dir / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {raw_episode_dir / 'manifest.json'}")

    frame_index = int(args.frame_index)
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"frame_index out of range: {frame_index}, available [0, {len(frames) - 1}]")

    calibration_path = _resolve_calibration_path(raw_episode_dir, manifest, args.calibration_path)
    calib = load_three_zed_calibration(calibration_path)

    frame = frames[frame_index]
    labels = [str(x) for x in manifest.get("camera_labels", list(calib.keys()))]
    chunks: list[np.ndarray] = []
    per_camera_counts: dict[str, int] = {}
    for label in labels:
        camera_rel = str(frame["cameras"][label])
        camera_frame = _load_frame_npz(raw_episode_dir / camera_rel)
        camera_matrix = (
            np.asarray(camera_frame["camera_matrix"], dtype=np.float32).reshape(3, 3)
            if "camera_matrix" in camera_frame
            else calib[label].camera_matrix.astype(np.float32)
        )
        pc_cam = _frame_to_point_cloud(
            camera_frame=camera_frame,
            camera_matrix=camera_matrix,
            min_depth_m=float(args.min_depth_m),
            max_depth_m=float(args.max_depth_m),
        )
        pc_world = transform_point_cloud(pc_cam, calib[label].t_world_from_cam)
        chunks.append(pc_world)
        per_camera_counts[label] = int(len(pc_world))

    merged = merge_point_clouds(chunks)
    cropped = _apply_workspace_crop(merged, args)
    exported = deterministic_resample(cropped, int(args.point_num)) if int(args.point_num) > 0 else cropped

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if str(args.output_path).strip()
        else raw_episode_dir / f"merged_frame_{frame_index:06d}.ply"
    )
    _write_ply(output_path, exported)

    print(f"[INFO] Episode: {raw_episode_dir}")
    print(f"[INFO] Frame index: {frame_index}")
    print(f"[INFO] Calibration: {calibration_path}")
    print(f"[INFO] Per-camera points: {per_camera_counts}")
    print(f"[INFO] Merged points before crop: {len(merged)}")
    print(f"[INFO] Merged points after crop: {len(cropped)}")
    print(f"[INFO] Exported points: {len(exported)}")
    print(f"[INFO] Saved PLY: {output_path}")


if __name__ == "__main__":
    main()
